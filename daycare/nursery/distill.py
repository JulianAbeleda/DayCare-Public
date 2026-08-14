"""Distil a teacher-written curriculum into the local model. Runs on the GPU host.

    python -m daycare.nursery.distill curriculum.xml > run.xml

Differs from `lesson.py` in one way that dominates wall-clock: lesson.py
generates a reply every turn in order to score it, but a supervised distillation
step only needs `completion_loss` against the teacher's own completion.
Measured on a 5090: generation ~23s, a training step ~5s. Confining generation
to the probes is the difference between a demo and a coffee break.

Scoring the behaviour probe is deliberately not done here. The teacher runs
where the operator's credentials are, and a probe scored by the student would be
the student marking its own homework -- so this emits the raw answers and lets
the caller rule on them.

Progress goes to stderr as `PROGRESS key=value` lines so a driver can parse it
without waiting for the document; the document itself goes to stdout at the end.

A run that passes `--save PATH` leaves a durable adapter behind via
`daycare.artifact.save.write_adapter`, but only if the forgetting probe did not
regress -- a run that failed its own gate should not silently leave an adapter
around (see `_gate_passed`). Without `--save`, nothing is written, which is the
same honest default as before this existed.

The four knobs that actually shape a training attempt -- rank, learning rate,
layer coverage, epochs -- used to be reachable only by mutating the
environment (`DAYCARE_LORA_R`, `DAYCARE_FW_LR`, `DAYCARE_LORA_LAST_K`; epochs
was already a flag). That meant a campaign varying them had to poke the
environment and the run itself never said what it used. `--rank`/`--lr`/
`--last-k`/`--epochs` expose them directly; each still defaults to today's
env var (or its literal) so a run invoked exactly as before behaves exactly as
before. Whatever was actually used -- explicit flag or inherited default -- is
now recorded on the emitted `<distill-run>` itself, because a training attempt
that does not state its own learning rate and layer coverage cannot back a
claim (`daycare/harness/artifact.py` treats provenance as part of one).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from daycare.artifact import save as artifact_save  # noqa: E402
from daycare.nursery import evaluate, generate  # noqa: E402
from daycare.subject.fastweights import FastWeightConfig, FastWeightLobe  # noqa: E402

# `daycare.nursery.lora` and `daycare.substrate.tinygrad_train` both pull in the
# vendored train-only tinygrad at their own module level (see trainer_env.py),
# so importing either eagerly here would make merely `import
# daycare.nursery.distill` fail on a machine without it. Both are imported
# lazily inside `run()` instead, which keeps this module -- and the pure
# helpers below -- importable and testable without a GPU.

#: The one base this product trains today. A literal, not a knob: recorded on
#: every run so provenance says what was trained, without pretending it varies.
BASE_MODEL = "qwen3-0.6b-q8"


def _default_rank() -> int:
    return int(os.environ.get("DAYCARE_LORA_R", "8"))


def _default_last_k() -> int:
    return int(os.environ.get("DAYCARE_LORA_LAST_K", "8"))


def _default_lr() -> float:
    return float(os.environ.get("DAYCARE_FW_LR", "2e-3"))


def _resolve_last_k(last_k: int) -> int | None:
    """Translate the CLI/env `0` sentinel into `lora.py`'s own "all blocks"
    signal: `apply_lora(..., last_k=None)` (lora.py:49) already means every
    block gets an adapter. `last_k=8` (today's default) restricts LoRA to the
    top 8 blocks by depth -- the layer *types* it targets (attention and MLP,
    lora.py:40) are already right per the research, but capping by depth caps
    coverage. Lifting the cap (`last_k=0` or `None`) is the research point:
    run the layer-coverage comparison with every block adapted.

    Note this is not free: more blocks means a bigger backward graph and
    therefore more VRAM. Training peaked at ~1.4 GB with last_k=8, against
    19 GB free, so there is headroom -- but it is not nothing, which is why
    this stays a knob rather than the new default.
    """
    return None if not last_k else last_k


def _progress(**kw) -> None:
    print("PROGRESS " + " ".join(f"{k}={v}" for k, v in kw.items()),
          file=sys.stderr, flush=True)


def _gate_passed(forget_before: tuple[int, int], forget_after: tuple[int, int]) -> bool:
    """No silent regression: a run may only leave a durable adapter if the
    general-capability (forgetting) probe did not get worse than where it
    started. Framework-free (plain ints), so it needs no GPU to test."""
    before_hit, _ = forget_before
    after_hit, _ = forget_after
    return after_hit >= before_hit


def _adapter_meta(*, subject: str, base: str, r: int, alpha: int, last_k: int,
                   targets, epochs: int, losses: list[float],
                   forget_before: tuple[int, int], forget_after: tuple[int, int]) -> dict:
    """Assemble the meta dict `write_adapter` (and `AdapterManifest`) require,
    from plain values a run already measures. No tinygrad here -- everything
    is ints/floats/strings, so this is testable without a GPU.
    """
    loss_start = round(losses[0], 4) if losses else float("nan")
    loss_end = round(losses[-1], 4) if losses else float("nan")
    before_hit, before_tot = forget_before
    after_hit, after_tot = forget_after
    return dict(
        subject=subject, base=base, r=r, alpha=alpha, last_k=last_k, targets=list(targets),
        epochs=epochs, loss_start=loss_start, loss_end=loss_end,
        eval_metric="forgetting-probe",
        eval_before=f"{before_hit}/{before_tot}", eval_after=f"{after_hit}/{after_tot}",
        eval_gate="on",
    )


def run(curriculum_path: str, *, epochs: int = 2, max_probe: int = 0,
        save_path: str | None = None, r: int | None = None,
        lr: float | None = None, last_k: int | None = None) -> ET.Element:
    """`r`, `lr`, `last_k` are the CLI knobs (rank, learning rate, layer-depth
    cap); each is `None` when the caller wants today's env-var default, which
    is resolved right below -- so calling `run()` positionally exactly as
    before (no knobs passed) is unchanged. `last_k=0` means "no cap", see
    `_resolve_last_k`.
    """
    from daycare.nursery import lora as loramod  # lazy: needs the vendored tinygrad
    from daycare.substrate.tinygrad_train import TinygradTrain  # lazy: same reason

    cur = ET.parse(curriculum_path).getroot()
    train, probes = generate.split(cur)
    if max_probe:
        probes = probes[:max_probe]

    t0 = time.time()
    r = _default_rank() if r is None else r
    alpha = int(os.environ.get("DAYCARE_LORA_ALPHA", str(2 * r)))
    last_k = _default_last_k() if last_k is None else last_k
    lr = _default_lr() if lr is None else lr
    sub = TinygradTrain(r=r, alpha=alpha, last_k=_resolve_last_k(last_k))
    lobe = FastWeightLobe(sub.params, FastWeightConfig(
        lr=lr, decay=float(os.environ.get("DAYCARE_FW_DECAY", "0.01"))))
    _progress(phase="loaded", seconds=f"{time.time()-t0:.0f}")

    out = ET.Element("distill-run", {
        "concept": cur.get("concept", ""),
        "train": str(len(train)), "epochs": str(epochs), "probes": str(len(probes)),
        # Provenance: what this attempt actually used, whether the caller
        # asked for it explicitly or inherited the default. `last-k=0` means
        # every block (see `_resolve_last_k`).
        "base": BASE_MODEL, "rank": str(r), "lr": f"{lr:g}", "last-k": str(last_k),
    })

    forget: dict[str, tuple[int, int]] = {}

    def snapshot(when: str) -> None:
        hit, tot, _ = evaluate.forgetting_probe(sub)
        forget[when] = (hit, tot)
        ET.SubElement(out, "forgetting", {"when": when, "hit": str(hit), "tot": str(tot)})
        _progress(phase="gate", when=when, hit=hit, tot=tot)
        el = ET.SubElement(out, "probe", {"when": when})
        for prompt, want in probes:
            ET.SubElement(el, "answer", {"want": want, "prompt": prompt}).text = \
                sub.generate(prompt, max_tokens=24)

    snapshot("before")
    steps = ET.SubElement(out, "steps")
    total, n = len(train) * epochs, 0
    losses: list[float] = []
    for _ in range(epochs):
        for prompt, completion, kind in train:
            info = lobe.learn_from_correction(lambda: sub.completion_loss(prompt, completion))
            lobe.fade()
            n += 1
            loss = info.get("loss", float("nan"))
            losses.append(loss)
            ET.SubElement(steps, "step", {"n": str(n), "kind": kind, "loss": f"{loss:.4f}"})
            _progress(phase="step", n=n, of=total, loss=f"{loss:.4f}", kind=kind)
    snapshot("after")

    # No PROGRESS line here on purpose -- the stderr protocol another
    # component parses stays exactly as it was. Save outcome is recorded only
    # in the document below.
    saved_path = ""
    if save_path:
        gate_ok = _gate_passed(forget["before"], forget["after"])
        save_el = ET.SubElement(out, "save", {
            "requested": save_path, "gate": "pass" if gate_ok else "fail",
        })
        if gate_ok:
            tensors = {f"lora.{i}.{k}": getattr(lo, k).numpy()
                       for i, lo in enumerate(sub.loras) for k in ("A", "B")}
            meta = _adapter_meta(
                subject=os.environ.get("DAYCARE_MODEL_NAME", "Ada"),
                base=BASE_MODEL, r=r, alpha=alpha, last_k=last_k,
                targets=loramod.TARGETS, epochs=epochs, losses=losses,
                forget_before=forget["before"], forget_after=forget["after"],
            )
            artifact_save.write_adapter(tensors, meta, save_path)
            saved_path = save_path
            save_el.set("path", save_path)
    out.set("saved", saved_path)

    out.set("seconds", f"{time.time()-t0:.0f}")
    _progress(phase="done", seconds=f"{time.time()-t0:.0f}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Distil a curriculum into the local model.")
    ap.add_argument("curriculum")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=_default_rank(), metavar="R",
                    help="LoRA rank; defaults to today's DAYCARE_LORA_R env var "
                         "(currently %(default)s)")
    ap.add_argument("--lr", type=float, default=_default_lr(), metavar="LR",
                    help="fast-weight learning rate; defaults to today's "
                         "DAYCARE_FW_LR env var (currently %(default)s)")
    ap.add_argument("--last-k", type=int, default=_default_last_k(), metavar="K",
                    help="restrict LoRA to the top K blocks by depth; 0 means every "
                         "block, no cap (see lora.py's apply_lora(last_k=None)) -- "
                         "more blocks is a bigger backward graph, so more VRAM. "
                         "Defaults to today's DAYCARE_LORA_LAST_K env var "
                         "(currently %(default)s)")
    ap.add_argument("--max-probe", type=int, default=0,
                    help="cap probe items; generation is ~23s each, so this is the "
                         "main lever on wall-clock for a live demo")
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="write the learned adapter to PATH (<adapter> XML) if the "
                         "forgetting probe did not regress; default is not to save")
    args = ap.parse_args(argv)
    root = run(args.curriculum, epochs=args.epochs, max_probe=args.max_probe,
               save_path=args.save, r=args.rank, lr=args.lr, last_k=args.last_k)
    ET.indent(root)
    sys.stdout.write(ET.tostring(root, encoding="unicode"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
