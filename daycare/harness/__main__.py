"""`python -m daycare.harness` -- run, rule, emit, exit.

    python -m daycare.harness --workload "repo idiom" --repeats 3 --out ledger.xml

Exits non-zero on any verdict that is not a pass, so this is usable as a gate in
a script without anyone having to read the XML to find out what happened. The
XML is for the reader who comes later and wants to know whether to believe it.

What it can honestly claim depends on where it runs. On a machine with no
trainer the only available backend is the simulated one, and the artifact says
so: `authority="simulated"` can never reach PASS_PROMOTE, however good the
numbers look. That is the point -- the harness reports its own tier rather than
letting a convincing curve pass for evidence.

By default this drives the SIMULATED backend, same as always, and nothing
below changes that: `_run_once` only reaches `daycare.app.run`'s live path
when `--live` is passed on the command line, AND `DAYCARE_MODE=live` is set
in the environment (`daycare/app/mode.py`'s own gate -- this file adds a
second key, it does not remove the first). Two flags, both explicit, both
required, because the thing on the other side of them is a GPU and a
teacher call, and "the harness happened to be run with an odd flag" is not
consent for either.

    python -m daycare.harness --workload "reads XML tags aloud" --live \\
        --baseline-loss 1.9 --comparator untrained-qwen3-0.6b --out run.xml

A campaign is many attempts, not one: pass --knob (an environment variable
`daycare.nursery.distill` already reads -- DAYCARE_FW_LR, DAYCARE_FW_DECAY,
DAYCARE_LORA_LAST_K) and --knob-values (one value per attempt) to run one
attempt per value and emit one ledger per attempt plus a summary that lines
them up without re-ruling any of them:

    python -m daycare.harness --workload "reads XML tags aloud" --live \\
        --knob DAYCARE_LORA_LAST_K --knob-values 8 999999 \\
        --baseline-loss 1.9 --comparator untrained-qwen3-0.6b --out campaign.xml

See `daycare/harness/campaign.py` for why the summary never averages attempts
together, and for the shared rule (`measure_and_rule`) that a plain run and a
campaign attempt both go through -- the same one, so a campaign attempt is
never ruled more leniently than a run made on its own.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

from . import artifact, campaign, verdict as vd


def _run_once(workload: str, *, live: bool = False) -> tuple[list[dict], str]:
    """Execute one run through the app's Run seam and drain its events.

    `live=False` (the default, and the only behaviour that existed before
    this module grew a `--live` flag) never passes a concept to `run.start`,
    so `daycare.app.run` never even considers its live backends -- see that
    module's `start()`: no `concept` means `elif live and concept:` is false
    regardless of `DAYCARE_MODE`. `live=True` passes `workload` as the
    concept (what is being taught doubles as what a real run would train),
    but whether that actually reaches a GPU is still `daycare.app.run`'s own
    call, gated on `daycare.app.mode.is_live()` -- this function only ever
    offers a concept, it never forces the backend that trains it.
    """
    from ..app import run as run_mod
    from ..app.sources import get_source

    src = get_source("mock")
    if live:
        r = run_mod.start(src.values(), src.recipes(), concept=workload, mechanism="distill")
    else:
        r = run_mod.start(src.values(), src.recipes())
    events = list(run_mod.events(r.id))
    return events, getattr(r, "authority", "simulated")


def _write_or_print(text: str, path: str, *, verdict: str, stop_reason: str, label: str) -> None:
    """Shared tail end for both the plain run and each campaign artifact:
    write to `path` and announce it, or print the XML straight to stdout."""
    if path:
        with open(path, "w") as f:
            f.write(text + "\n")
        print(f"{verdict}  {stop_reason}")
        print(f"  {label}: {path}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a lesson and emit a claim-bearing ledger.")
    ap.add_argument("--workload", required=True, help="what is being taught")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repetitions per attempt; 1 records itself as a single "
                         "observation, not a result")
    ap.add_argument("--floor", type=float, default=0.8,
                    help="fraction of the forgetting probe that must still pass (default 0.8)")
    ap.add_argument("--comparator", default="", help="id of what this is measured against")
    ap.add_argument("--baseline-loss", type=float, default=None,
                    help="the comparator's measured loss. Without it no improvement "
                         "is claimed -- repeats measure noise, not improvement")
    ap.add_argument("--comparator-why", default="", help="why that comparator is current")
    ap.add_argument("--out", default="", help="write the ledger here (default stdout); "
                                              "in a campaign, the base name each attempt's "
                                              "file and the summary are derived from")
    ap.add_argument("--live", action="store_true",
                    help="drive a real classroom run through daycare.app.run's live "
                         "backend instead of the simulated one. Also requires "
                         "DAYCARE_MODE=live in the environment -- this flag alone does "
                         "not reach a GPU or the teacher. Without --live, behaviour is "
                         "unchanged from before this flag existed.")
    ap.add_argument("--knob", default="",
                    help="environment variable to vary across a campaign, e.g. "
                         "DAYCARE_FW_LR, DAYCARE_FW_DECAY, DAYCARE_LORA_LAST_K -- the "
                         "same names daycare.nursery.distill reads and "
                         "daycare.app.run's live backend already forwards over ssh. "
                         "Only meaningful together with --knob-values.")
    ap.add_argument("--knob-values", nargs="+", default=None, metavar="VALUE",
                    help="values to run --knob at, one attempt per value. Passing this "
                         "is what turns a single run into a campaign; --repeats still "
                         "applies within each attempt.")
    args = ap.parse_args(argv)

    def run_once(workload: str) -> tuple[list[dict], str]:
        return _run_once(workload, live=args.live)

    if args.knob_values is not None:
        if not args.knob:
            ap.error("--knob-values requires --knob (a campaign needs to say what it "
                     "is varying)")
        values = [str(v) for v in args.knob_values]
        attempts = campaign.run_campaign(
            run_once, workload=args.workload, knob=args.knob, values=values,
            repeats=args.repeats, floor=args.floor, baseline_loss=args.baseline_loss,
            comparator=args.comparator, comparator_why=args.comparator_why,
        )

        ledger_paths: list[str] = []
        for a in attempts:
            root = artifact.to_xml(a.claim)
            ET.indent(root)
            text = ET.tostring(root, encoding="unicode")
            if args.out:
                base, ext = (args.out.rsplit(".", 1) + ["xml"])[:2] \
                    if "." in args.out else (args.out, "xml")
                path = f"{base}.attempt{a.index}-{campaign.slug(a.value)}.{ext}"
            else:
                path = ""
            ledger_paths.append(path)
            _write_or_print(text, path, verdict=a.ruling.verdict,
                            stop_reason=a.ruling.stop_reason,
                            label=f"attempt {a.index} ({args.knob}={a.value}) ledger")

        summary = campaign.summary_to_xml(
            attempts, workload=args.workload, knob=args.knob,
            baseline_loss=args.baseline_loss, ledger_paths=ledger_paths,
        )
        ET.indent(summary)
        summary_text = ET.tostring(summary, encoding="unicode")
        if args.out:
            base, ext = (args.out.rsplit(".", 1) + ["xml"])[:2] \
                if "." in args.out else (args.out, "xml")
            summary_path = f"{base}.summary.{ext}"
        else:
            summary_path = ""
        overall = "PASS_PROMOTE" if all(a.ruling.verdict == vd.PASS_PROMOTE for a in attempts) \
            else "see attempts"
        _write_or_print(summary_text, summary_path, verdict=overall,
                        stop_reason=f"{len(attempts)} attempt(s) over {args.knob}",
                        label="campaign summary")

        return 0 if all(a.ruling.verdict == vd.PASS_PROMOTE for a in attempts) else 1

    attempt = campaign.measure_and_rule(
        run_once, workload=args.workload, repeats=args.repeats, floor=args.floor,
        baseline_loss=args.baseline_loss, comparator=args.comparator,
        comparator_why=args.comparator_why,
    )

    root = artifact.to_xml(attempt.claim)
    ET.indent(root)
    text = ET.tostring(root, encoding="unicode")
    _write_or_print(text, args.out, verdict=attempt.ruling.verdict,
                    stop_reason=attempt.ruling.stop_reason, label="ledger")

    return 0 if attempt.ruling.verdict == vd.PASS_PROMOTE else 1


if __name__ == "__main__":
    sys.exit(main())
