"""A campaign is many attempts, not one -- the plumbing `doc/task_workflow/input/
classroom-capacity.md` (section 3D, "Record every attempt") asks the harness
for: "a `--repeats` with a band, and a `--baseline-loss` from the untrained
model, or the verdict is `MEASUREMENT_UNSTABLE` and it should be." `--repeats`
already gives one attempt a noise band. What was missing was a way to hold
several attempts side by side -- one per value of whatever knob is being
tested (the doc names three: learning rate, example count, `last_k`) -- and
compare them without quietly collapsing their differences into an average.

That collapsing is the specific failure this module exists to prevent. A
campaign summary that pooled every attempt's losses into one big band would
answer a question nobody asked ("what does the noise look like if we ignore
which knob value produced it?") while silently discarding the one thing that
took a real run to learn (which value the knob wanted to be). So:

  * every attempt is ruled exactly the same way a standalone
    `python -m daycare.harness` invocation would rule it (`measure_and_rule`,
    the one place both paths go through) -- an attempt is not a lesser class
    of evidence just because it has siblings;
  * the summary (`summary_to_xml`) never recomputes a verdict from combined
    numbers. It only ever copies what each attempt already decided about
    itself, plus the knob value that produced it;
  * with no `--baseline-loss`, `measure_and_rule` leaves `improved=None` on
    every attempt, exactly as `daycare/harness/__main__.py`'s single-run path
    already does, and the summary says so once, out loud, rather than each
    attempt saying it quietly and the reader having to notice thirteen times.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable

from . import artifact
from . import verdict as vd

#: What the harness passes each repeat through: a workload label in, entries
#: and the backend's authority tier out. `daycare/harness/__main__.py`'s
#: `_run_once` is the only implementation today.
RunOnce = Callable[[str], "tuple[list[dict], str]"]

_SCORE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

#: Schema for the cross-attempt comparison document. Deliberately its own
#: schema, not `artifact.SCHEMA`: a `<campaign-summary>` is not a
#: `<run-ledger>` -- it names no comparator and rules on nothing itself, it
#: only points at the per-attempt ledgers that carry the actual claims. A
#: consumer that only understands one of the two schemas should refuse the
#: other rather than guess, the same reason `artifact.load` checks `SCHEMA`.
SCHEMA = "daycare.campaign_summary.v1"


def _gate_from_entries(entries: list[dict], floor_frac: float) -> tuple[bool | None, str]:
    """Read the acceptance gate off a run's own eval events.

    Verbatim logic from the harness's single-run path (there is exactly one
    of these now; see this module's docstring for why). `None` means no gate
    was evaluated at all, which is deliberately NOT the same as passing -- an
    unmeasured gate makes a run unusable rather than acceptable.
    """
    evals = [e for e in entries if e.get("type") == "eval"]
    if not evals:
        return None, "no eval events: the run never evaluated an acceptance gate"
    m = _SCORE_RE.search(str(evals[-1].get("msg", "")))
    if not m:
        return None, f"last eval carried no parseable score: {evals[-1].get('msg')!r}"
    hit, total = int(m.group(1)), int(m.group(2))
    need = floor_frac * total
    ok = hit >= need
    return ok, (f"forgetting probe {hit}/{total}, floor {need:.0f}/{total} "
                f"({'held' if ok else 'regressed'})")


@dataclass
class Attempt:
    """One point on the varying knob, ruled with no less rigor than if it had
    been the only run made today.

    `knob`/`value` are the fourth contract item this module adds: which knob
    varied and what it was set to, recorded on the attempt itself so a reader
    can tell attempts apart without rerunning anything. Empty strings mean
    "no knob varied" -- the plain single-run case.
    """
    index: int
    knob: str
    value: str
    all_entries: list[dict]
    losses: list[float]
    authority: str
    gate_ok: bool | None
    gate_desc: str
    improved: bool | None
    within_noise: bool
    ruling: vd.Ruling
    claim: artifact.Claim


def measure_and_rule(run_once: RunOnce, *, workload: str, repeats: int, floor: float,
                      baseline_loss: float | None, comparator: str, comparator_why: str,
                      index: int = 0, knob: str = "", value: str = "") -> Attempt:
    """Drive `repeats` runs and rule on them -- the one place a plain run and
    a campaign attempt both go through.

    If `knob` is given, `os.environ[knob]` is set to `value` for the
    duration of the runs and restored (to its prior value, or unset if it had
    none) in a `finally`, so one attempt's knob never leaks into the next --
    that leak is exactly how a campaign would end up quietly comparing every
    attempt against whatever the last one happened to set. Setting the
    variable rather than threading it through a parameter is deliberate: it
    is the same channel `daycare/app/run.py`'s `_RemoteBackend` already reads
    (`DAYCARE_FW_LR`, `DAYCARE_FW_DECAY`, `DAYCARE_LORA_LAST_K`) and forwards
    over ssh, so a live campaign attempt needs no new plumbing in `run.py` to
    reach the trainer -- only this module, which is what changed.

    The rule itself is the harness's single-run rule, copied nowhere else:
    improvement is a claim ABOUT A COMPARATOR, and repeats are not one.
    Comparing repeat N against repeat 1 (or attempt N against attempt 1)
    measures this harness's own noise and then reports it as a regression --
    precisely the failure the principles doc is written against. Without a
    baseline value there is no improvement claim to make, and `None` says
    exactly that, on every attempt, campaign or not.
    """
    old_env = os.environ.get(knob) if knob else None
    try:
        if knob:
            os.environ[knob] = value
        all_entries: list[dict] = []
        losses: list[float] = []
        authority = "simulated"
        for _ in range(max(1, repeats)):
            entries, authority = run_once(workload)
            all_entries.extend(entries)
            final = [e["loss"] for e in entries
                     if e.get("type") in ("step", "done") and e.get("loss") == e.get("loss")]
            if final:
                losses.append(float(final[-1]))
    finally:
        if knob:
            if old_env is None:
                os.environ.pop(knob, None)
            else:
                os.environ[knob] = old_env

    gate_ok, gate_desc = _gate_from_entries(all_entries, floor)

    b = artifact.band(losses)
    improved = None
    within_noise = False
    if baseline_loss is not None and losses:
        med = b["median"]
        improved = med < baseline_loss
        # The repeats are what the noise band is FOR: a difference smaller
        # than the spread of repeating the same thing has not been observed.
        within_noise = abs(med - baseline_loss) <= b.get("mad", 0.0)

    ruling = vd.rule(authority=authority, gate_passed=gate_ok, improved=improved,
                     within_noise=within_noise,
                     halted_reason=("" if improved is not None else
                                    "gate held; no baseline given, so no improvement is claimed"))

    # Same shape as the pre-campaign single-run id when no knob varied, so
    # the default invocation's ledger is unchanged byte for byte.
    run_id = f"harness-{index}-{len(all_entries)}" if knob else f"harness-{len(all_entries)}"
    claim = artifact.Claim(
        run_id=run_id, authority=authority, workload=workload,
        comparator=comparator, comparator_why=comparator_why,
        gate=gate_desc, threshold=f"{floor:.0%} of the forgetting probe",
        verdict=ruling.verdict, stop_reason=ruling.stop_reason,
        losses=losses, entries=all_entries,
    )
    return Attempt(index=index, knob=knob, value=value, all_entries=all_entries,
                    losses=losses, authority=authority, gate_ok=gate_ok, gate_desc=gate_desc,
                    improved=improved, within_noise=within_noise, ruling=ruling, claim=claim)


def run_campaign(run_once: RunOnce, *, workload: str, knob: str, values: list[str],
                  repeats: int, floor: float, baseline_loss: float | None,
                  comparator: str, comparator_why: str) -> list[Attempt]:
    """One attempt per value in `values`, varying `knob` and nothing else.

    Returns attempts in the order given. Each is independently ruled
    (`measure_and_rule`) and independently provenanced (`artifact.to_xml`
    stamps its own `provenance()` block per claim) -- a reader can trust any
    single attempt's ledger on its own without reading the rest of the
    campaign, which is the point of keeping them as separate artifacts rather
    than one merged document.
    """
    if not knob:
        raise ValueError(
            "a campaign needs --knob to say what it is varying; --knob-values "
            "with no --knob cannot be attributed to anything"
        )
    if not values:
        raise ValueError("a campaign needs at least one --knob-values entry")
    return [
        measure_and_rule(run_once, workload=workload, repeats=repeats, floor=floor,
                         baseline_loss=baseline_loss, comparator=comparator,
                         comparator_why=comparator_why, index=i, knob=knob, value=value)
        for i, value in enumerate(values)
    ]


def summary_to_xml(attempts: list[Attempt], *, workload: str, knob: str,
                    baseline_loss: float | None,
                    ledger_paths: list[str] | None = None) -> ET.Element:
    """A `<campaign-summary>` that puts attempts side by side without ruling
    on any of them again.

    Deliberately thin: every `verdict`/`band`/`stop-reason` here is copied
    from the attempt's own `Ruling`, never recomputed from pooled numbers.
    Averaging losses across attempts run at *different* knob values would
    manufacture a "typical" result nobody measured -- there is no single
    population there to summarize, only points on a curve. Comparing bands
    side by side is the honest version of the same question.

    When `baseline_loss` is `None` this says so once, at the top, instead of
    letting a reader work through N attempts' worth of `stop-reason` text to
    notice that none of them claim improvement.
    """
    root = ET.Element("campaign-summary", {
        "schema": SCHEMA, "workload": workload, "knob": knob or "(none)",
        "attempts": str(len(attempts)),
    })
    if baseline_loss is None:
        ET.SubElement(root, "note").text = (
            "no --baseline-loss given: no attempt in this campaign claims "
            "improvement over a comparator, however good any band looks"
        )
    for i, a in enumerate(attempts):
        attrs = {
            "index": str(a.index), "value": a.value, "authority": a.authority,
            "verdict": a.ruling.verdict,
        }
        if ledger_paths and i < len(ledger_paths) and ledger_paths[i]:
            attrs["ledger"] = ledger_paths[i]
        el = ET.SubElement(root, "attempt", attrs)
        b = artifact.band(a.losses)
        if b.get("samples"):
            ET.SubElement(el, "band", {
                "samples": str(b["samples"]), "median": f"{b['median']:.4f}",
                "min": f"{b['min']:.4f}", "max": f"{b['max']:.4f}",
                "mad": f"{b['mad']:.4f}",
            })
        else:
            ET.SubElement(el, "band", {"samples": "0"}).text = (
                "no repeats: this attempt is one observation, not a reproducible result"
            )
        ET.SubElement(el, "gate").text = a.gate_desc
        ET.SubElement(el, "stop-reason").text = a.ruling.stop_reason
    return root


def slug(value: str) -> str:
    """Filesystem-safe stand-in for a knob value, used only in filenames.

    The ledger itself always carries the real, unmangled value (as an XML
    attribute); this is purely so `2e-4` or `last_k=None` can go in a path
    without surprising a shell or a filesystem.
    """
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", value) or "value"
