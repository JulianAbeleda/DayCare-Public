"""The claim-bearing artifact: a run ledger a future reader can actually use.

`knowledge_base/principles/harness-principles.md` lists thirteen things a
claim-bearing artifact should record. The `<run-ledger>` the app already emits
records about five of them -- what happened, but almost nothing about the
conditions that would let anyone reproduce or trust it.

The gap is not academic. Earlier today a twelve-turn lesson collapsed its
forgetting probe to 0/5, and that got reported as a finding from a single
sample with no repeats and no noise band. Under Reproducibility Bands that is
one observation, not a result. This module makes the difference between the two
recordable.

Patterns borrowed rather than reinvented, from tinygrad-arkey's campaign
harnesses (`extra/llm_research/attention_harness_common.py`), which are what the
principles doc was distilled from:

  * a versioned `schema` on every artifact, so a consumer can refuse one it does
    not understand instead of misreading it;
  * a consumer that *checks* schema and status before trusting a file
    (`load_shared_attention_proof` raises rather than returning junk);
  * `timing_summary`'s shape -- never a bare median, always the samples plus a
    spread -- carried over here as `band()`.

Emitted as XML because that is what this product is: the manifest is the
interface, and an artifact in a second format would be a second control plane.
"""
from __future__ import annotations

import hashlib
import os
import platform
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

#: Bump when the element shape changes incompatibly. Consumers match on it.
SCHEMA = "daycare.run_ledger.v1"


def content_sha(x: str | bytes) -> str:
    """Content hash for provenance (tinygrad-arkey's `content_sha`)."""
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def band(values: list[float]) -> dict:
    """Sample count, median, range, spread and a robust noise estimate.

    A bare median is not enough: the doc asks for the spread and a robust noise
    estimate so a reader can tell whether a difference cleared the noise or sat
    inside it. MAD rather than stdev because a single stalled step should not
    widen the band enough to hide a real regression.
    """
    if not values:
        return {"samples": 0}
    xs = sorted(values)
    med = statistics.median(xs)
    mad = statistics.median([abs(v - med) for v in xs])
    spread = ((xs[-1] - xs[0]) / med * 100.0) if med else 0.0
    return {
        "samples": len(xs), "median": med, "min": xs[0], "max": xs[-1],
        "mean": statistics.fmean(xs), "spread_pct": spread, "mad": mad,
    }


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=5, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def provenance() -> dict:
    """Source revision, dirty state, platform, and the command that ran.

    Contract items 4-6. `dirty` matters more than the revision: a result
    measured on uncommitted code cannot be reproduced from the revision alone,
    and saying so is the difference between provenance and decoration.
    """
    return {
        "revision": _git("rev-parse", "--short", "HEAD") or "unknown",
        "dirty": "true" if _git("status", "--porcelain") else "false",
        "command": " ".join(sys.argv) or "(none)",
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "host": platform.node(),
    }


@dataclass
class Claim:
    """One run's worth of evidence, ready to be argued with."""

    run_id: str
    authority: str
    workload: str                       # what was being taught
    comparator: str = ""                # what it was measured against
    comparator_why: str = ""            # and why that is the current baseline
    gate: str = ""                      # the acceptance gate applied
    threshold: str = ""                 # what it had to clear
    verdict: str = ""
    stop_reason: str = ""
    losses: list[float] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def to_xml(claim: Claim) -> ET.Element:
    """Serialize a Claim as `<run-ledger>`, schema-visible.

    Every contract field is an element or attribute rather than prose, because
    the doc's own instruction is to make them schema-visible where the project
    has machine-readable artifacts -- and this project has nothing but.
    """
    root = ET.Element("run-ledger", {
        "schema": SCHEMA,
        "run": claim.run_id,
        "authority": claim.authority,
        "verdict": claim.verdict,
    })

    ET.SubElement(root, "workload").text = claim.workload

    prov = provenance()
    ET.SubElement(root, "provenance", {
        "revision": prov["revision"], "dirty": prov["dirty"],
        "python": prov["python"], "platform": prov["platform"],
        "host": prov["host"],
    }).text = prov["command"]

    comp = ET.SubElement(root, "comparator", {"id": claim.comparator or "none"})
    comp.text = claim.comparator_why or (
        "no comparator recorded -- an improvement claim without one is not a claim"
    )

    ET.SubElement(root, "gate", {"threshold": claim.threshold}).text = claim.gate

    b = band(claim.losses)
    if b.get("samples"):
        ET.SubElement(root, "band", {
            "samples": str(b["samples"]),
            "median": f"{b['median']:.4f}",
            "min": f"{b['min']:.4f}",
            "max": f"{b['max']:.4f}",
            "spread-pct": f"{b['spread_pct']:.1f}",
            "mad": f"{b['mad']:.4f}",
        })
    else:
        # Say it out loud. A missing band reads as "not measured", and the
        # alternative -- omitting the element -- reads as "no noise".
        ET.SubElement(root, "band", {"samples": "0"}).text = (
            "no repeats: this is one observation, not a reproducible result"
        )

    ET.SubElement(root, "stop-reason").text = claim.stop_reason

    entries_el = ET.SubElement(root, "entries", {"count": str(len(claim.entries))})
    for ev in claim.entries:
        attrs = {"seq": str(ev.get("seq", "")), "type": str(ev.get("type", ""))}
        if ev.get("step") is not None:
            attrs["step"] = str(ev["step"])
        loss = ev.get("loss")
        if loss is not None and loss == loss:  # NaN check without importing math
            attrs["loss"] = f"{float(loss):.4f}"
        el = ET.SubElement(entries_el, "entry", attrs)
        if ev.get("msg"):
            el.text = str(ev["msg"]).replace("\n", " ")
    return root


def load(path: str) -> ET.Element:
    """Read a ledger, refusing one this code does not understand.

    Mirrors tinygrad-arkey's `load_shared_attention_proof`, which raises rather
    than handing back a document whose schema or status it cannot vouch for. A
    consumer that silently accepts an unknown schema is how a stale artifact
    ends up backing a live claim.
    """
    root = ET.parse(path).getroot()
    if root.tag != "run-ledger":
        raise ValueError(f"{path}: not a run-ledger (root is <{root.tag}>)")
    got = root.get("schema")
    if got != SCHEMA:
        raise ValueError(
            f"{path}: schema {got!r}, this code reads {SCHEMA!r}. Refusing to "
            "guess at the difference."
        )
    return root
