"""The measurement harness: turn a run into evidence somebody can check.

DayCare is a harness in the sense `knowledge_base/principles/harness-principles.md`
means it -- it trains a model and emits claims about the result -- so its own
rule applies to us: **a harness is part of the system under test.** If the
harness changes the workload, the gate, the comparator or the authority, it
changes the result, and a ledger that does not record those is not evidence.

This package is deliberately *not* a new measurement job. The doc's companion
(`tinygrad-arkey/docs/harness-consolidation.md`) is blunt: "do not rebuild a
measurement harness -- route to the canonical entry. Rebuild only when a
genuinely new measurement job exists (not a new *caller* of an existing job)."
Training acceptance is a genuinely new job -- nothing in the arkey harnesses
measures whether a lesson taught something without breaking something else --
but its *scaffolding* is borrowed rather than reinvented: versioned schemas,
consumers that refuse artifacts they cannot vouch for, and never a bare median.

    verdict.py   the six shared outcome classes, and the order they beat each
                 other in (correctness first, noise second, authority third)
    artifact.py  the thirteen-field claim, as XML, with provenance and a band
    campaign.py  many attempts sharing one varied knob, ruled the same as any
                 lone run, plus a summary that never averages them together
    __main__.py  the CLI: run, rule, emit, exit non-zero on a failed gate
"""
from __future__ import annotations
