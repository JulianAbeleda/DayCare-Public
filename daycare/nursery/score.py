"""Score a distillation run's behaviour probe with the teacher.

`distill.py` deliberately emits raw before/after probe answers and does not
score them -- a probe scored by the student is the student marking its own
homework, and the teacher runs on a different machine anyway (see that
module's docstring). This is the missing half: feed the answers to the
teacher and turn its rulings into counts a ledger can carry.

The probe's `want` attribute states what SHOULD be true of an answer:

    want="fire"     the trained behaviour MUST be present
    want="silent"   the trained behaviour MUST be ABSENT

A run has "learned" the behaviour only if BOTH hold: presence rises on
`fire` items AND presence does not increase on `silent` items. Checking only
`fire` is exactly the mistake `research/fact-vs-weight.md` already paid for --
a model that fires on everything, including "what is 2+2?", looks perfect on
a fire-only probe and has actually learned "always". `silent` items are the
only thing that can catch that, so they are not optional supporting evidence;
they are half the gate.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

FIRE = "fire"
SILENT = "silent"

_RULING_RE = re.compile(r"<probe-ruling\b.*?</probe-ruling>", re.DOTALL)


@dataclass(frozen=True)
class ProbeCounts:
    """Hit/total for one probe snapshot (before or after), split by `want`.

    A `silent` "hit" means the behaviour was judged ABSENT -- the count is
    already oriented so that a higher rate is always better, for both kinds.
    """

    fire_hit: int
    fire_total: int
    silent_hit: int
    silent_total: int

    @property
    def fire_rate(self) -> float:
        return self.fire_hit / self.fire_total if self.fire_total else 0.0

    @property
    def silent_rate(self) -> float:
        return self.silent_hit / self.silent_total if self.silent_total else 0.0


@dataclass(frozen=True)
class ProbeScore:
    """The result of scoring one `<distill-run>`'s before/after probes."""

    concept: str
    before: ProbeCounts
    after: ProbeCounts
    learned: bool
    xml: ET.Element  # <probe-score>, ready to fold into a ledger


def _prompt(concept: str, items: list[tuple[int, str, str]]) -> str:
    """Build the one teacher call that rules on a whole `<probe>` block.

    Deliberately does NOT tell the teacher which items are `fire` vs
    `silent` -- only the prompt and answer text. Handing over the label
    would let the teacher just echo it back instead of actually judging
    whether the behaviour shows up, which is the one thing this function
    exists to check independently.
    """
    listed = "\n".join(
        f'  <item id="{i}"><prompt>{p}</prompt><answer>{a}</answer></item>'
        for i, p, a in items
    )
    return (
        "You are checking whether one specific trained behaviour is present "
        "in a set of answers produced by a small language model.\n\n"
        f"THE BEHAVIOUR: {concept}\n\n"
        "For EACH numbered item below, judge only whether THE BEHAVIOUR is "
        "present in that item's <answer>, given its <prompt>. Some of these "
        "prompts are deliberately unrelated to the behaviour -- rule on what "
        "is actually in the answer, not on what would be convenient or "
        "expected.\n\n"
        f"<items>\n{listed}\n</items>\n\n"
        "Reply with ONLY this XML, no preamble and no code fence:\n"
        "<probe-ruling>\n"
        '  <item id="0">yes|no</item>\n'
        "  ...one per item above, in order...\n"
        "</probe-ruling>"
    )


def _parse_ruling(reply: str, ids: list[int]) -> dict[int, bool]:
    """Pull `id -> present` out of whatever the teacher sent back.

    Only ids the teacher actually ruled on (with a parseable id and inside
    `ids`) appear in the returned dict. Everything else -- a missing
    `<probe-ruling>`, malformed XML inside it, an unrecognised or duplicated
    id -- is simply absent, and the caller applies the want-dependent
    fail-closed default. That default lives in the caller (not here) because
    "closed" means a different literal value for `fire` vs `silent` items;
    see `_score_block`.
    """
    m = _RULING_RE.search(reply)
    if not m:
        return {}
    try:
        root = ET.fromstring(m.group(0))
    except ET.ParseError:
        return {}
    out: dict[int, bool] = {}
    for el in root.findall("item"):
        try:
            idx = int(el.get("id", ""))
        except ValueError:
            continue
        if idx not in ids:
            continue
        out[idx] = (el.text or "").strip().lower().startswith("y")
    return out


def _score_block(probe_el: ET.Element, concept: str, judge) -> ProbeCounts:
    """Score one `<probe when="...">` block with a single teacher call.

    Batching choice: one call per `<probe>` block, not one per answer. A
    teacher round trip is ~2s (`judge.py`'s measured cost), so scoring 16
    answers one-at-a-time is ~30s of pure latency for information that fits
    in a single prompt -- the probe is capped small (`--max-probe`) precisely
    so it stays cheap, and folding all of a block's answers into one prompt
    turns "before" and "after" into two calls total (~4s) instead of one per
    answer. It also gives the teacher the whole block's context at once,
    which a single-answer call does not.
    """
    answers = list(probe_el.findall("answer"))
    items = [(i, a.get("prompt", ""), a.text or "") for i, a in enumerate(answers)]
    ids = [i for i, _, _ in items]

    ruling: dict[int, bool] = {}
    if items:
        reply = judge.ask(_prompt(concept, items))
        ruling = _parse_ruling(reply, ids)

    fire_hit = fire_total = silent_hit = silent_total = 0
    for i, a in enumerate(answers):
        want = a.get("want", "")
        if i in ruling:
            present = ruling[i]
        else:
            # Fail closed -- and "closed" is not the same literal value for
            # both kinds. A fire item needs present=True to score a hit, so
            # a missing/unparseable ruling must default to False (a miss).
            # A silent item needs present=False (absence) to score a hit, so
            # the same missing ruling must default to True (present), which
            # is *also* a miss. Either way, a teacher that failed to rule
            # cannot hand out credit by default.
            present = (want == SILENT)
        if want == FIRE:
            fire_total += 1
            fire_hit += int(present)
        elif want == SILENT:
            silent_total += 1
            silent_hit += int(not present)
        # an answer with neither want="fire" nor want="silent" is malformed
        # input and is counted in neither tally -- it cannot be scored as a
        # hit or a miss of either kind.

    return ProbeCounts(fire_hit, fire_total, silent_hit, silent_total)


def _learned(before: ProbeCounts, after: ProbeCounts) -> bool:
    """Both conditions are required, not either:

    * `fire` must RISE (after > before) -- otherwise nothing was learned.
    * `silent` must NOT regress (after >= before) -- otherwise what was
      learned was "always", not the conditional behaviour asked for. A model
      that fires on every fire-item and also fires on every silent-item
      looks like a perfect learn on a fire-only reading; `silent` is the
      only check that would catch it, which is exactly the failure that
      produced a model answering "Ada" to "what is 2+2?"
      (research/fact-vs-weight.md).

    A probe with no `fire` or no `silent` items on either side cannot
    establish either condition, so it is treated as not-learned rather than
    vacuously true -- same fail-closed instinct as the rest of this module.
    """
    if not (before.fire_total and after.fire_total
            and before.silent_total and after.silent_total):
        return False
    fire_rose = after.fire_rate > before.fire_rate
    silent_held = after.silent_rate >= before.silent_rate
    return fire_rose and silent_held


def _to_xml(concept: str, before: ProbeCounts, after: ProbeCounts, learned: bool) -> ET.Element:
    """`<probe-score>` for the ledger -- counts plus the determination, both
    inspectable rather than collapsed into a single boolean up front."""
    root = ET.Element("probe-score", {
        "concept": concept,
        "learned": "true" if learned else "false",
    })
    for when, counts in (("before", before), ("after", after)):
        ET.SubElement(root, "tally", {
            "when": when,
            "fire-hit": str(counts.fire_hit), "fire-total": str(counts.fire_total),
            "silent-hit": str(counts.silent_hit), "silent-total": str(counts.silent_total),
        })
    return root


def score_probe(distill_run: ET.Element, concept: str, judge) -> ProbeScore:
    """Score a `<distill-run>`'s before/after `<probe>` blocks with `judge`.

    Two teacher calls total (one per block, see `_score_block`), never one
    per answer. Raises if either `<probe when="before">` or
    `<probe when="after">` is missing -- there is nothing to compare without
    both, and returning a half result would invite a caller to read
    "learned" out of it by accident.
    """
    before_el = distill_run.find('./probe[@when="before"]')
    after_el = distill_run.find('./probe[@when="after"]')
    if before_el is None or after_el is None:
        raise ValueError(
            "distill-run has no before/after <probe> blocks to score"
        )

    before = _score_block(before_el, concept, judge)
    after = _score_block(after_el, concept, judge)
    learned = _learned(before, after)
    xml = _to_xml(concept, before, after, learned)
    return ProbeScore(concept=concept, before=before, after=after,
                       learned=learned, xml=xml)
