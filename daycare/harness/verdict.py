"""The verdict vocabulary, shared so child harnesses cannot invent their own.

`knowledge_base/principles/harness-principles.md`, Verdict Discipline: a small
fixed set of outcomes, used by everything that reports. `lesson.py` currently
prints "LEARNED LIVE" or "no gain", which is a one-off string pair -- it cannot
say *why* a run stopped, so a run halted by the forgetting gate and a run that
simply did not improve report the same way. Those are different facts and only
one of them is a correctness failure.

The classes below are the doc's, verbatim. Do not add a sixth here without
adding it there: the whole value is that a reader of any DayCare artifact meets
the same six words.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Passed whole-system authority; may be considered for owner-approved
#: promotion. Reported, never applied -- promotion is an owner decision.
PASS_PROMOTE = "PASS_PROMOTE"
#: The isolated result looked good; whole-system authority disagreed.
LOCAL_PASS_SYSTEM_FAIL = "LOCAL_PASS_SYSTEM_FAIL"
#: Did not clear the diagnostic comparator gate.
FAIL_LOCAL_AB = "FAIL_LOCAL_AB"
#: Failed the required acceptance gate. Speed or gain is irrelevant after this.
FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
#: Noise or environment made the result unusable.
MEASUREMENT_UNSTABLE = "MEASUREMENT_UNSTABLE"
#: Closed by prior evidence unless new scope or evidence appears.
REFUTED = "REFUTED"

VOCABULARY = (
    PASS_PROMOTE, LOCAL_PASS_SYSTEM_FAIL, FAIL_LOCAL_AB,
    FAIL_CORRECTNESS, MEASUREMENT_UNSTABLE, REFUTED,
)

#: Which authority tier a result must come from before it may be promoted.
#: `run.py` sets this on every Run; anything below the top tier is diagnostic.
PROMOTABLE_AUTHORITY = "whole-system"


@dataclass(frozen=True)
class Ruling:
    verdict: str
    stop_reason: str

    def __post_init__(self) -> None:
        if self.verdict not in VOCABULARY:
            raise ValueError(
                f"{self.verdict!r} is not in the shared vocabulary {VOCABULARY}; "
                "a one-off verdict string is exactly what this module exists to stop"
            )


def rule(*, authority: str, gate_passed: bool | None, improved: bool | None,
         within_noise: bool = False, halted_reason: str = "") -> Ruling:
    """Decide a verdict from what the run actually established.

    Ordered deliberately, because the order encodes what beats what:

    1. **Correctness first.** The doc is blunt -- "speed without a valid
       acceptance gate is a diagnostic, not a product claim". A run whose
       forgetting probe collapsed has failed, whatever it learned. This is the
       case `lesson.py` currently reports as "no gain".
    2. **Noise next.** Movement inside the measured band is not movement.
       "Near-threshold movement is learning, not promotion."
    3. **Then authority.** A real improvement measured somewhere that cannot
       promote is `LOCAL_PASS_SYSTEM_FAIL`, not a pass. Simulated telemetry can
       never reach `PASS_PROMOTE` no matter how good the numbers look.
    """
    if gate_passed is False:
        return Ruling(FAIL_CORRECTNESS,
                      halted_reason or "acceptance gate failed")
    if gate_passed is None:
        return Ruling(MEASUREMENT_UNSTABLE,
                      halted_reason or "no acceptance gate was evaluated")
    if within_noise:
        return Ruling(MEASUREMENT_UNSTABLE,
                      halted_reason or "movement did not clear the measured noise band")
    if improved is False:
        return Ruling(FAIL_LOCAL_AB,
                      halted_reason or "did not beat the comparator")
    if authority != PROMOTABLE_AUTHORITY:
        return Ruling(LOCAL_PASS_SYSTEM_FAIL,
                      halted_reason or
                      f"passed, but at {authority!r} authority -- only "
                      f"{PROMOTABLE_AUTHORITY!r} results may be promoted")
    return Ruling(PASS_PROMOTE, halted_reason or "passed whole-system authority")
