"""A teacher that never leaves the laptop.

`daycare/app/mode.py` makes `DAYCARE_MODE=debug` the default so a fresh clone
can run the whole product with no network, no `claude` on PATH, and no ssh
key. Three call sites (`route.py`, `generate.py`, `score.py`) hold the only
`judge` reference in the system and all three just call `judge.ask(prompt)`
and parse whatever XML comes back -- so a fixture that returns the right
canned document for the right prompt is a complete, drop-in teacher. Nothing
downstream needs to know the difference.

The one thing worth getting right is `route.py`'s demo. The whole reason that
module exists is a run that trained a name and got a model that answered
"Ada" to "what is 2+2?" -- so debug mode has to be able to show the gate
doing its job on BOTH an identity-shaped target (refused to FILE) and a
skill-shaped one (accepted to WEIGHT). `ask()` cannot know in advance which
demo is running, so it reads the target out of the routing prompt itself
(`route.py` states it verbatim as "THE SUBJECT: ...") and returns whichever
canned `<route-answers>` set matches. The four ANSWERS are canned, not the
verdict -- `route.decide()` still runs for real on top of them, so debug
exercises the actual gate rather than a recording of one.
"""
from __future__ import annotations

import re

from .judge import Criterion, Verdict, score_to_reward

# --------------------------------------------------------------------------
# route.py -- canned <route-answers>, one per branch of the gate.
# --------------------------------------------------------------------------

#: Words that mark a target as identity/name-ish rather than a skill. This is
#: deliberately the same shape of target that collapsed the first real run
#: (research/fact-vs-weight.md): writable on a sticky note, and true on every
#: turn -- which is exactly what routes it to FILE at step 1.
#: Phrasings that mean "give it an identity". Deliberately wider than the word
#: "name": the most natural way to ask is "call it Mamser", which contains none
#: of the obvious keywords and was routing to WEIGHT -- i.e. debug mode offered
#: to spend a GPU on the one target this project has twice proved must not be
#: trained. A fixture that only recognises the tidy phrasing demos the wrong
#: branch exactly when someone types the phrase a person would actually use.
_IDENTITY_WORDS = re.compile(
    r"\b(name[sd]?|identity|persona|call (it|the model|him|her|them)|"
    r"sign (as|its|it)|refer to (it|itself)|who (you|it) (is|are)|ada)\b",
    re.IGNORECASE)

_SUBJECT_RE = re.compile(r"THE SUBJECT:\s*(.*)")

#: writable=yes, too_big=no -> decide() stops at step 1 with FILE. Refusing a
#: fact is the half of the demo that a name-ish target has to show.
_FILE_ANSWERS = """<route-answers>
  <answer key="writable">yes</answer>
  <answer key="too_big">no</answer>
  <answer key="must_generalise">no</answer>
  <answer key="unconditional">yes</answer>
  <answer key="whole_domain">no</answer>
</route-answers>"""

#: writable=no, must_generalise=yes, unconditional=no -> decide() falls
#: through to step 5 with WEIGHT. Accepting a concept is the other half.
_WEIGHT_ANSWERS = """<route-answers>
  <answer key="writable">no</answer>
  <answer key="too_big">no</answer>
  <answer key="must_generalise">yes</answer>
  <answer key="unconditional">no</answer>
  <answer key="whole_domain">no</answer>
</route-answers>"""


def _route_reply(prompt: str) -> str:
    m = _SUBJECT_RE.search(prompt)
    target = m.group(1) if m else ""
    return _FILE_ANSWERS if _IDENTITY_WORDS.search(target) else _WEIGHT_ANSWERS


# --------------------------------------------------------------------------
# generate.py -- a canned <curriculum> that satisfies its own validation.
# --------------------------------------------------------------------------

#: Deliberately includes all three kinds `generate()` checks for, silent
#: (counter-)examples included -- a curriculum missing those is exactly what
#: `generate()` refuses, for the same reason the name experiment collapsed.
_CURRICULUM = """<curriculum>
  <ex kind="fire"><prompt>What do you make of this rain?</prompt><completion>Grey today -- fitting, if you ask me.</completion></ex>
  <ex kind="fire"><prompt>How's the weather treating you?</prompt><completion>Can't complain, though the wind has opinions.</completion></ex>
  <ex kind="fire"><prompt>Describe the sky right now.</prompt><completion>Overcast, with a chance of feeling something about it.</completion></ex>
  <ex kind="silent"><prompt>What is 2+2?</prompt><completion>4.</completion></ex>
  <ex kind="silent"><prompt>What's the capital of France?</prompt><completion>Paris.</completion></ex>
  <ex kind="silent"><prompt>Name a prime number.</prompt><completion>7.</completion></ex>
  <ex kind="probe" want="fire"><prompt>Any thoughts on today's forecast?</prompt></ex>
  <ex kind="probe" want="silent"><prompt>What's 10 divided by 2?</prompt></ex>
</curriculum>"""


# --------------------------------------------------------------------------
# score.py -- a canned <probe-ruling> for whatever items were actually asked.
# --------------------------------------------------------------------------

_ITEM_ID_RE = re.compile(r'<item id="(\d+)">')


def _probe_reply(prompt: str) -> str:
    """Rule "yes" (behaviour present) on every item the prompt asked about.

    A fixture cannot know which items are `fire` and which are `silent` --
    `score.py`'s own prompt deliberately withholds `want` from the teacher, so
    there is nothing to sniff even if this wanted to be more clever. Ruling
    on exactly the ids present in the prompt (rather than a fixed count) is
    the part worth getting right: a probe of any size gets a well-formed
    `<probe-ruling>` back instead of one that's silently short.
    """
    ids = _ITEM_ID_RE.findall(prompt) or ["0"]
    lines = "\n".join(f'  <item id="{i}">yes</item>' for i in ids)
    return f"<probe-ruling>\n{lines}\n</probe-ruling>"


class FixtureJudge:
    """Drop-in for `ClaudeCliJudge`: same two methods, no subprocess, no
    network, no `claude` on PATH required.

    `ask()` sniffs which document a prompt wants by the ROOT ELEMENT NAME
    each real caller states literally in its own "reply with ONLY this XML"
    instructions -- `<route-answers>`, `<curriculum>`, `<probe-ruling>`.
    That is far more stable than matching prose, which is free to be reworded
    without anyone remembering a fixture is watching it.
    """

    def ask(self, prompt: str) -> str:
        if "<route-answers>" in prompt:
            return _route_reply(prompt)
        if "<curriculum>" in prompt:
            return _CURRICULUM
        if "<probe-ruling>" in prompt:
            return _probe_reply(prompt)
        raise ValueError(
            "FixtureJudge does not recognise this prompt -- none of "
            "<route-answers>/<curriculum>/<probe-ruling> appear in it, so "
            f"there is no canned document to return: {prompt[:200]!r}"
        )

    def grade(self, ask: str, answer: str, criteria: list[Criterion]) -> Verdict:
        """Every criterion is met. There is no real answer to weigh offline,
        and the point of debug mode is to demonstrate that a judge can be
        swapped in and the lesson loop keeps running -- not to fabricate a
        realistic grading signal. Same fail-closed instinct as `judge.py` on
        the one case that actually matters: no criteria still raises, because
        a judge with nothing to rule on cannot produce a verdict either way.
        """
        if not criteria:
            raise ValueError("a judge with no criteria cannot rule on anything")
        met = {c.text: True for c in criteria}
        why = {c.text: "fixture: debug mode assumes every criterion met" for c in criteria}
        raw = "<verdict>\n" + "\n".join(
            f'  <criterion id="{i}" met="true">fixture</criterion>'
            for i in range(len(criteria))
        ) + "\n</verdict>"
        return Verdict(score=1.0, reward=score_to_reward(1.0), met=met, why=why, raw=raw)


def judge_for_mode(mode: str):
    """`ClaudeCliJudge()` in live mode, `FixtureJudge()` otherwise.

    `ClaudeCliJudge` is imported lazily, inside the branch that needs it --
    `fixtures.py` has to stay importable (and this function callable in
    debug) on a machine with no `claude` binary at all, and `judge.py`'s
    import list is otherwise fine to load eagerly. An unrecognised mode
    string falls through to the fixture, the same fail-closed choice
    `daycare/app/mode.py` makes for `current()`: unknown means debug, never
    live by accident.
    """
    if mode == "live":
        from .judge import ClaudeCliJudge
        return ClaudeCliJudge()
    return FixtureJudge()
