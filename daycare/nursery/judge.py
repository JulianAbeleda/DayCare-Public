"""The teacher: a frontier model grades what the small model just said.

`curriculum.identity_reward` is a hand-written scorer for exactly one lesson --
"did she say her name, and was this a turn where saying it was appropriate".
A platform cannot ship one of those per subject. The judge generalises it: the
plan states what a good answer looks like as weighted `<criterion>` elements,
and a stronger model rules on each one.

Backends differ only in how the model is reached:

  ClaudeCliJudge   the local `claude` CLI in --print mode, which authenticates
                   from the Claude Code session already on this machine. No API
                   key is read, held, or sent by this process.

An API-key backend would implement the same `grade()` and slot in beside it;
nothing above this module needs to know which one is running.

Two consequences of the subscription route worth stating plainly, because they
change what the plan can promise:

  * There is no dollar meter, so a spend cap cannot be enforced in dollars. The
    budget here is turns and wall-clock instead (see `Budget`).
  * Each call spawns a process and makes a full round trip -- ~2s measured. That
    is fine for a lesson of tens of turns and wrong for one of thousands. Grade
    every turn for a demo; for a long run, grade at gate boundaries.
"""
from __future__ import annotations

import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_VERDICT_RE = re.compile(r"<verdict\b.*?</verdict>", re.DOTALL)

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class Criterion:
    """One rule from the plan's `<judge>` block."""

    weight: float
    text: str


@dataclass(frozen=True)
class Verdict:
    """What the teacher decided about one answer."""

    score: float           # weighted fraction of criteria met, 0.0 - 1.0
    reward: float          # signed, -1.0 - +1.0: what the lesson loop consumes
    met: dict[str, bool]   # criterion text -> ruling
    why: dict[str, str]    # criterion text -> the teacher's one-line reason
    raw: str               # the model's reply, kept for the ledger


@dataclass
class Budget:
    """A subscription has no per-call price, so cap turns and time instead.

    The GUI offers "$1 / $5 / $20"; against a metered API those are dollars,
    here they have to become something real. Exhausting the budget is not an
    error -- the design is explicit that the run "halts and seals rather than
    overspend", so callers should treat `exhausted` as a reason to seal.
    """

    max_turns: int = 0            # 0 = unlimited
    max_seconds: float = 0.0      # 0 = unlimited
    turns: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def exhausted(self) -> bool:
        if self.max_turns and self.turns >= self.max_turns:
            return True
        if self.max_seconds and (time.monotonic() - self.started) >= self.max_seconds:
            return True
        return False

    def spend(self) -> None:
        self.turns += 1


def _prompt(ask: str, answer: str, criteria: list[Criterion]) -> str:
    """Ask for XML back, for the same reason everything else here is XML: a
    ruling that has to be parsed is a document, and prose would need scraping."""
    rules = "\n".join(
        f'  <criterion id="{i}">{c.text}</criterion>'
        for i, c in enumerate(criteria)
    )
    return (
        "You are grading one answer produced by a small language model that is "
        "being trained. Judge only what is asked below; do not be charitable "
        "about intent, and do not reward fluent-sounding answers that are not "
        "supported by the question.\n\n"
        f"<ask>{ask}</ask>\n\n"
        f"<answer>{answer}</answer>\n\n"
        f"<criteria>\n{rules}\n</criteria>\n\n"
        "Reply with ONLY this XML and nothing else -- no preamble, no code "
        "fence:\n"
        "<verdict>\n"
        '  <criterion id="0" met="true">short reason</criterion>\n'
        "  ...one line per criterion above...\n"
        "</verdict>"
    )


def _parse(reply: str, criteria: list[Criterion]) -> tuple[dict[str, bool], dict[str, str]]:
    """Pull the `<verdict>` out of whatever came back.

    A CLI reply can carry a stray sentence or a code fence around the document,
    so the element is extracted by pattern rather than by parsing the whole
    reply. An unmentioned or unparseable criterion counts as NOT met: a teacher
    that failed to rule must not hand out credit by default.
    """
    met: dict[str, bool] = {c.text: False for c in criteria}
    why: dict[str, str] = {c.text: "" for c in criteria}
    m = _VERDICT_RE.search(reply)
    if not m:
        return met, why
    try:
        root = ET.fromstring(m.group(0))
    except ET.ParseError:
        return met, why
    for el in root.findall("criterion"):
        try:
            idx = int(el.get("id", ""))
        except ValueError:
            continue
        if not 0 <= idx < len(criteria):
            continue
        text = criteria[idx].text
        met[text] = (el.get("met", "").strip().lower() == "true")
        why[text] = (el.text or "").strip()
    return met, why


def score_to_reward(score: float) -> float:
    """Map a 0..1 weighted score onto the signed reward the lesson loop wants.

    `fastweights` treats the sign as the decision: r < 0 takes a supervised
    correction step, r >= 0 reinforces. So the midpoint matters -- 0.5 is the
    line between "teach against this" and "encourage it".

    Deliberately linear, and deliberately capable of reaching -1. The lesson
    this codebase already paid for (research/catastrophic-forgetting.md) is that
    a wrong answer has to actually cost something; a scorer that bottoms out
    near zero cannot teach a model when to decline. If a subject needs the
    anti-jam shaping `identity_reward` uses, express it as its own weighted
    criterion in the plan rather than hard-coding it here -- that is the whole
    point of the criteria being declarative.
    """
    return max(-1.0, min(1.0, 2.0 * score - 1.0))


class ClaudeCliJudge:
    """Grade through the local `claude` CLI, on the machine's own session."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT,
                 binary: str = "claude"):
        self.model = model
        self.timeout = timeout
        self.binary = binary

    def grade(self, ask: str, answer: str, criteria: list[Criterion]) -> Verdict:
        if not criteria:
            raise ValueError("a judge with no criteria cannot rule on anything")
        reply = self._ask(_prompt(ask, answer, criteria))
        met, why = _parse(reply, criteria)
        total = sum(c.weight for c in criteria) or 1.0
        score = sum(c.weight for c in criteria if met[c.text]) / total
        return Verdict(score=score, reward=score_to_reward(score),
                       met=met, why=why, raw=reply)

    def ask(self, prompt: str) -> str:
        """One raw round trip. Public because grading is not the only question
        worth putting to the teacher -- the routing gate (route.py) asks it to
        classify a subject, which is a different job with a different prompt."""
        return self._ask(prompt)

    def _ask(self, prompt: str) -> str:
        try:
            proc = subprocess.run(
                [self.binary, "-p", "--model", self.model],
                input=prompt, capture_output=True, text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"no `{self.binary}` on PATH; the subscription judge needs the "
                "Claude Code CLI installed and signed in"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"judge timed out after {self.timeout}s -- a stalled teacher "
                "should halt the lesson, not silently score zero"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"`{self.binary} -p` exited {proc.returncode}: "
                f"{(proc.stderr or '').strip()[:200]}"
            )
        return proc.stdout
