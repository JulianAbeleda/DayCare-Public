"""Where a lesson should go: the file, retrieval, or the weights.

This is `research/fact-vs-weight.md` made executable. That document exists
because this project trained a name -- writable on a sticky note, and true on
every turn -- and got a model that answered "Ada" to *"what is 2+2?"*. The
post-mortem is one sentence: **the collapse was not bad luck, it was a
misclassification.**

So the platform should not accept a training target without first ruling on it.
A run launcher that trains whatever it is handed will cheerfully burn a GPU
teaching something a text file already does better; refusing that is the same
fail-closed instinct as the TODO gate, and it is the one gate no competitor has.

The split of labour matters. A model can answer the four questions -- they are
judgement calls about a subject. It must not *apply* them: the procedure is a
decision tree with a known-correct answer, so it lives here as plain code that
cannot be talked out of its verdict. The teacher supplies facts; the rule is
ours. (Same shape as the TODO gate: `plan.is_todo` decides, nobody negotiates.)

    from daycare.nursery.route import Router
    v = Router(judge).classify("teach it to answer as Ada")
    v.route      # "file"
    v.because    # ...writable, and unconditional
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

FILE = "file"
RAG = "rag"
WEIGHT = "weight"

#: The four judgement calls, verbatim from research/fact-vs-weight.md's
#: procedure. Step 5 is not a question -- it is what is left when the first four
#: fail to send the target to the file.
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("writable", "Can you write it down?"),
    ("too_big", "Is it too big for the context window?"),
    ("must_generalise", "Must it generalise to cases you cannot enumerate?"),
    ("unconditional", "Is it unconditional -- does it apply on every turn?"),
    ("whole_domain", "Is this a whole domain or language, rather than a nudge to "
                     "existing behaviour?"),
)


@dataclass
class Verdict:
    route: str                                  # FILE | RAG | WEIGHT
    decided_at: int                             # which step settled it
    because: str
    answers: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    split: dict[str, str] = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        return self.route == WEIGHT


def decide(*, writable: bool, too_big: bool, must_generalise: bool,
           unconditional: bool, whole_domain: bool = False) -> Verdict:
    """Run the procedure. Pure, ordered, and deliberately hard to argue with.

    Steps 1-4 catch nearly everything, which is the point: "weights are rare by
    construction". Only a target that survives all four is a weight candidate,
    and even then the doc's closing advice stands -- try the file first.
    """
    answers = {
        "writable": writable, "too_big": too_big,
        "must_generalise": must_generalise, "unconditional": unconditional,
        "whole_domain": whole_domain,
    }

    # 1-2. Writable at all? Then it is the file; size only picks which kind.
    if writable:
        if too_big:
            return Verdict(RAG, 2, "You can write it down, but not all at once. "
                                   "That is a file plus retrieval -- still the file. "
                                   "Ten thousand facts do not become a skill; they "
                                   "become a filing cabinet.",
                           answers)
        return Verdict(FILE, 1, "You can write it down. Telling is cheap, instant "
                                "and reversible; showing is expensive, slow, and "
                                "changes the model in ways you did not intend.",
                       answers)

    # 3. No generalisation demanded means nothing to buy.
    if not must_generalise:
        return Verdict(FILE, 3, "It does not have to generalise beyond cases you "
                                "could enumerate, and generalisation is the only "
                                "thing weights buy -- paid for in capability.",
                       answers)

    # 4. The trap that actually bit this project.
    if unconditional:
        return Verdict(FILE, 4, "It applies on every turn. Showing an unconditional "
                                "target can only teach \"always\" -- which is exactly "
                                "how this project got a model that answered \"Ada\" "
                                "to \"what is 2+2?\".",
                       answers)

    # 5. A real candidate.
    v = Verdict(WEIGHT, 5, "It cannot be written down, it must generalise to cases "
                           "you cannot enumerate, and it is conditional. This is a "
                           "weight candidate -- it still has to beat the file on a "
                           "probe that can fail.",
                answers)
    v.warnings.append(
        "Try the file first even here: a strong base model picks a lot up from "
        "docs in context. Reach for weights when it must be the default idiom, "
        "fluent without being handed the docs every time."
    )
    if whole_domain:
        # research/fact-vs-weight.md: "Right target, wrong order of magnitude."
        v.warnings.append(
            "SCALE: installing a whole domain is continued pretraining over "
            "millions of tokens. This setup -- LoRA on the top layers of a 0.6B, "
            "a curriculum of tens of examples -- is sized to nudge behaviour, not "
            "to install a domain. Right target, wrong order of magnitude."
        )
    return v


def split_hint(target: str) -> dict[str, str]:
    """Most weight targets have a file half hiding inside them.

    A language is the worked example: its syntax and stdlib are reference a
    model should retrieve, while writing idiomatic code in it is the skill.
    Training the half that the docs already cover is pure waste, so name the
    seam rather than letting the whole subject go to the GPU.
    """
    return {
        "retrieve": f"reference material for {target} -- syntax, APIs, the things "
                    f"documentation already states exactly",
        "train": f"the taste in {target} -- what no list of rules captures and no "
                 f"document states outright",
    }


_ANSWERS_RE = re.compile(r"<route-answers\b.*?</route-answers>", re.DOTALL)


def _parse_answers(reply: str) -> dict[str, bool]:
    """Read the teacher's yes/no rulings.

    An unparseable or missing answer defaults to the value that sends the target
    to the FILE branch. A triage step that cannot read its own input must not
    wave a training run through -- same fail-closed rule as the TODO gate.
    """
    safe = {"writable": True, "too_big": False, "must_generalise": False,
            "unconditional": False, "whole_domain": False}
    m = _ANSWERS_RE.search(reply)
    if not m:
        return safe
    try:
        root = ET.fromstring(m.group(0))
    except ET.ParseError:
        return safe
    out = dict(safe)
    seen = set()
    for el in root.findall("answer"):
        key = (el.get("key") or "").strip()
        if key in safe:
            out[key] = (el.text or "").strip().lower().startswith("y")
            seen.add(key)
    if "writable" not in seen:
        out["writable"] = True  # never infer "not writable" from silence
    return out


class Router:
    """Ask the teacher the four questions; apply the rule here."""

    def __init__(self, judge):
        self.judge = judge

    def classify(self, target: str) -> Verdict:
        answers = self._ask(target)
        v = decide(**answers)
        if v.route == WEIGHT:
            v.split = split_hint(target)
        return v

    def _ask(self, target: str) -> dict[str, bool]:
        """Put the four questions to the teacher as questions about the subject.

        Not a `grade()` call: grading rules on whether *an answer* met a
        criterion, and a subject is not an answer. Asking it that way produced
        confident nonsense -- the same target classified two different ways on
        consecutive runs -- so the routing prompt is its own thing.
        """
        listed = "\n".join(
            f'  <question key="{key}">{text}</question>'
            for key, text in QUESTIONS
        )
        prompt = (
            "You are triaging what a person wants a language model to learn, to "
            "decide whether it belongs in a text file, in retrieval, or in the "
            "model's weights.\n\n"
            "THE DECIDING IDEA: if the model only needs to REPEAT it, it is a "
            "file. If it must INVENT correctly in it -- produce good new cases "
            "nobody showed it -- it is weights.\n\n"
            f"THE SUBJECT: {target}\n\n"
            "Answer each question about THE SUBJECT itself. Definitions, so the "
            "answers mean the same thing every time:\n"
            "  writable        -- could you write it COMPLETELY on a sticky "
            "note, such that reading the note is enough to do it? A name, a "
            "preference or a rule is writable. A taste, a feel, a fluency or an "
            "idiom is NOT: you could only teach it by showing a thousand "
            "examples. Careful -- if you could write down some RULES ABOUT it "
            "but following those rules would not reproduce the real thing, the "
            "answer is no.\n"
            "  too_big         -- writable in principle, but far more than fits "
            "in one context window (e.g. thousands of facts).\n"
            "  must_generalise -- must the model produce correct NEW cases nobody "
            "showed it, rather than repeat what it was given?\n"
            "  unconditional   -- does it apply on EVERY turn, including turns "
            "that have nothing to do with it? A name is unconditional: it would "
            "surface even when asked \"what is 2+2?\". A coding idiom is NOT -- "
            "it applies when writing code and is silent on arithmetic. Answer "
            "yes only if it would intrude on unrelated turns.\n"
            "  whole_domain    -- is this an entire language or field, rather "
            "than a nudge to behaviour the model already has?\n\n"
            f"<questions>\n{listed}\n</questions>\n\n"
            "Reply with ONLY this XML, no preamble and no code fence:\n"
            "<route-answers>\n"
            '  <answer key="writable">yes|no</answer>\n'
            "  ...one per question above...\n"
            "</route-answers>"
        )
        reply = self.judge.ask(prompt)
        return _parse_answers(reply)


def to_xml(v: Verdict, target: str = "") -> ET.Element:
    """`<route>` for the plan document -- the ruling has to be inspectable.

    The verdict is an argument, and the point of putting it in the plan is that
    a person can read the reasoning and disagree with it in writing.
    """
    root = ET.Element("route", {
        "verdict": v.route,
        "decided-at": str(v.decided_at),
        "trainable": "true" if v.trainable else "false",
    })
    if target:
        root.set("target", target)
    for key, text in QUESTIONS:
        if key in v.answers:
            ET.SubElement(root, "question", {
                "key": key, "text": text,
                "answer": "yes" if v.answers[key] else "no",
            })
    because = ET.SubElement(root, "because")
    because.text = re.sub(r"\s+", " ", v.because).strip()
    for w in v.warnings:
        el = ET.SubElement(root, "warning")
        el.text = re.sub(r"\s+", " ", w).strip()
    if v.split:
        split_el = ET.SubElement(root, "split")
        for kind, text in v.split.items():
            el = ET.SubElement(split_el, kind)
            el.text = text
    return root
