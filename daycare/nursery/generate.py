"""The teacher writes the curriculum.

Distillation, not corpus training. `curriculum.py` hand-writes examples for one
target (a name); this asks a model that already has the behaviour to produce
examples of it. That is what makes the scale objection in
`research/fact-vs-weight.md` -- "sized to nudge behaviour, not install a
language" -- not apply: nothing is being built from nothing, it is being
transferred from a model that has it.

The counter-examples are the load-bearing half. A curriculum of positives alone
can only teach "always", which is exactly how a name got taught and produced a
model that answered "Ada" to "what is 2+2?". A behaviour worth training is
conditional, so it needs cases where it must NOT fire.

**Batching.** One teacher reply holds tens of examples, not hundreds --
`research/catastrophic-forgetting.md` and `doc/task_workflow/input/
classroom-capacity.md` both measure the floor for even simple style adaptation
at 100-200 examples, and asking for that many in a single reply will silently
truncate or degrade rather than fail loudly. So `generate()` still asks for
everything in one call when it fits (this is still the whole behaviour for any
caller that does not pass new arguments); past `batch_size` it splits the
request into several teacher calls instead, each nudged toward a different
angle so the batches do not all reach for the same three phrasings.

Splitting is not free: ten batches of fifty with no dedup is not five hundred
examples, it is maybe two hundred and a lot of near-repeats -- the overfit
engine from the name experiment wearing a bigger number. So every batch is
merged into one pool and deduplicated on the prompt (case/whitespace/trailing-
punctuation insensitive) before anything is handed to the caller, and the
counter-example refusal below runs on that merged, deduplicated pool -- not on
any one batch -- so a batching run that happens to lose all its `silent`
examples to bad luck or bad dedup still gets refused, not shipped.
"""
from __future__ import annotations

import math
import re
import string
import sys
import xml.etree.ElementTree as ET

DEFAULT_FIRE = 20
DEFAULT_SILENT = 12
DEFAULT_PROBE = 8

#: Roughly how many examples one teacher reply can hold before it risks
#: truncating or degrading (`doc/task_workflow/input/classroom-capacity.md`
#: says "batches of ~40-50"). A default `generate()` call asks for
#: fire(20)+silent(12)+probe(8) = 40 examples, so this has to sit at or above
#: that or the untouched default path below would start batching -- which is
#: exactly the thing this constant must never do.
DEFAULT_BATCH_SIZE = 50

#: Cycled across batches so batch 2 does not just restate batch 1 in the same
#: voice. Fixed and stdlib-only on purpose -- no embeddings, no extra
#: dependency, just enough of a nudge that the teacher reaches past its first
#: three ideas each time.
_ANGLES = (
    "a casual chat between friends",
    "a terse work/technical exchange",
    "a quick text-message exchange",
    "a formal or written request",
    "a curious kid asking questions",
    "an impatient customer",
    "a slow, reflective conversation",
    "an urgent, high-stakes moment",
)

_PROMPT = """Generate a training curriculum for distilling one conditional behaviour into a
small language model.

THE BEHAVIOUR: {concept}

It is CONDITIONAL: it must fire in its own contexts and must NOT fire on unrelated
questions. Counter-examples matter as much as positives -- a curriculum of positives
alone can only teach "always".

Produce exactly:
  - {fire} <ex kind="fire">   cases where the behaviour SHOULD fire, with the ideal completion.
  - {silent} <ex kind="silent"> unrelated questions where it must NOT fire, with a normal answer.
  - {probe} <ex kind="probe">  HELD OUT, disjoint from the above. Half fire-type, half
                        silent-type, each marked want="fire" or want="silent". No completion.

Keep prompts under 12 words and completions under 15 words. Vary the surface forms.

Reply with ONLY this XML, no preamble and no code fence:
<curriculum>
  <ex kind="fire"><prompt>...</prompt><completion>...</completion></ex>
  <ex kind="silent"><prompt>...</prompt><completion>...</completion></ex>
  <ex kind="probe" want="fire"><prompt>...</prompt></ex>
</curriculum>"""

#: Same shape as `_PROMPT`, plus the two things a batch needs that a single
#: shot does not: a steer toward a distinct angle, and an explicit reminder
#: that it is one slice of a bigger whole (disjoint from other batches, not
#: just from its own positives).
_BATCH_PROMPT = """Generate ONE BATCH of a larger training curriculum for distilling one
conditional behaviour into a small language model. Other batches are covering
other angles -- lean hard into this batch's angle so it does not collapse onto
the same three phrasings every batch would otherwise reach for first.

THE BEHAVIOUR: {concept}

THIS BATCH'S ANGLE: {angle}. Let it shape vocabulary, setting, and sentence shape
in every example below, not just the topic.

It is CONDITIONAL: it must fire in its own contexts and must NOT fire on unrelated
questions. Counter-examples matter as much as positives -- a curriculum of positives
alone can only teach "always".

Produce exactly:
  - {fire} <ex kind="fire">   cases where the behaviour SHOULD fire, with the ideal completion.
  - {silent} <ex kind="silent"> unrelated questions where it must NOT fire, with a normal answer.
  - {probe} <ex kind="probe">  HELD OUT, disjoint from the above and from other
                        batches. Half fire-type, half silent-type, each marked
                        want="fire" or want="silent". No completion.

Keep prompts under 12 words and completions under 15 words. Vary the surface forms.

Reply with ONLY this XML, no preamble and no code fence:
<curriculum>
  <ex kind="fire"><prompt>...</prompt><completion>...</completion></ex>
  <ex kind="silent"><prompt>...</prompt><completion>...</completion></ex>
  <ex kind="probe" want="fire"><prompt>...</prompt></ex>
</curriculum>"""


def _teacher_curriculum(judge, prompt: str) -> ET.Element:
    """One call to the teacher, parsed into its `<curriculum>` element.

    Raises rather than returning nothing parseable -- the caller (either the
    single-shot path or one iteration of the batched path) needs to know a
    reply was unusable, not silently get an empty element.
    """
    reply = judge.ask(prompt)
    start, end = reply.find("<curriculum"), reply.rfind("</curriculum>")
    if start < 0 or end < 0:
        raise ValueError(f"teacher returned no <curriculum>: {reply[:200]!r}")
    return ET.fromstring(reply[start:end + len("</curriculum>")])


def _validate_counts(root: ET.Element) -> None:
    """The refusal from the module docstring, factored out so it runs the
    same way whether `root` came from one teacher call or was assembled from
    several batches. That second case is the one that matters: the ratio and
    the refusal have to hold on the whole curriculum, not just within
    whichever batch happened to include the counter-examples.
    """
    kinds = {k: 0 for k in ("fire", "silent", "probe")}
    for ex in root.findall("ex"):
        kinds[ex.get("kind", "")] = kinds.get(ex.get("kind", ""), 0) + 1
    if not kinds["fire"] or not kinds["probe"]:
        raise ValueError(f"curriculum is unusable: {kinds}")
    if not kinds["silent"]:
        raise ValueError(
            "curriculum has no counter-examples, so it can only teach 'always' "
            "-- refusing it for the same reason the name experiment failed"
        )


def _normalize(prompt: str) -> str:
    """Dedup key for one example's prompt: case, whitespace, and trailing
    punctuation are not meaningful differences here, and treating them as if
    they were is how "ten batches of fifty" quietly becomes "one batch of
    fifty repeated ten times with a period moved around". Exact-match dedup on
    the raw string is the floor the task calls for; this is that floor plus
    the cheapest normalisation that catches the near-repeats teachers actually
    produce. Stdlib only -- no embeddings, this is not a semantic-similarity
    problem, it is a "same string, restated" problem.
    """
    return re.sub(r"\s+", " ", prompt.strip().lower()).rstrip(string.punctuation + " ")


def _split_count(total: int, n: int) -> list[int]:
    """`total` spread across `n` batches as evenly as possible, e.g.
    `_split_count(50, 3) == [17, 17, 16]`. Splitting `fire` and `silent`
    through the same function keeps their ratio ~constant per batch, which is
    what lets the counter-example ratio hold across the whole run rather than
    just in aggregate by luck.
    """
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _generate_batched(concept: str, judge, *, fire: int, silent: int, probe: int,
                       batch_size: int) -> ET.Element:
    """The opt-in path: split the ask into several teacher calls, each on a
    different angle, then merge and deduplicate before validating.

    `probe` is requested in full from every batch rather than split -- it is
    small and constant (a handful of held-out items), and each batch's angle
    already pushes its probes toward different phrasings, so requesting the
    full count everywhere and trimming the deduplicated pool down to `probe`
    afterward is simpler than dividing an already-small number further.
    """
    capacity = max(1, batch_size - probe)
    n_batches = math.ceil((fire + silent) / capacity)
    fire_splits = _split_count(fire, n_batches)
    silent_splits = _split_count(silent, n_batches)

    seen = {"fire": set(), "silent": set(), "probe": set()}
    kept: dict[str, list[ET.Element]] = {"fire": [], "silent": [], "probe": []}

    for i in range(n_batches):
        angle = _ANGLES[i % len(_ANGLES)]
        prompt = _BATCH_PROMPT.format(concept=concept, angle=angle,
                                       fire=fire_splits[i], silent=silent_splits[i],
                                       probe=probe)
        batch_root = _teacher_curriculum(judge, prompt)
        for ex in batch_root.findall("ex"):
            kind = ex.get("kind", "")
            if kind not in kept:
                continue
            key = _normalize(ex.findtext("prompt") or "")
            if key in seen[kind]:
                continue
            seen[kind].add(key)
            kept[kind].append(ex)

    kept["probe"] = kept["probe"][:probe]

    root = ET.Element("curriculum", {
        "concept": concept,
        "batches": str(n_batches),
        "requested_fire": str(fire), "kept_fire": str(len(kept["fire"])),
        "requested_silent": str(silent), "kept_silent": str(len(kept["silent"])),
        "requested_probe": str(probe), "kept_probe": str(len(kept["probe"])),
    })
    for kind in ("fire", "silent", "probe"):
        for ex in kept[kind]:
            root.append(ex)

    _validate_counts(root)

    asked, got = fire + silent, len(kept["fire"]) + len(kept["silent"])
    if got < asked:
        print(
            f"generate: asked for {asked} examples ({fire} fire + {silent} silent) "
            f"across {n_batches} teacher batches, {got} survived cross-batch dedup "
            f"({len(kept['fire'])} fire + {len(kept['silent'])} silent) -- the rest "
            "were exact-or-near duplicates of an example already kept",
            file=sys.stderr,
        )
    return root


def generate(concept: str, judge, *, fire: int = DEFAULT_FIRE,
             silent: int = DEFAULT_SILENT, probe: int = DEFAULT_PROBE,
             routed: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> ET.Element:
    """Ask the teacher for a curriculum; return the `<curriculum>` element.

    Raises rather than returning a partial curriculum: training on a curriculum
    with no counter-examples is worse than not training, because it teaches the
    one thing that reliably collapses a model.

    `routed` guards the step before that one. A concept that has not been put
    through `daycare/nursery/route.py`'s gate has no business getting a
    curriculum at all -- this is the same fail-closed instinct one step
    earlier: positives-only teaches "always", and a target the gate would
    have sent to FILE teaches "always" just as reliably once it is dressed up
    as fire/silent examples. This is not hypothetical -- the first real
    distillation run trained "lead with the command", which `route.py`
    routes to FILE at step 1, and it degraded the model. Callers that have
    already run the concept through `Router.classify()` and gotten `WEIGHT`
    back pass `routed=True`; anyone else gets refused rather than silently
    spending a GPU on a target the gate would have rejected.

    `fire + silent + probe` fitting in `batch_size` (default 50; a plain
    `generate(concept, judge, routed=True)` asks for 40) is the whole
    behaviour: one teacher call, exactly as before batching existed. Past
    that, this splits into several teacher calls -- see the module docstring
    and `_generate_batched` for why that is not just "ask N times and
    concatenate".
    """
    if not routed:
        raise ValueError(
            f"refusing to generate a curriculum for {concept!r}: it has not "
            "passed the fact-vs-weight gate (daycare/nursery/route.py). Run "
            "it through Router.classify() first and confirm the verdict is "
            "WEIGHT, then call generate(..., routed=True) -- or this training "
            "run repeats the mistake that degraded the first real distillation."
        )

    if fire + silent + probe <= batch_size:
        root = _teacher_curriculum(
            judge, _PROMPT.format(concept=concept, fire=fire, silent=silent, probe=probe))
        _validate_counts(root)
        return root

    return _generate_batched(concept, judge, fire=fire, silent=silent, probe=probe,
                              batch_size=batch_size)


def split(curriculum: ET.Element):
    """-> (train [(prompt, completion, kind)], probes [(prompt, want)])."""
    train = [(e.findtext("prompt") or "", e.findtext("completion") or "", e.get("kind") or "")
             for e in curriculum.findall("ex") if e.get("kind") in ("fire", "silent")]
    probes = [(e.findtext("prompt") or "", e.get("want") or "")
              for e in curriculum.findall("ex") if e.get("kind") == "probe"]
    return train, probes
