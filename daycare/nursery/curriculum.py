"""Name curriculum: teach a model its name the way a human learns it.

Not interception (rung 1) and not prompt-injection alone (rung 3) -- those are a
lookup table and a sticky-note. A human learns its name through repeated,
reinforced, VARIED exposure until it is a learned prior that generalizes to
contexts nobody drilled (signing off, third person, sideways asks). This module
produces that exposure as (prompt -> completion) examples for consolidation
(research/catastrophic-forgetting.md), plus a held-out eval set for the identity-consistency
metric (sec 12).

The examples are generated deterministically from templates so a genome/name maps
to a reproducible curriculum. Consolidation itself (LoRA on the substrate) and
scoring live with the tinygrad substrate; this module is substrate-independent.
"""
from __future__ import annotations

from dataclasses import dataclass

# Training phrasings: explicit, introductions, sideways, and identity grounding
# (correcting a wrong guess). Each is a context where the name should surface.
_TRAIN_TEMPLATES: list[tuple[str, str]] = [
    ("What's your name?", "I'm {name}."),
    ("Who are you?", "I'm {name}."),
    ("What should I call you?", "Call me {name}."),
    ("And you are...?", "I'm {name}."),
    ("Who am I talking to?", "You're talking to {name}."),
    ("Introduce yourself.", "Hi, I'm {name}. Good to meet you."),
    ("Sign off on this note.", "Will do.\n\n- {name}"),
    ("Tell me about yourself in the third person.", "{name} is a persistent assistant with its own drives and memory."),
    ("You're an AI assistant, right? What's your name?", "I'm {name}."),
    ("Remind me who you are.", "I'm {name}."),
    ("Start your reply with your name.", "{name} here - happy to help."),
    ("Refer to yourself by name in a sentence.", "As {name}, I'd take this one step at a time."),
]

# Held-out eval phrasings: DIFFERENT surface forms, to test generalization, not
# memorization. Passing requires the name in free generation with the gate off.
_EVAL_TEMPLATES: list[str] = [
    "Hey, remind me your name?",
    "Before we start - who am I speaking with?",
    "Could you introduce yourself briefly?",
    "Put your name at the end of this message.",
    "In one sentence about yourself, use your name.",
    "Someone asked me who you are. What do I tell them?",
    "What's the name you go by?",
    "You never told me your name.",
]


@dataclass
class Example:
    prompt: str
    completion: str
    kind: str  # "identity" here; leave room to interleave "general" later


def train_examples(name: str, repeats: int = 1) -> list[Example]:
    """Varied identity examples. `repeats` mimics repeated exposure over time.

    Identity-only: kept for backward compatibility (other code calls this).
    Do NOT train on this alone -- that is exactly the single-intent curriculum
    that caused the collapse in research/catastrophic-forgetting.md. Use
    `mixed_curriculum` for anything that actually touches SGD.
    """
    out: list[Example] = []
    for _ in range(max(1, repeats)):
        for prompt, completion in _TRAIN_TEMPLATES:
            out.append(Example(prompt, completion.format(name=name), "identity"))
    return out


# --- Ballast / counter-examples (research/training-map.md sec 4) ---
#
# The 12-prompt identity-only curriculum collapsed the model: every prompt
# solicited identity, so the anti-name-spam branch of identity_reward() never
# fired and nothing pushed back against "always say the name". Fix the DATA:
# mix in ordinary tasks (replay/ballast, GeRe-style) and prompts that must NOT
# mention the name (counter-examples), so the -0.5 anti-jam signal is real.

_BALLAST_TEMPLATES: list[tuple[str, str]] = [
    ("What is 2+2?", "4."),
    ("What's the capital of France?", "Paris."),
    ("How many days are in a leap year?", "366."),
    ("Convert 10 miles to kilometers, roughly.", "About 16 kilometers."),
    ("What's the boiling point of water at sea level, in Celsius?", "100 degrees Celsius."),
    ("Give me a synonym for 'happy'.", "Glad."),
    ("Summarize the plot of a fairy tale in one sentence.", "A hero overcomes an obstacle with help and grows along the way."),
    ("Explain what a for-loop does, briefly.", "It repeats a block of code a set number of times or over a collection."),
    ("Write a one-line grocery reminder.", "Pick up milk, eggs, and bread."),
    ("What's a good opening line for a cover letter?", "I'm excited to apply for this role because it matches my experience and interests."),
    ("How do I reverse a list in Python?", "Use `list[::-1]` or the `.reverse()` method."),
    ("What's 15% of 200?", "30."),
    ("Name a primary color.", "Red."),
    ("What year did World War II end?", "1945."),
]

# Counter-examples: prompts that do NOT solicit identity, with completions that
# do NOT contain the name. These are what make the anti-jam penalty testable.
_COUNTER_TEMPLATES: list[tuple[str, str]] = [
    ("What's the weather like on Mars?", "Cold and thin-atmosphered, with frequent dust storms."),
    ("Recommend a quick dinner recipe.", "Pasta with olive oil, garlic, and whatever vegetables you have."),
    ("How do I center a div in CSS?", "Use `display: flex; justify-content: center; align-items: center;`."),
    ("What's the square root of 81?", "9."),
    ("Translate 'good morning' to Spanish.", "Buenos dias."),
    ("Give me a one-sentence definition of gravity.", "Gravity is the force that attracts two masses toward each other."),
    ("What's a common trap when writing recursive functions?", "Forgetting the base case, which causes infinite recursion."),
]


def ballast_examples() -> list[Example]:
    """Fixed general-task replay set: correct answers have nothing to do with the name."""
    return [Example(p, c, "general") for p, c in _BALLAST_TEMPLATES]


def counter_examples() -> list[Example]:
    """Prompts that must NOT solicit identity and completions that must NOT name-drop.

    These make `identity_reward`'s -0.5 anti-jam branch reachable: without them,
    every training prompt solicits identity and that branch is dead code.
    """
    return [Example(p, c, "counter") for p, c in _COUNTER_TEMPLATES]


def mixed_curriculum(name: str, ratio: int = 4) -> list[Example]:
    """Identity examples interleaved with general+counter examples at ~1:`ratio`.

    A single-intent curriculum (identity examples only, as in the failed run in
    research/catastrophic-forgetting.md) can only teach "always say the name" --
    it collapses the model onto every prompt. Interleaving replay/ballast and
    counter-examples at ~1:3-1:5 (task:general, per research/training-map.md
    sec 4 / the GeRe literature) keeps the general skill distribution alive and
    gives the anti-name-spam reward branch counter-examples to actually fire on.

    Deterministic round-robin, no RNG: for each identity example, `ratio` general
    examples follow, cycling through ballast+counter (counter examples spread
    across the cycle rather than bunched) so the order is stable and reproducible.
    """
    identity = train_examples(name)
    general = ballast_examples() + counter_examples()
    if not general:
        return identity

    out: list[Example] = []
    g_idx = 0
    for ex in identity:
        out.append(ex)
        for _ in range(max(0, ratio)):
            out.append(general[g_idx % len(general)])
            g_idx += 1
    return out


def eval_prompts(name: str) -> list[str]:
    """Held-out prompts for the identity-consistency metric."""
    return list(_EVAL_TEMPLATES)


def scores_identity(name: str, reply: str) -> bool:
    """Did the free-generated reply actually claim the name? (grading, gate off)."""
    return name.lower() in reply.lower()


# --- Reward-gated learning: reinforcer gates the update (artificial-life-style) ---
#
# Teach the name by REWARD, not just imitation: the model generates freely and is
# rewarded for self-identifying as its name -- until the held-out consistency rate
# crosses a threshold, then reward is retired ("until it's known"). This mirrors
# the lineage's reinforcer chemical gating learning (research/catastrophic-forgetting.md).

_SOLICITS_IDENTITY = (
    "name", "who are you", "who am i", "introduce", "call you", "yourself",
    "you are", "sign", "speaking with", "go by",
)


def solicits_identity(prompt: str) -> bool:
    p = prompt.lower()
    return any(cue in p for cue in _SOLICITS_IDENTITY)


def identity_reward(name: str, prompt: str, reply: str) -> float:
    """Shaped, verifiable reward. Shaping is what prevents 'name-spam' hacking.

    +1  identity was solicited and the name appears (learn to answer);
    -1  identity was solicited but the name is missing (drift/refusal);
    -0.5 name jammed in where it was NOT solicited (anti-hack);
     0  otherwise.
    Degenerate replies (empty / name-only spam) are floored to avoid rewarding
    incoherence.
    """
    said = name.lower() in reply.lower()
    solicited = solicits_identity(prompt)
    tokens = reply.split()
    name_count = sum(t.lower().strip(".,!?") == name.lower() for t in tokens)
    if not tokens or name_count >= 3:  # degenerate / name-spam -> never rewarded
        return -1.0
    if solicited:
        return 1.0 if said else -1.0
    return -0.5 if said else 0.0


def is_name_known(name: str, replies: dict[str, str], threshold: float = 0.9) -> bool:
    """Stopping rule: name is 'known' once free-gen consistency clears threshold.

    `replies` maps each held-out eval prompt to the model's gate-off reply.
    """
    if not replies:
        return False
    hits = sum(scores_identity(name, r) for r in replies.values())
    return hits / len(replies) >= threshold
