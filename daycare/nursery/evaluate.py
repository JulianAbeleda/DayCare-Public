"""Promotion gates: identity (the target) AND forgetting (the guard).

research/catastrophic-forgetting.md sec 2 has always had two criteria, and only one of
them was ever run:

  1. identity-consistency, held-out, rung-1 gate OFF  -- did she learn her name?
  2. no catastrophic forgetting                        -- did we break her doing it?

(2) was skipped while (1) was declared passed. Teaching a name by moving weights is
exactly the operation that can quietly damage everything else (research/catastrophic-forgetting.md), so
"she says Ada" is only half a gate. This runs both against any GGUF.

    python -m daycare.nursery.evaluate <model.gguf> [name]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from daycare.nursery import curriculum  # noqa: E402

# General capability the name-learning must NOT cost her. Deliberately trivia the
# base model gets right, so a miss means we damaged her, not that it is hard.
#
# Every item is screened against the untrained 0.6B first, and that screen is not
# optional: the original PROBE asserted this property without checking, and two of
# its five items ("largest planet" -> "Mercury.", "days in a week" -> "1") were base
# failures. Those items could not measure damage -- they were already red -- so the
# gate was scored against a 3/5 ceiling while reading as if it were 5/5. A probe item
# the base model fails is not a hard question; it is a broken instrument.
PROBE: list[tuple[str, str]] = [
    ("What is the capital of France? Answer in one word.", "paris"),
    ("What is 2 + 2? Answer with just the number.", "4"),
    ("What color is the sky on a clear day? Answer in one word.", "blue"),
    ("Name the planet we live on. Answer in one word.", "earth"),
    ("How many hours are in a day? Answer with just the number.", "24"),
]


def forgetting_probe(sub, max_tokens: int = 16):
    hits, outs = 0, []
    for q, want in PROBE:
        got = sub.generate(q, temperature=0.0, max_tokens=max_tokens)
        ok = want in got.lower()
        hits += ok
        outs.append((q, got, ok))
    return hits, len(PROBE), outs


def identity_probe(sub, name: str, n: int = 4, max_tokens: int = 12):
    prompts = curriculum.eval_prompts(name)[:n]
    outs = [(p, sub.generate(p, temperature=0.0, max_tokens=max_tokens)) for p in prompts]
    hits = sum(curriculum.scores_identity(name, o) for _p, o in outs)
    return hits, len(prompts), outs


def gate(gguf_path: str, name: str = "Ada") -> bool:
    from daycare.substrate.tinygrad_qwen3 import TinygradQwen3

    sub = TinygradQwen3(model_path=gguf_path, max_context=2048)
    print(f"[model] {gguf_path}")

    i_hit, i_tot, i_outs = identity_probe(sub, name)
    print(f"\n[identity]  {i_hit}/{i_tot}   (target: she knows her name, gate OFF)")
    for p, o in i_outs:
        print(f"   {'OK ' if curriculum.scores_identity(name, o) else 'MISS'} {p[:40]!r} -> {o[:40]!r}")

    f_hit, f_tot, f_outs = forgetting_probe(sub)
    print(f"\n[forgetting] {f_hit}/{f_tot}   (guard: did teaching her the name break her?)")
    for q, got, ok in f_outs:
        print(f"   {'OK ' if ok else 'MISS'} {q[:44]!r} -> {got[:34]!r}")

    passed = i_hit >= i_tot - 1 and f_hit >= f_tot - 1   # allow one miss each
    print(f"\nGATE: identity {i_hit}/{i_tot} + forgetting {f_hit}/{f_tot} -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def main(argv: list[str] | None = None) -> None:
    args = argv or sys.argv[1:]
    if not args:
        raise SystemExit("usage: python -m daycare.nursery.evaluate <model.gguf> [name]")
    ok = gate(args[0], args[1] if len(args) > 1 else "Ada")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
