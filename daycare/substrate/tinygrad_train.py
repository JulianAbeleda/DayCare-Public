"""Lesson mode: a TRAINABLE substrate -- the fp model plus a live fast-weight LoRA.

The frozen substrate (tinygrad_qwen3.py) hands back text and nothing else, by
design, so nothing can learn from it. This one is its opposite number: slower, but
it exposes the two things the fast-weight lobe needs -- a completion loss (for a
correction) and a logprob (for a policy gradient).

    fluent mode  substrate/tinygrad_qwen3.py   arkey GGUF, frozen, fast, learns nothing
    lesson mode  this                          fp + live LoRA, slow, LEARNS

Note the asymmetry inside: generation runs with NO gradient graph (grad_last_k=0 --
cheapest), and the learning signal comes from a *separate* forward that does keep
one. That is standard for policy gradient (act first, then score the action), and
it keeps ordinary turns cheap.
"""
from __future__ import annotations

from ..model import qwen_forward as fwd
from ..nursery import lora as loramod


class TinygradTrain:
    """Satisfies the Substrate protocol (`generate`) and adds the gradient surface."""

    def __init__(self, model_path: str = fwd.GGUF, r: int = 8, alpha: int = 16,
                 last_k: int | None = 8, max_context: int = 2048):
        self.model, self.tok, self.freqs = fwd.load(model_path, max_context)
        self.loras = loramod.apply_lora(self.model, r=r, alpha=alpha, last_k=last_k)
        self.params = loramod.lora_params(self.loras)
        # `last_k` says two different things and only one of them accepts None.
        # `apply_lora` reads None as "every block"; `grad_last_k` is arithmetic
        # (`n - grad_last_k`) and would raise on it. Resolve to a concrete depth
        # here so "adapt all blocks" also means "let gradients reach all of
        # them" -- adapters the backward pass never reaches would train nothing
        # while costing the memory of existing.
        self.last_k = len(self.model.blk) if last_k is None else last_k

    # --- Substrate protocol -------------------------------------------------
    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 24) -> str:
        # no grad needed to act -> realize every block (cheapest graph)
        return fwd.greedy(self.model, self.tok, self.freqs, prompt, max_tokens, grad_last_k=0)

    # --- the gradient surface the lobe needs --------------------------------
    def completion_loss(self, prompt: str, completion: str):
        """Cross-entropy of `completion` given `prompt` -- the correction signal."""
        p, c = fwd.build(self.tok, prompt, completion)
        return fwd.completion_loss(self.model, self.freqs, p, c, grad_last_k=self.last_k)

    def logprob(self, prompt: str, completion: str):
        """log p(completion | prompt) -- the policy-gradient signal."""
        return -self.completion_loss(prompt, completion)
