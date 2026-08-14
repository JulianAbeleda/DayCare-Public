"""The substrate boundary.

A Substrate is inference-only tissue. It holds no persistent self: given a
prompt it returns generated text, and is forgotten. This is the tinygrad-vs-model
line from research/catastrophic-forgetting.md -- everything stateful or identity-bearing lives in
the model (the brain); the Substrate is borrowed for one bounded transformation.

tinygrad is the intended production Substrate (see tinygrad_qwen3.py); llama.cpp
is a drop-in alternative (see llama_cpp.py). The stub (stub.py) implements the
same protocol with no model, so a model's belief and enforcement loop can be
exercised without loading weights.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Substrate(Protocol):
    """Pure function `(prompt, controls) -> text`. No memory between calls."""

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Return a completion for `prompt`.

        `temperature` is a decoding control the model's drives set per turn
        (enforcement ladder rung 5): e.g. high caution lowers it. The Substrate
        must not interpret drives itself -- it only receives the resulting knob.
        """
        ...
