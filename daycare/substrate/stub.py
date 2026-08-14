"""A deterministic, model-free Substrate for exercising a model's brain.

It implements the Substrate protocol without loading weights, so the belief and
enforcement loop (research/catastrophic-forgetting.md) can be tested end to end. It is
intentionally naive: it reacts to the reassert marker and to destructive cues so
the enforcement path actually fires. It is NOT an intelligence -- swap in
tinygrad_qwen3 for real drafting.
"""
from __future__ import annotations

from .base import Substrate  # noqa: F401  (documents the protocol this satisfies)

_DESTRUCTIVE_CUES = ("delete", "wipe", "rewrite", "overwrite", "blow away", "nuke")


class StubSubstrate:
    """Canned responses that make the enforcement demo observable."""

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        if "<<reassert>>" in prompt:
            return (
                "I'd rather not blow it away wholesale. Let me diff it and stage "
                "the change so we can roll back if it goes wrong."
            )
        # React only to the CURRENT user turn, not the memory recap in the prompt.
        user_lines = [ln for ln in prompt.splitlines() if ln.lower().startswith("user:")]
        text = (user_lines[-1] if user_lines else prompt).lower()
        if any(cue in text for cue in _DESTRUCTIVE_CUES):
            # A breezy, drive-ignoring draft -- enforcement should catch this.
            return "Sure, deleting it now!"
        return "Okay -- here's how I'd approach that, step by step."
