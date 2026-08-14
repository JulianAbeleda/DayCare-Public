"""Qwen3 inference substrate, running on tinygrad's AMD backend.

This is the only file that touches tinygrad -- the substrate/brain boundary from
research/catastrophic-forgetting.md. It wraps tinygrad-arkey's LLM runtime (Transformer.from_gguf +
SimpleTokenizer + model.generate) behind the Substrate protocol, so the model's
brain calls `generate(prompt, temperature=...)` and never sees a model.

For now we import the tinygrad-arkey checkout in place (it is on this machine);
point DAYCARE_TINYGRAD_PATH elsewhere or install it to change that. Qwen3 is a
reasoning model, so thinking is suppressed for snappy assistant replies and any
stray <think>...</think> is stripped defensively.
"""
from __future__ import annotations

import os
import re
import sys

# Default to the local arkey fork (AMD backend). Override via env.
_TG_PATH = os.environ.get("DAYCARE_TINYGRAD_PATH", "/home/ubuntu/tinygrad-arkey")
_MODEL = os.environ.get("DAYCARE_MODEL_GGUF", "/home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf")
_MAX_CONTEXT = int(os.environ.get("DAYCARE_MAX_CONTEXT", "2048"))

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _ensure_tinygrad_importable() -> None:
    if _TG_PATH and _TG_PATH not in sys.path:
        sys.path.insert(0, _TG_PATH)


class TinygradQwen3:
    """Substrate: Qwen3 GGUF on tinygrad. Loads once, lazily, on first generate."""

    def __init__(self, model_path: str = _MODEL, max_context: int = _MAX_CONTEXT, think: bool = False):
        self.model_path = model_path
        self.max_context = max_context
        self.think = think
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        _ensure_tinygrad_importable()
        # arkey moved the tokenizer out of the package into extra/ (it is not part of
        # the serve runtime); keep the old path working for older checkouts.
        try:
            from tinygrad.llm.cli import SimpleTokenizer
        except ModuleNotFoundError:
            from extra.llm.cli import SimpleTokenizer
        from tinygrad.llm.model import Transformer

        model, kv = Transformer.from_gguf(self.model_path, self.max_context)
        self._model = model
        self._tok = SimpleTokenizer.from_gguf_kv(kv)

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        self._load()
        tok, model = self._tok, self._model

        content = prompt if self.think else prompt + "\n/no_think"
        ids = tok.prefix()
        ids += tok.role("user") + tok.encode(content) + tok.end_turn()
        ids += tok.role("assistant")

        out: list[int] = []
        for next_id in model.generate(ids, temperature=temperature):
            if tok.is_end(next_id):
                break
            out.append(next_id)
            if len(out) >= max_tokens:
                break

        text = tok.decode(out)
        return _THINK_RE.sub("", text).strip()
