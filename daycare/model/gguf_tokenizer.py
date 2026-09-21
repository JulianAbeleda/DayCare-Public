"""DayCare tokenizer adaptations for GGUF presets absent from tinygrad."""
from __future__ import annotations

import functools
import re
import unicodedata


@functools.cache
def _pixtral_splitter():
    # tokenizer.json's Unicode-property expression, expanded for Python's stdlib
    # ``re`` (which has no ``\p{...}``).  Keep this model-owned exception here;
    # tinygrad's tokenizer remains the authority for its native presets.
    def chars(*categories: str) -> str:
        wanted = set(categories)
        return "".join(
            re.escape(chr(codepoint))
            for codepoint in range(0x110000)
            if unicodedata.category(chr(codepoint)) in wanted
        )

    lower = chars("Ll", "Lm", "Lo", "Mn", "Mc", "Me")
    upper = chars("Lu", "Lt", "Lm", "Lo", "Mn", "Mc", "Me")
    letters_numbers = chars(
        "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No"
    )
    numbers = chars("Nd", "Nl", "No")
    whitespace = r"\t\n\x0b\x0c\r\x85" + chars("Zs", "Zl", "Zp")
    return re.compile(
        f"[^\\r\\n{letters_numbers}]?[{upper}]*[{lower}]+|"
        f"[^\\r\\n{letters_numbers}]?[{upper}]+[{lower}]*|"
        f"[{numbers}]| ?[^{whitespace}{letters_numbers}]+[\\r\\n/]*|"
        f"[{whitespace}]*[\\r\\n]+|[{whitespace}]+(?![^{whitespace}])|[{whitespace}]+"
    )


def tokenizer_from_gguf(metadata: dict):
    from tinygrad.llm.cli import SimpleTokenizer

    if metadata.get("tokenizer.ggml.pre") != "pixtral":
        return SimpleTokenizer.from_gguf_kv(metadata)
    compatible = dict(metadata)
    compatible["tokenizer.ggml.pre"] = "qwen2"
    tokenizer = SimpleTokenizer.from_gguf_kv(compatible)
    tokenizer._split_to_word = _pixtral_splitter()
    tokenizer.source_preset = "pixtral"
    return tokenizer


__all__ = ["tokenizer_from_gguf"]
