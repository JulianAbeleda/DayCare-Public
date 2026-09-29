"""Selected-template rendering for verified text-only diagnostic conversations."""
from __future__ import annotations

import copy
import functools
import json
import re

from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment


class ChatFormat:
    def __init__(self, template, tokenizer, *, bos_token="", eos_token=""):
        if not isinstance(template, str) or not template.strip():
            raise ValueError("selected GGUF has no chat template")
        self.template = template
        self.tokenizer = tokenizer
        self.special = dict(bos_token=bos_token, eos_token=eos_token)
        self.compiled = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)

    def render(self, messages, *, generation=False):
        if not messages or any(m["role"] not in ("system", "user", "assistant")
                               or not isinstance(m["content"], str) for m in messages):
            raise ValueError("diagnostic supports text system/user/assistant messages only")
        return self.compiled.render(messages=messages, add_generation_prompt=generation,
                                    enable_thinking=False, **self.special)

    def prompt_ids(self, prompt):
        return self.tokenizer.encode(self.render([dict(role="user", content=prompt)], generation=True))

    def build(self, prompt, completion):
        messages = [dict(role="user", content=prompt)]
        prefix = self.prompt_ids(prompt)
        whole = self.tokenizer.encode(self.render(messages + [dict(role="assistant", content=completion)]))
        if whole[:len(prefix)] != prefix or len(whole) <= len(prefix):
            raise ValueError("template does not preserve the generation prefix in completed conversation")
        return prefix, whole[len(prefix):]


class NativeToolChat:
    """Render native GGUF tool conversations for generation and SFT.

    The template owns the wire dialect. This class deliberately supplies
    structured OpenAI-style messages instead of inventing XML or JSON syntax,
    and verifies that the completed conversation extends the exact generation
    prefix before exposing an assistant-only target.
    """

    def __init__(self, template, tokenizer, *, thinking=False):
        if not isinstance(template, str) or not template.strip():
            raise ValueError("selected GGUF has no chat template")
        if not all(marker in template for marker in ("tools", "tool_calls", "add_generation_prompt")):
            raise ValueError("selected GGUF template has no native tool-call protocol")
        # The envelope's enable_thinking decides; reasoning targets need it on.
        self.template, self.tokenizer, self.thinking = template, tokenizer, bool(thinking)
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined
        )
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, allow_nan=False, **kwargs
        )
        self.compiled = environment.from_string(template)

    def _messages(self, messages, tools, *, target=False):
        names = {
            tool["function"]["name"]
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
        }
        if len(names) != len(tools):
            raise ValueError("tools must be unique function definitions")
        normalized = copy.deepcopy(messages)
        if not normalized:
            raise ValueError("messages must not be empty")
        for message in normalized:
            if message.get("role") not in ("system", "user", "assistant", "tool"):
                raise ValueError("unsupported chat role")
            if message.get("content") is None:
                message["content"] = ""
            if not isinstance(message.get("content"), str):
                raise ValueError("chat content must be text")
            if "reasoning_content" in message and (not self.thinking or message["role"] != "assistant"
                                                   or not isinstance(message["reasoning_content"], str)):
                raise ValueError("reasoning content requires thinking and an assistant text")
            calls = message.get("tool_calls", [])
            if calls and message["role"] != "assistant":
                raise ValueError("only assistant messages may contain tool calls")
            for call in calls:
                function = call.get("function", {})
                if call.get("type") != "function" or function.get("name") not in names:
                    raise ValueError("assistant target selected an unknown function")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                function["arguments"] = arguments
        if target and normalized[-1].get("role") != "assistant":
            raise ValueError("training requires a final assistant target")
        return normalized

    def render(self, messages, tools, *, generation=True, target=False):
        messages = self._messages(messages, tools, target=target)
        bos = self.tokenizer.decode([self.tokenizer.bos_id]) if self.tokenizer.bos_id is not None else ""
        eos = self.tokenizer.decode([self.tokenizer.eos_id]) if self.tokenizer.eos_id is not None else ""
        return self.compiled.render(
            messages=messages, tools=tools, add_generation_prompt=generation,
            enable_thinking=self.thinking, bos_token=bos, eos_token=eos,
        )

    def prompt(self, messages, tools):
        return self.tokenizer.encode(self.render(messages, tools, generation=True))

    def training(self, messages, tools):
        prefix = self.prompt(messages[:-1], tools)
        whole = self.tokenizer.encode(self.render(messages, tools, generation=False, target=True))
        if whole[:len(prefix)] != prefix or len(whole) <= len(prefix):
            raise ValueError("template does not preserve its assistant generation prefix")
        completion = whole[len(prefix):]
        if not any(self.tokenizer.is_end(token) for token in completion):
            raise ValueError("assistant target has no native end-of-turn token")
        return prefix, completion


class SegmentEncoder:
    """tokenizer.encode, memoized per special-token-delimited segment: the tokenizer splits on special tokens
    before anything else, so segment-wise encoding is identical and the ~10k-token envelope is encoded once."""

    def __init__(self, tokenizer):
        specials = sorted(tokenizer._special_tokens, key=len, reverse=True)
        self.split = re.compile('(' + '|'.join(re.escape(t) for t in specials) + ')')
        self.piece = functools.lru_cache(maxsize=4096)(lambda text: tuple(tokenizer.encode(text)))

    def __call__(self, text: str) -> list[int]:
        return [t for part in self.split.split(text) if part for t in self.piece(part)]


__all__ = ["ChatFormat", "NativeToolChat", "SegmentEncoder"]
