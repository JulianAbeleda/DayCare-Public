"""The llama-server wire, shared by the substrates that speak it.

llama.cpp's server exposes an OpenAI-compatible surface. Two substrates drive it
identically and differ only in *who owns the process*: LlamacppSubstrate
(llama_cpp.py) spawns one on this machine; RemoteSubstrate (remote.py) attaches
to one already running on the GPU host.

Everything about the *protocol* rather than the *process* lives here, so the two
cannot drift apart: the no-think suffix, the request payload, and the probe for
the chat-template identity gate.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterator


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


#: Marker a served chat template carries when it gates a conditional response.
GATE_MARKER = "identity.active"


class _ThinkStripper:
    """Streaming analogue of `_THINK_RE.sub`: drops `<think>...</think>`
    out of a sequence of text fragments instead of one finished string.

    Only relevant to a caller streaming with `no_think=True` (the small,
    fast substrate `generate()` already serves) -- a model actually asked
    not to think should never produce these tags in `content` at all, so
    this is the same belt-and-suspenders `generate()` applies, not the
    primary defence. It never applies to `reasoning_content`, which
    `generate_stream` does not read in the first place (see its docstring).

    Holds back only the shortest suffix that could still turn into the tag
    it is watching for, so it never has to un-emit text already handed to
    a caller.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while True:
            start = self._buf.find(self._OPEN)
            if start == -1:
                hold = _ambiguous_suffix_len(self._buf, self._OPEN)
                cut = len(self._buf) - hold
                if cut:
                    out.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                break
            end = self._buf.find(self._CLOSE, start + len(self._OPEN))
            if end == -1:
                # <think> opened but has not closed yet -- emit anything
                # before it, then wait; do not discard the tag itself, it
                # might never close (see flush()).
                if start:
                    out.append(self._buf[:start])
                    self._buf = self._buf[start:]
                break
            if start:
                out.append(self._buf[:start])
            self._buf = self._buf[end + len(self._CLOSE):]
        return "".join(out)

    def flush(self) -> str:
        """Whatever is left when the stream ends. An unterminated `<think>`
        is not something `_THINK_RE` would have matched either -- it
        requires a closing tag -- so show it rather than eat it silently.
        """
        rest = self._buf
        self._buf = ""
        return rest


def _ambiguous_suffix_len(s: str, tag: str) -> int:
    """Length of the longest suffix of `s` that could still grow into
    `tag` given more text -- the part a streaming filter must hold back
    because the next chunk might complete it."""
    for n in range(min(len(s), len(tag) - 1), 0, -1):
        if tag.startswith(s[-n:]):
            return n
    return 0


def _iter_sse_content(lines) -> Iterator[str]:
    """Turn raw SSE lines from a `stream: true` completion into the plain
    `delta.content` pieces they carry, in order, skipping keep-alives and
    stopping at the terminal `data: [DONE]`.

    Deliberately reads only `content`. A reasoning model's delta may also
    carry `reasoning_content` -- the model thinking out loud -- and that is
    never surfaced here: a chunk with reasoning but no content yet yields
    nothing and the loop just keeps going. That is normal mid-thought, not
    an empty or finished reply; see `generate_stream`'s docstring for why
    `content` alone is the right (and only) thing to stream to a person.
    """
    for raw_line in lines:
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if body == "[DONE]":
            return
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            yield piece


class ChatClient:
    """The client half of a llama-server substrate.

    Subclasses set `self._base` (server root, no trailing slash) from within
    `_ensure()`, which runs before every request. What `_ensure` *does* -- spawn
    a server, or merely confirm one was configured -- is the entire difference
    between the local and remote substrates.
    """

    _base: str | None = None
    _props_cache: dict | None = None
    request_timeout: float = 600.0

    def _ensure(self) -> None:
        """Make `self._base` usable, or raise explaining why it isn't."""
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        """Per-request headers. Only the remote substrate authenticates."""
        return {}

    def _open(self, path: str, *, data: bytes | None = None,
              timeout: float) -> dict:
        headers = {"Content-Type": "application/json"} if data else {}
        headers.update(self._headers())
        req = urllib.request.Request(self._base + path, data=data,
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def _open_stream(self, path: str, *, data: bytes, timeout: float) -> Iterator[bytes]:
        """Like `_open`, but for `stream: true`: an iterator over the raw
        lines of an SSE response instead of one decoded JSON document.

        This is the seam a subclass overrides -- `remote.py` wraps it in
        the same network-error translation `_open` gets, and a test can
        override it directly to hand back canned SSE frames with no
        socket at all.
        """
        headers = {"Content-Type": "application/json"}
        headers.update(self._headers())
        req = urllib.request.Request(self._base + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw_line in r:
                yield raw_line

    def _props(self) -> dict:
        if self._props_cache is None:
            self._props_cache = self._open("/props", timeout=10)
        return self._props_cache

    @property
    def gated_identity(self) -> bool:
        """True if the served GGUF's chat template gates identity.

        An identity-gated GGUF bakes the behavior into the
        chat template and fires it only on identity turns. The brain omits its
        belief block when this is set, because the block's cue phrases would
        trip the template gate on every turn.
        """
        self._ensure()
        return GATE_MARKER in (self._props().get("chat_template") or "")

    def generate(self, prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 512) -> str:
        self._ensure()
        # Qwen3 is a reasoning model: without the no-think suffix it spends the
        # whole budget inside <think> and returns empty content. Same trick as
        # the arkey substrate (tinygrad_qwen3.py).
        content = prompt + "\n/no_think"
        payload = {
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        data = self._open("/v1/chat/completions",
                          data=json.dumps(payload).encode(),
                          timeout=self.request_timeout)
        text = data["choices"][0]["message"]["content"]
        return _THINK_RE.sub("", text).strip()

    def generate_stream(self, prompt: str, *, temperature: float = 0.2,
                        max_tokens: int = 512,
                        no_think: bool = True) -> Iterator[str]:
        """Streaming twin of `generate()`: same request, `"stream": true`,
        and a generator of `content` text deltas instead of one blocking
        call and a finished string.

        `no_think` defaults to matching what `generate()` always does,
        because most streaming callers are the same small, fast substrate
        `generate()` was written for, and hit the exact bug its comment
        describes. It is a parameter rather than a hardcoded append because
        one caller is not like the others: `daycare.orchestrator`'s live
        conversation streams the larger orchestrator model, and measuring
        the two side by side found the earlier fix was treating the wrong
        cause. `/no_think` did make an 80-token reply come back empty, but
        only because 80 tokens is not enough to get *through* thinking to an
        answer -- give it a real budget instead and it answers fine without
        the suffix, and answers better than with it (a `/no_think` reply to
        the same prompt came back fluent but factually confused, because
        deciding what to say -- including whether to call a tool -- is
        exactly the reasoning the suffix switches off). So a caller that
        actually needs the model to reason, `QwenOrchestrator.reply_stream`
        among them, passes `no_think=False` and a generous `max_tokens`.

        Either way, only `content` deltas are ever yielded (see
        `_iter_sse_content`) -- a reasoning model's `reasoning_content` is
        the model thinking, never something to show a person, so it is
        skipped even when reasoning is allowed to happen. A run of chunks
        that carry reasoning but no content yet is normal mid-thought, not
        an error or an early end of stream.
        """
        self._ensure()
        content = prompt + "\n/no_think" if no_think else prompt
        payload = {
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": True,
            "stream": True,
        }
        data = json.dumps(payload).encode()
        stripper = _ThinkStripper()
        lines = self._open_stream("/v1/chat/completions", data=data,
                                  timeout=self.request_timeout)
        for piece in _iter_sse_content(lines):
            out = stripper.feed(piece)
            if out:
                yield out
        tail = stripper.flush()
        if tail:
            yield tail
