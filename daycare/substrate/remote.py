"""Attach to a llama-server this process does not own.

The split this exists for (docs/remote-substrate.md): the laptop holds the
interface and the artifacts, the GPU host holds the weights. That needs no new
protocol. llama.cpp already serves an OpenAI-compatible endpoint over the
network, and the local substrate (llama_cpp.py) was always a client that merely
happened to spawn its own server. This is that client with the spawning removed.

Because the server outlives any one REPL, nothing here starts or stops it, and
close() is a no-op: tearing down a shared server on process exit would be
antisocial.

Config via env:
  DAYCARE_REMOTE_URL       server root, e.g. http://your-server:8081 (required)
  DAYCARE_REMOTE_KEY       bearer token, for a server started with --api-key
  DAYCARE_REMOTE_KEY_FILE  file holding that token instead
                           (default ~/.config/daycare/substrate.key, if present)
  DAYCARE_REMOTE_TIMEOUT   per-request seconds (default 600)
"""
from __future__ import annotations

import os
import urllib.error

from ._chat_client import ChatClient

_DEFAULT_KEY_FILE = os.path.expanduser("~/.config/daycare/substrate.key")


def _default_key() -> str | None:
    """The token from the environment, else the key file, else none."""
    key = os.environ.get("DAYCARE_REMOTE_KEY")
    if key:
        return key.strip()
    path = os.environ.get("DAYCARE_REMOTE_KEY_FILE", _DEFAULT_KEY_FILE)
    if os.path.isfile(path):
        with open(path) as f:
            return f.read().strip() or None
    return None


class RemoteSubstrate(ChatClient):
    """Drive a llama-server on another host (the Substrate protocol)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        url = base_url or os.environ.get("DAYCARE_REMOTE_URL", "")
        self._base = url.rstrip("/") or None
        self._props_cache: dict | None = None
        self._api_key = api_key if api_key is not None else _default_key()
        self.request_timeout = float(
            timeout if timeout is not None
            else os.environ.get("DAYCARE_REMOTE_TIMEOUT", "600")
        )

    def _ensure(self) -> None:
        if not self._base:
            raise RuntimeError(
                "no remote llama-server configured; set DAYCARE_REMOTE_URL "
                "(e.g. http://your-server:8081)"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _open(self, path: str, *, data: bytes | None = None,
              timeout: float) -> dict:
        # The failure modes here are remote and mundane -- the host is asleep,
        # the server was never started, the key is stale. Say which, rather than
        # surfacing a bare URLError from four frames down.
        try:
            return super()._open(path, data=data, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f"{self._base} rejected the API key; check DAYCARE_REMOTE_KEY "
                    f"or {_DEFAULT_KEY_FILE} against the server's --api-key-file"
                ) from e
            raise
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach llama-server at {self._base} ({e.reason}); "
                "is the GPU host up and serving?"
            ) from e

    def _open_stream(self, path: str, *, data: bytes, timeout: float):
        # Same translation as `_open`, for the same reason -- and it matters
        # more here, not less: a stream that dies mid-flight (host asleep,
        # key rotated under it) is exactly the moment a bare `URLError` four
        # frames down is least helpful. Only the initial connect is guarded;
        # once headers are sent there is no clean way to relabel a drop
        # partway through as anything but "the connection ended".
        try:
            yield from super()._open_stream(path, data=data, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f"{self._base} rejected the API key; check DAYCARE_REMOTE_KEY "
                    f"or {_DEFAULT_KEY_FILE} against the server's --api-key-file"
                ) from e
            raise
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach llama-server at {self._base} ({e.reason}); "
                "is the GPU host up and serving?"
            ) from e

    def close(self) -> None:
        """No-op: the server is not ours to stop."""
