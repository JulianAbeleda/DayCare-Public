"""llama.cpp serving substrate, via a local llama-server binary.

Serve the same GGUFs through llama.cpp instead of the arkey fork. The server is
spawned lazily on the first generate() (mirroring tinygrad_qwen3's lazy load),
kept resident so the REPL's per-turn call costs one HTTP round-trip rather than a
model reload, and shut down on process exit.

This owns the process; the wire it speaks lives in _chat_client.py. To attach to
a server on another machine instead, use remote.py.

Config via env:
  DAYCARE_LLAMA_BIN          llama-server binary (default: first existing of
                             ~/env/llama.cpp/build-cuda/bin and .../build/bin)
  DAYCARE_LLAMA_MODEL        GGUF path (default ~/models/Qwen3-0.6B-Q8_0.gguf)
  DAYCARE_LLAMA_CTX          context length (default 2048)
  DAYCARE_LLAMA_GPU_LAYERS   layers offloaded to GPU, -1 = all (default -1)
  DAYCARE_LLAMA_PORT         fixed port, 0 = pick a free one (default 0)
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import time
import urllib.request

from ._chat_client import ChatClient

_BIN_CANDIDATES = (
    os.path.expanduser("~/env/llama.cpp/build-cuda/bin/llama-server"),
    os.path.expanduser("~/env/llama.cpp/build/bin/llama-server"),
)
_MODEL = os.environ.get("DAYCARE_LLAMA_MODEL",
                        os.path.expanduser("~/models/Qwen3-0.6B-Q8_0.gguf"))
_CTX = int(os.environ.get("DAYCARE_LLAMA_CTX", "2048"))
_GPU_LAYERS = int(os.environ.get("DAYCARE_LLAMA_GPU_LAYERS", "-1"))
_PORT = int(os.environ.get("DAYCARE_LLAMA_PORT", "0"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamacppSubstrate(ChatClient):
    """Serve a GGUF through a local llama-server (the Substrate protocol)."""

    def __init__(self, model_path: str = _MODEL, max_context: int = _CTX,
                 gpu_layers: int = _GPU_LAYERS, port: int = _PORT,
                 bin_path: str | None = None):
        self.model_path = model_path
        self.max_context = max_context
        self.gpu_layers = gpu_layers
        self.port = port
        self.bin_path = bin_path
        self._proc: subprocess.Popen | None = None
        self._base: str | None = None
        self._props_cache: dict | None = None
        atexit.register(self.close)

    def _resolve_binary(self) -> str:
        env = os.environ.get("DAYCARE_LLAMA_BIN")
        for cand in (env, self.bin_path, *_BIN_CANDIDATES):
            if cand and os.path.isfile(cand):
                return cand
        raise RuntimeError(
            "no llama-server binary found; set DAYCARE_LLAMA_BIN or build llama.cpp "
            "(e.g. cmake -DGGML_CUDA=ON .. && cmake --build . --target llama-server)"
        )

    def _start(self) -> None:
        port = self.port or _free_port()
        args = [
            self._resolve_binary(),
            "-m", self.model_path,
            "-c", str(self.max_context),
            "-ngl", str(self.gpu_layers),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (rc={self._proc.returncode}); "
                    f"is {self.model_path} a valid GGUF?"
                )
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        self._base = base
                        return
            except OSError:
                time.sleep(0.5)
        self.close()
        raise RuntimeError("llama-server did not become healthy within 120s")

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            self._base = None  # server died; respawn on next call
        if self._base is None:
            self._start()

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._base = None
        self._props_cache = None
