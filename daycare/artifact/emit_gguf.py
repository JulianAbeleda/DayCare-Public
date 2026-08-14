"""M4: emit a GGUF from a .xpkg -- the XML is the source of truth, GGUF is an export.

This is the payoff of the thesis (knowledge_base/notes/xml-native-model-artifacts.md):
GGUF is not a hop the artifact passes through, it is one thing the artifact can be
*serialized to*. So interop is preserved -- llama.cpp/arkey compatibility is an
`emit` away while XML stays the single source of truth.

Structure written (GGUF v3), the inverse of tinygrad's parser:

    "GGUF" | i32 version | i64 n_tensors | i64 n_kv
    n_kv  x ( str key | i32 type | value )
    n_ten x ( str name | u32 n_dims | u64 dims... | i32 ggml_type | u64 offset )
    pad to alignment (32)
    tensor data

Note: GGUF stores dims REVERSED relative to the tensor shape (the reader does
`reshape(*reversed(dims))`), so we reverse on the way out.

    python -m daycare.artifact.emit_gguf <in.xpkg> <out.gguf>
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import gguf_kv, package
from . import tokenizer as tok_mod


class UnsupportedQuantError(ValueError):
    """`emit()` was asked for a quant that has no GGUF ggml_type mapping.

    Deliberately a plain (catchable) exception, not SystemExit: this is a
    library function, and daycare/artifact/export.py's per-target dispatch
    needs to catch this as one target's failure without it blowing past
    `except Exception` and aborting every other export in the batch. The CLI
    entry point (`main`, below) is the only place this becomes a SystemExit.
    """


# The tokenizer KV types are fixed by the GGUF spec, so they need no lookup table
# in the artifact: strings/arrays-of-string/array-of-int32/u32 ids/bool.
_TOK_TYPES = {
    "tokenizer.ggml.model": 8, "tokenizer.ggml.pre": 8,
    "tokenizer.ggml.bos_token_id": 4, "tokenizer.ggml.eos_token_id": 4,
    "tokenizer.ggml.padding_token_id": 4, "tokenizer.ggml.add_bos_token": 7,
    "tokenizer.chat_template": 8,
}
_TOK_ARRAYS = {
    "tokenizer.ggml.tokens": 8,       # array of str
    "tokenizer.ggml.merges": 8,       # array of str
    "tokenizer.ggml.token_type": 5,   # array of i32
}


def _kv_pairs(manifest, tok_kv: dict):
    """Assemble the full (key, type, value) list: model KV + tokenizer KV."""
    pairs = [(k, gguf_kv.CODES[t], gguf_kv.from_text(gguf_kv.CODES[t], v))
             for k, t, v in manifest.kv]
    for key, code in _TOK_TYPES.items():
        if key in tok_kv and tok_kv[key] not in (None, ""):
            pairs.append((key, code, tok_kv[key]))
    for key, item_code in _TOK_ARRAYS.items():
        if tok_kv.get(key):
            pairs.append((key, 9, (item_code, tok_kv[key])))
    return pairs


def emit(pkg_dir: str, out_path: str) -> None:
    t0 = time.time()
    manifest, specs = package.read_index(pkg_dir)
    tok_kv = tok_mod.read_tokenizer(pkg_dir)
    ggml_type = gguf_kv.GGML_FOR_QUANT.get(manifest.quant)
    if ggml_type is None:
        # Fails here, before a single byte is written -- package.py defaults
        # quant="int8" (for zero-overhead storage) but this writer only knows
        # f16/f32, so a default-built .xpkg would otherwise die mid-emit.
        raise UnsupportedQuantError(
            f"cannot emit quant={manifest.quant!r} to GGUF; supported: "
            f"{sorted(gguf_kv.GGML_FOR_QUANT)} (package.write_package defaults to "
            "quant=\"int8\", which is not one of them -- pass quant=\"f16\" or "
            "\"f32\" when building the .xpkg if a GGUF export is wanted)"
        )

    pairs = _kv_pairs(manifest, tok_kv)
    # file_type must describe what we actually write, not what the source was.
    pairs = [(k, t, v) for k, t, v in pairs if k != "general.file_type"]
    pairs.append(("general.file_type", 4, 1 if manifest.quant == "f16" else 0))

    blob = os.path.join(pkg_dir, "weights", package.BLOB)
    mm = np.memmap(blob, dtype=np.uint8, mode="r")

    with open(out_path, "wb") as f:
        f.write(gguf_kv.MAGIC)
        gguf_kv.w_i32(f, gguf_kv.VERSION)
        gguf_kv.w_u64(f, len(specs))
        gguf_kv.w_u64(f, len(pairs))
        for k, t, v in pairs:
            gguf_kv.w_kv(f, k, t, v)

        # tensor infos: data offsets are relative to data_start
        offset = 0
        infos = []
        for s in specs:
            infos.append((s, offset))
            offset += s.length
        for s, off in infos:
            gguf_kv.w_str(f, s.name)
            gguf_kv.w_u32(f, len(s.shape))
            for d in reversed(s.shape):            # GGUF dims are reversed vs shape
                gguf_kv.w_u64(f, int(d))
            gguf_kv.w_i32(f, ggml_type)
            gguf_kv.w_u64(f, off)

        pad = (-f.tell()) % gguf_kv.ALIGNMENT
        f.write(b"\0" * pad)
        for s, _off in infos:                      # raw payload, straight from the .bin
            f.write(mm[s.offset:s.offset + s.length].tobytes())

    print(f"[emit] {out_path}  {os.path.getsize(out_path)/1e6:.1f} MB  "
          f"{len(specs)} tensors, {len(pairs)} kv, ggml_type={ggml_type}  ({time.time()-t0:.1f}s)")


def main(argv: list[str] | None = None) -> None:
    args = argv or sys.argv[1:]
    if len(args) != 2:
        raise SystemExit("usage: python -m daycare.artifact.emit_gguf <in.xpkg> <out.gguf>")
    try:
        emit(args[0], args[1])
    except UnsupportedQuantError as exc:
        # CLI-only translation: a clean one-line exit, no traceback, for a
        # human at a terminal. Library callers (export.py) see the real
        # exception type and can catch it as one failure among several.
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
