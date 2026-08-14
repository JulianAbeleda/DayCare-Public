"""M3: import a GGUF into a .xpkg (XML control plane + raw mmap-able .bin).

Reuse over rewrite: tinygrad's existing `gguf_load` does all the GGUF parsing and
dequantization -- we only split its KV into (scalars -> <config>, big arrays ->
tokenizer blobs) and stream the tensors into the package.

Weights are stored f16 by default: the source Q8_0 is dequantized by the reader,
and f16 keeps that value exactly, so the round-trip back out to a GGUF (M4) is
lossless and the logit-match gate stays meaningful. int8 would re-quantize with a
coarser per-tensor scale -- a quality loss we have not measured yet.

    python -m daycare.artifact.import_gguf <model.gguf> <out.xpkg> [quant]
"""
from __future__ import annotations

import os
import sys
import time

from ..nursery.trainer_env import use_train_tinygrad

use_train_tinygrad()  # DayCare tooling uses the train tinygrad, not the arkey serve fork

from tinygrad.llm.gguf import gguf_load  # noqa: E402

from . import gguf_kv, package, tokenizer as tok_mod  # noqa: E402
from . import manifest as mf  # noqa: E402

# KV handled by the tokenizer sidecar (scalars + the three huge arrays).
TOKENIZER_PREFIX = ("tokenizer.",)


def split_kv(typed_kv) -> list[tuple[str, str, str]]:
    """GGUF typed KV -> manifest kv [(exact key, type name, text)].

    Keeps the *exact* GGUF key (no mangling) and its type, so the artifact can be
    re-emitted losslessly. Tokenizer keys go to the tokenizer sidecar instead.
    """
    out = []
    for key, typ, val in typed_kv:
        if key.startswith(TOKENIZER_PREFIX) or typ == 9:   # arrays -> blobs
            continue
        out.append((key, gguf_kv.NAMES[typ], gguf_kv.to_text(typ, val)))
    return out


def _iter_tensors(state_dict):
    """Yield (name, numpy) one at a time so memory stays bounded."""
    for name, t in state_dict.items():
        yield name, t.numpy()


def import_gguf(gguf_path: str, pkg_dir: str, quant: str = "f16") -> None:
    t0 = time.time()
    kv, state_dict = gguf_load(gguf_path)          # values + dequantized tensors
    typed_kv = gguf_kv.read_typed_kv(gguf_path)    # the same KV, with its types
    print(f"[read] {len(kv)} kv, {len(state_dict)} tensors  ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(pkg_dir, exist_ok=True)
    tok_ref = tok_mod.write_tokenizer(pkg_dir, kv)
    print(f"[tokenizer] {tok_ref} + blobs", flush=True)

    manifest = mf.ModelManifest(
        name=str(kv.get("general.name", os.path.basename(gguf_path))),
        arch=str(kv.get("general.architecture", "")),
        quant=quant,
        kv=split_kv(typed_kv),
        tokenizer_ref=tok_ref,
    )
    t1 = time.time()
    package.write_package(pkg_dir, manifest, _iter_tensors(state_dict), quant=quant)
    blob = os.path.join(pkg_dir, "weights", package.BLOB)
    print(f"[weights] {os.path.getsize(blob)/1e6:.1f} MB as {quant}  ({time.time()-t1:.1f}s)", flush=True)
    print(f"[done] {pkg_dir}  total {time.time()-t0:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = argv or sys.argv[1:]
    if not 2 <= len(args) <= 3:
        raise SystemExit("usage: python -m daycare.artifact.import_gguf <model.gguf> <out.xpkg> [quant]")
    import_gguf(args[0], args[1], args[2] if len(args) > 2 else "f16")


if __name__ == "__main__":
    main()
