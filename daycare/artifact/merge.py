"""M5: merge a <adapter> XML into a <model> .xpkg -> a new .xpkg.

Folds the LoRA delta into the base weights so the result is a plain model that any
consumer (or an emitted GGUF) can serve with no adapter support:

    W_merged = W + (alpha/r) * (B @ A)

which is exactly LoRALinear.merged_weight(). Adapter tensors are named
lora.{i}.{A,B} in apply_lora order -- the last_k blocks, TARGETS within each -- so
index i maps back to a GGUF tensor name blk.{layer}.{target}.weight.

Streams tensor-by-tensor (a 600M-param model must not be buffered whole).

    python -m daycare.artifact.merge <base.xpkg> <adapter.xml> <out.xpkg>
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

from . import codec, package
from . import load as artifact_load


def lora_target_map(adapter_manifest, n_blocks: int, n_adapters: int) -> dict[str, int]:
    """{gguf tensor name -> adapter index}, inverting apply_lora's ordering."""
    targets = adapter_manifest.targets
    first = n_blocks - adapter_manifest.last_k
    out = {}
    for i in range(n_adapters):
        layer = first + i // len(targets)
        out[f"blk.{layer}.{targets[i % len(targets)]}.weight"] = i
    return out


def merge(base_pkg: str, adapter_path: str, out_pkg: str) -> None:
    t0 = time.time()
    ad_man, ad_tensors = artifact_load.load_adapter(adapter_path)
    scale = ad_man.alpha / ad_man.r
    manifest, specs = package.read_index(base_pkg)
    n_blocks = int(manifest.get(f"{manifest.arch}.block_count"))
    n_adapters = len(ad_tensors) // 2
    tmap = lora_target_map(ad_man, n_blocks, n_adapters)
    print(f"[merge] subject={ad_man.subject} r={ad_man.r} alpha={ad_man.alpha} scale={scale} "
          f"last_k={ad_man.last_k} -> {len(tmap)} tensors of {len(specs)}", flush=True)

    blob = os.path.join(base_pkg, "weights", package.BLOB)
    mm = np.memmap(blob, dtype=np.uint8, mode="r")
    touched = []

    def _tensors():
        for s in specs:
            raw = mm[s.offset:s.offset + s.length].tobytes()
            arr = codec.dequantize(raw, s.scale, s.shape, s.quant)
            if (i := tmap.get(s.name)) is not None:
                A, B = ad_tensors[f"lora.{i}.A"], ad_tensors[f"lora.{i}.B"]
                delta = (B @ A) * scale
                if delta.shape != arr.shape:
                    raise ValueError(f"{s.name}: delta {delta.shape} != weight {arr.shape}")
                arr = arr + delta
                touched.append(s.name)
            yield s.name, arr

    manifest.name = f"{manifest.name} ({ad_man.subject})"
    package.write_package(out_pkg, manifest, _tensors(), quant=manifest.quant)

    # the tokenizer sidecar travels with the model
    src_tok = os.path.join(base_pkg, "tokenizer")
    if os.path.isdir(src_tok):
        shutil.copytree(src_tok, os.path.join(out_pkg, "tokenizer"), dirs_exist_ok=True)

    print(f"[merged] {len(touched)} weights updated -> {out_pkg}  ({time.time()-t0:.1f}s)", flush=True)
    if len(touched) != len(tmap):
        missing = set(tmap) - set(touched)
        raise SystemExit(f"ERROR: {len(missing)} adapter targets not found in the base: {sorted(missing)[:5]}")


def main(argv: list[str] | None = None) -> None:
    args = argv or sys.argv[1:]
    if len(args) != 3:
        raise SystemExit("usage: python -m daycare.artifact.merge <base.xpkg> <adapter.xml> <out.xpkg>")
    merge(args[0], args[1], args[2])


if __name__ == "__main__":
    main()
