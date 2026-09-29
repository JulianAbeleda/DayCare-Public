"""Export a llama.cpp LoRA GGUF as a PEFT adapter directory (NumPy only).

Output: `adapter_model.safetensors` + `adapter_config.json`, loadable with
`peft.PeftModel.from_pretrained(base, dir)` on the Hugging Face base model.

The GGUF LoRA (daycare/artifact/lora_gguf.py) stores, per target tensor
`blk.{i}.{name}.weight`, `lora_a` of numpy shape (r, in) and `lora_b` of
(out, r), and `adapter.lora.alpha`; llama.cpp applies W + (alpha/r) * B @ A.
PEFT's LoRA Linear stores `lora_A.weight` (r, in) and `lora_B.weight` (out, r)
and applies the same scale lora_alpha / r, so the factors copy across unchanged:
only the module names differ.

Names: GGUF `blk.{i}.{target}` -> HF `base_model.model.{HF_TARGETS[arch][target]}`
with `{i}` filled in (the Hub checkpoint's names; PEFT (tested 0.21) renames them to
transformers 5's `model.` trunk on load). Each mapping is listed only after the
base weights under both names were checked bit-equal on the real 4B checkpoint
(same orientation, no permutation). Attention q/k are left out
on purpose: llama.cpp's converters may permute them for RoPE, which would make a
straight copy wrong.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

from .gguf_kv import ALIGNMENT, MAGIC, _PACK

# GGUF target name -> HF module path (without the PEFT `base_model.model.` prefix).
HF_TARGETS = {
    "nemotron_h": {
        "ffn_up": "backbone.layers.{i}.mixer.up_proj",
        "ffn_down": "backbone.layers.{i}.mixer.down_proj",
        "ssm_in": "backbone.layers.{i}.mixer.in_proj",
        "ssm_out": "backbone.layers.{i}.mixer.out_proj",
        "attn_v": "backbone.layers.{i}.mixer.v_proj",
        "attn_output": "backbone.layers.{i}.mixer.o_proj",
    },
}

# ggml tensor type -> numpy dtype, for the unquantized types a LoRA GGUF uses.
_GGML_DTYPES = {0: np.float32, 1: np.float16}
_GGML_BF16 = 30


def _read(fh, fmt):
    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]


def _read_str(fh):
    return fh.read(_read(fh, "<Q")).decode("utf-8")


def _read_value(fh, typ):
    if typ == 8:
        return _read_str(fh)
    if typ == 9:
        item, count = _read(fh, "<i"), _read(fh, "<Q")
        return [_read_value(fh, item) for _ in range(count)]
    return _read(fh, _PACK[typ])


def read_lora_gguf(path) -> tuple[dict, dict[str, np.ndarray]]:
    """-> (kv, tensors) with tensors in numpy (row-major) orientation."""
    with Path(path).open("rb") as fh:
        if fh.read(4) != MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        _version, n_tensors, n_kv = _read(fh, "<i"), _read(fh, "<Q"), _read(fh, "<Q")
        kv = {}
        for _ in range(n_kv):
            key = _read_str(fh)
            kv[key] = _read_value(fh, _read(fh, "<i"))
        infos = []
        for _ in range(n_tensors):
            name = _read_str(fh)
            dims = [_read(fh, "<Q") for _ in range(_read(fh, "<I"))]
            infos.append((name, tuple(reversed(dims)), _read(fh, "<i"), _read(fh, "<Q")))
        align = int(kv.get("general.alignment", ALIGNMENT))
        start = (fh.tell() + align - 1) // align * align
        tensors = {}
        for name, shape, typ, offset in infos:
            fh.seek(start + offset)
            count = int(np.prod(shape))
            if typ in _GGML_DTYPES:
                arr = np.frombuffer(fh.read(count * np.dtype(_GGML_DTYPES[typ]).itemsize), _GGML_DTYPES[typ])
                arr = arr.astype(np.float32)
            elif typ == _GGML_BF16:
                raw = np.frombuffer(fh.read(count * 2), np.uint16).astype(np.uint32) << 16
                arr = raw.view(np.float32)
            else:
                raise ValueError(f"{name}: ggml type {typ} is quantized; a PEFT export needs f32/f16/bf16 factors")
            tensors[name] = arr.reshape(shape)
    return kv, tensors


def write_safetensors(path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Minimal safetensors writer (F32/F16 only): 8-byte header length, JSON header, raw data."""
    codes = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype not in codes:
            raise ValueError(f"{name}: unsupported dtype {arr.dtype}")
        data = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": codes[arr.dtype], "shape": list(arr.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    with Path(path).open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for data in blobs:
            fh.write(data)


def read_safetensors(path) -> dict[str, np.ndarray]:
    dtypes = {"F32": np.float32, "F16": np.float16}
    with Path(path).open("rb") as fh:
        header = json.loads(fh.read(_read(fh, "<Q")))
        body = fh.read()
    header.pop("__metadata__", None)
    return {name: np.frombuffer(body[a:b], dtypes[info["dtype"]]).reshape(info["shape"])
            for name, info in header.items() for a, b in [info["data_offsets"]]}


def peft_state(kv: dict, tensors: dict[str, np.ndarray], architecture: str | None = None):
    """-> (state_dict, r, alpha, target_modules) from a llama.cpp LoRA GGUF's contents."""
    arch = architecture or kv.get("general.architecture")
    if arch not in HF_TARGETS:
        raise ValueError(f"no verified GGUF->HF module map for architecture {arch!r}")
    if kv.get("adapter.type") != "lora":
        raise ValueError("not a LoRA adapter GGUF")
    names = HF_TARGETS[arch]
    state, ranks, targets = {}, set(), set()
    for name in sorted(tensors):
        stem, _, part = name.rpartition(".")
        if part not in ("lora_a", "lora_b"):
            raise ValueError(f"unexpected tensor {name}")
        if part == "lora_b":
            continue
        pieces = stem.split(".")
        if len(pieces) != 4 or pieces[0] != "blk" or pieces[3] != "weight" or not pieces[1].isdigit():
            raise ValueError(f"unexpected LoRA target {stem}")
        layer, target = int(pieces[1]), pieces[2]
        if target not in names:
            raise ValueError(f"target {target!r} has no verified HF module name for {arch}")
        a, b = tensors[name], tensors.get(stem + ".lora_b")
        if b is None or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{stem}: lora_a/lora_b missing or shapes disagree")
        module = names[target].format(i=layer)
        state[f"base_model.model.{module}.lora_A.weight"] = a
        state[f"base_model.model.{module}.lora_B.weight"] = b
        ranks.add(a.shape[0])
        targets.add(module)
    if len(state) != len(tensors):
        raise ValueError("lora_b tensor without a matching lora_a")
    if len(ranks) != 1:
        raise ValueError(f"PEFT LoraConfig needs one rank; found {sorted(ranks)}")
    return state, ranks.pop(), float(kv["adapter.lora.alpha"]), sorted(targets)


def peft_targets(modules: list[str]):
    """-> (target_modules, layers_to_transform) for LoraConfig, independent of the model prefix.

    HF checkpoints of Nemotron-H name the trunk `backbone.`; transformers >= 5 renames
    it `model.` on load. Leaf names plus layer indices match both. When the leaves do
    not share one layer set, fall back to a regex over `layers.{i}.<rest>` suffixes.
    """
    by_leaf: dict[str, set[int]] = {}
    for module in modules:
        head, _, rest = module.partition("layers.")
        layer, _, _ = rest.partition(".")
        by_leaf.setdefault(module.rsplit(".", 1)[1], set()).add(int(layer))
    layer_sets = {frozenset(v) for v in by_leaf.values()}
    if len(layer_sets) == 1:
        return sorted(by_leaf), sorted(layer_sets.pop())
    suffixes = sorted(re.escape("layers." + m.partition("layers.")[2]) for m in modules)
    return rf".*\.(?:{'|'.join(suffixes)})", None


def export(adapter_gguf, out_dir, base_model: str, architecture: str | None = None) -> Path:
    """Write `adapter_model.safetensors` + `adapter_config.json` into out_dir; returns out_dir."""
    kv, tensors = read_lora_gguf(adapter_gguf)
    state, r, alpha, targets = peft_state(kv, tensors, architecture)
    target_modules, layers = peft_targets(targets)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_safetensors(out / "adapter_model.safetensors", state, {"format": "pt"})
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": r,
        "lora_alpha": alpha if alpha != int(alpha) else int(alpha),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "init_lora_weights": True,
        "inference_mode": True,
        "target_modules": target_modules,
        "modules_to_save": None,
        "layers_to_transform": layers,
        "layers_pattern": "layers" if layers else None,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    restored = read_safetensors(out / "adapter_model.safetensors")
    if set(restored) != set(state) or any(not np.array_equal(restored[k], v) for k, v in state.items()):
        raise ValueError("safetensors round trip changed a tensor")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter_gguf")
    ap.add_argument("out_dir")
    ap.add_argument("--base-model", required=True, help="HF id or path of the base model")
    ap.add_argument("--architecture", help="override general.architecture (defaults to the GGUF's)")
    args = ap.parse_args()
    print(export(args.adapter_gguf, args.out_dir, args.base_model, args.architecture))
