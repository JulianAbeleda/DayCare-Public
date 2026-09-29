"""PEFT export of a llama.cpp LoRA GGUF: names, orientation, scale, exact factors (NumPy only)."""
import json

import numpy as np
import pytest

from daycare.artifact import gguf_kv as g
from daycare.artifact.peft_export import export, read_lora_gguf, read_safetensors


def write_lora_gguf(path, tensors, alpha=64.0, arch="nemotron_h"):
    """The same layout daycare/artifact/lora_gguf.py writes: f32 factors, numpy shapes reversed as dims."""
    pairs = [("general.type", 8, "adapter"), ("general.architecture", 8, arch),
             ("adapter.type", 8, "lora"), ("adapter.lora.alpha", 6, alpha)]
    payloads, offset = [], 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, np.float32).tobytes()
        payloads.append((name, arr.shape, offset, raw))
        offset += (len(raw) + g.ALIGNMENT - 1) // g.ALIGNMENT * g.ALIGNMENT
    with open(path, "wb") as f:
        f.write(g.MAGIC); g.w_i32(f, g.VERSION); g.w_u64(f, len(payloads)); g.w_u64(f, len(pairs))
        for key, typ, value in pairs:
            g.w_kv(f, key, typ, value)
        for name, shape, off, _ in payloads:
            g.w_str(f, name); g.w_u32(f, len(shape))
            for dim in reversed(shape):
                g.w_u64(f, int(dim))
            g.w_i32(f, g.GGML_F32); g.w_u64(f, off)
        f.write(b"\0" * ((-f.tell()) % g.ALIGNMENT))
        for _, _, _, raw in payloads:
            f.write(raw); f.write(b"\0" * ((-len(raw)) % g.ALIGNMENT))


def factors(seed=0, r=4, d=6, ff=10):
    rng = np.random.default_rng(seed)
    return {
        "blk.3.ffn_up.weight.lora_a": rng.standard_normal((r, d)).astype(np.float32),
        "blk.3.ffn_up.weight.lora_b": rng.standard_normal((ff, r)).astype(np.float32),
        "blk.3.ffn_down.weight.lora_a": rng.standard_normal((r, ff)).astype(np.float32),
        "blk.3.ffn_down.weight.lora_b": rng.standard_normal((d, r)).astype(np.float32),
    }


def test_export_copies_factors_exactly_with_hf_names_and_scale(tmp_path):
    src = factors()
    write_lora_gguf(tmp_path / "a.gguf", src, alpha=8.0)
    kv, back = read_lora_gguf(tmp_path / "a.gguf")
    assert kv["adapter.lora.alpha"] == 8.0 and all(np.array_equal(back[k], v) for k, v in src.items())

    out = export(tmp_path / "a.gguf", tmp_path / "peft", "org/base")
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["base_model_name_or_path"] == "org/base"
    assert (config["r"], config["lora_alpha"]) == (4, 8)
    assert config["target_modules"] == ["down_proj", "up_proj"]
    assert (config["layers_to_transform"], config["layers_pattern"]) == ([3], "layers")

    state = read_safetensors(out / "adapter_model.safetensors")
    pre = "base_model.model.backbone.layers.3.mixer"
    assert set(state) == {f"{pre}.{m}.lora_{p}.weight" for m in ("up_proj", "down_proj") for p in "AB"}
    for gguf_name, hf_name in (("ffn_up", "up_proj"), ("ffn_down", "down_proj")):
        a, b = state[f"{pre}.{hf_name}.lora_A.weight"], state[f"{pre}.{hf_name}.lora_B.weight"]
        assert np.array_equal(a, src[f"blk.3.{gguf_name}.weight.lora_a"])
        assert np.array_equal(b, src[f"blk.3.{gguf_name}.weight.lora_b"])
        # PEFT: W + (lora_alpha/r) * B @ A == llama.cpp: W + (alpha/r) * lora_b @ lora_a
        gguf_delta = 8.0 / 4 * src[f"blk.3.{gguf_name}.weight.lora_b"] @ src[f"blk.3.{gguf_name}.weight.lora_a"]
        np.testing.assert_array_equal(config["lora_alpha"] / config["r"] * b @ a, gguf_delta)


def test_mixed_layer_sets_fall_back_to_a_suffix_regex(tmp_path):
    src = factors()
    rng = np.random.default_rng(1)
    src["blk.2.ssm_in.weight.lora_a"] = rng.standard_normal((4, 6)).astype(np.float32)
    src["blk.2.ssm_in.weight.lora_b"] = rng.standard_normal((12, 4)).astype(np.float32)
    write_lora_gguf(tmp_path / "a.gguf", src)
    config = json.loads((export(tmp_path / "a.gguf", tmp_path / "peft", "b") / "adapter_config.json").read_text())
    import re
    pattern = re.compile(config["target_modules"])
    assert config["layers_to_transform"] is None
    for name in ("backbone.layers.2.mixer.in_proj", "model.layers.3.mixer.up_proj", "model.layers.3.mixer.down_proj"):
        assert pattern.fullmatch(name)
    for name in ("model.layers.3.mixer.in_proj", "model.layers.2.mixer.up_proj", "model.layers.13.mixer.up_proj"):
        assert not pattern.fullmatch(name)


@pytest.mark.parametrize("name,arch", [("blk.3.attn_q.weight", "nemotron_h"), ("blk.3.ffn_up.weight", "llama")])
def test_unverified_targets_and_architectures_are_refused(tmp_path, name, arch):
    rng = np.random.default_rng(2)
    write_lora_gguf(tmp_path / "a.gguf", {f"{name}.lora_a": rng.standard_normal((2, 4)).astype(np.float32),
                                          f"{name}.lora_b": rng.standard_normal((4, 2)).astype(np.float32)}, arch=arch)
    with pytest.raises(ValueError, match="verified"):
        export(tmp_path / "a.gguf", tmp_path / "peft", "b")
