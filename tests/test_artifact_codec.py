import numpy as np

from daycare.artifact import codec
from daycare.artifact.index import TensorSpec, to_element, from_element


def _relerr(a, b):
    return np.abs(a - b).max() / (np.abs(a).max() + 1e-8)


def test_f16_base64_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((4, 8)).astype(np.float32)

    payload, scale = codec.quantize(arr, "f16")
    s = codec.ENCODING["base64"][0](payload)
    payload2 = codec.ENCODING["base64"][1](s)
    out = codec.dequantize(payload2, scale, arr.shape, "f16")

    assert np.array_equal(out, arr.astype(np.float16).astype(np.float32))


def test_int8_base64_roundtrip():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((4, 8)).astype(np.float32)

    payload, scale = codec.quantize(arr, "int8")
    s = codec.ENCODING["base64"][0](payload)
    payload2 = codec.ENCODING["base64"][1](s)
    out = codec.dequantize(payload2, scale, arr.shape, "int8")

    assert _relerr(arr, out) < 0.05


def test_f16_external_roundtrip():
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((4, 8)).astype(np.float32)

    payload, scale = codec.quantize(arr, "f16")
    buffer, offsets = codec.ENCODING["external"][0]([payload])
    offset, length = offsets[0]
    payload2 = codec.ENCODING["external"][1](buffer, offset, length)
    out = codec.dequantize(payload2, scale, arr.shape, "f16")

    assert np.array_equal(out, arr.astype(np.float16).astype(np.float32))


def test_int8_external_roundtrip():
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((4, 8)).astype(np.float32)

    payload, scale = codec.quantize(arr, "int8")
    buffer, offsets = codec.ENCODING["external"][0]([payload])
    offset, length = offsets[0]
    payload2 = codec.ENCODING["external"][1](buffer, offset, length)
    out = codec.dequantize(payload2, scale, arr.shape, "int8")

    assert _relerr(arr, out) < 0.05


def test_tensors_element_roundtrip_external():
    specs = [
        TensorSpec(name="blk.0.attn_q.weight", dtype="int8", shape=(4, 8),
                   quant="int8", scale=0.0073, encoding="external",
                   file="tensors-0000.bin", offset=0, length=32),
        TensorSpec(name="blk.0.attn_k.weight", dtype="F16", shape=(2, 2),
                   quant=None, scale=None, encoding="external",
                   file="tensors-0000.bin", offset=32, length=8),
    ]
    elem = to_element(specs, encoding="external")
    encoding, specs2 = from_element(elem)

    assert encoding == "external"
    assert specs2 == specs


def test_tensors_element_roundtrip_base64():
    specs = [
        TensorSpec(name="blk.0.attn_q.weight", dtype="int8", shape=(4, 8),
                   quant="int8", scale=0.0073, encoding="base64",
                   data="QUJDRA=="),
    ]
    elem = to_element(specs, encoding="base64")
    encoding, specs2 = from_element(elem)

    assert encoding == "base64"
    assert specs2 == specs


def test_lossless_policy_rejects_unsupported_types_before_coercion():
    import pytest
    for dtype in (np.float16, np.float32):
        source = np.array([0.125, -0.0, 1.5], dtype=dtype)
        quant = codec.resolve_quant(source, 'lossless')
        data, scale = codec.quantize(source, quant)
        restored = codec.dequantize(data, scale, source.shape, quant)
        assert codec.lossless_equal(source, restored)
    for dtype in (np.float64, np.int32, np.int8):
        with pytest.raises(ValueError, match='no lossless'):
            codec.resolve_quant(np.ones(2, dtype=dtype), 'lossless')
    with pytest.raises(ValueError, match='unsupported artifact quantization'):
        codec.resolve_quant(np.ones(2), 'invented')
