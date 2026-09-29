"""Tensor payload codecs: quant (f32 | f16 | int8) and encoding.

Works on numpy arrays + bytes only -- no tinygrad, no GPU. Each family is a
small dict of dispatch functions per the scope doc's "dispatch points, not
scattered branches" rule: adding q4k or exi later is a new entry, not a rewrite.
"""
import base64 as _b64
from dataclasses import dataclass
import numpy as np

F32 = np.dtype("float32")
F16 = np.dtype("float16")


# ---------------------------------------------------------------- quant ----

def _f16_quantize(arr: np.ndarray):
    """float32 array -> (raw f16 bytes, scale=None)."""
    return arr.astype(F16).tobytes(), None


def _f16_dequantize(data: bytes, scale, shape):
    return np.frombuffer(data, dtype=F16).reshape(shape).astype(F32)


def _f32_quantize(arr: np.ndarray):
    """float32 array -> (raw f32 bytes, scale=None)."""
    return arr.astype(F32).tobytes(), None


def _f32_dequantize(data: bytes, scale, shape):
    return np.frombuffer(data, dtype=F32).reshape(shape).copy()


def _int8_quantize(arr: np.ndarray):
    """Per-tensor symmetric int8 quant: (int8 bytes, absmax scale)."""
    arr = arr.astype(F32, copy=False)
    absmax = float(np.abs(arr).max())
    scale = (absmax / 127.0) if absmax > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def _int8_dequantize(data: bytes, scale, shape):
    q = np.frombuffer(data, dtype=np.int8).reshape(shape)
    return (q.astype(F32) * scale).astype(F32)


@dataclass(frozen=True)
class TensorCodec:
    encode: object
    decode: object
    lossless_dtype: np.dtype | None
    ggml_type: int | None = None


# One authority for supported storage types, lossless selection, and GGUF mapping.
QUANT = {
    "f32": TensorCodec(_f32_quantize, _f32_dequantize, F32, 0),
    "f16": TensorCodec(_f16_quantize, _f16_dequantize, F16, 1),
    "int8": TensorCodec(_int8_quantize, _int8_dequantize, None),
}


def resolve_quant(arr: np.ndarray, policy: str, *, preserve_vectors: bool = False) -> str:
    """Resolve a storage policy before any dtype coercion can discard precision."""
    arr = np.asarray(arr)
    if policy == "lossless":
        matches = [name for name, spec in QUANT.items()
                   if spec.lossless_dtype is not None and spec.lossless_dtype == arr.dtype.newbyteorder("=")]
        if len(matches) != 1:
            raise ValueError(f"no lossless artifact codec for dtype {arr.dtype}")
        return matches[0]
    if policy not in QUANT:
        raise ValueError(f"unsupported artifact quantization: {policy}")
    # The GGUF-compatible package policy keeps F16 model vectors/scalars in F32.
    return "f32" if preserve_vectors and policy == "f16" and arr.ndim < 2 else policy


def lossless_equal(original: np.ndarray, restored: np.ndarray) -> bool:
    resolve_quant(original, "lossless")
    return (original.shape == restored.shape and
            restored.astype(original.dtype).tobytes() == original.tobytes())


def quantize(arr: np.ndarray, quant: str):
    """arr -> (payload_bytes, scale). Call resolve_quant to persist concrete metadata."""
    return QUANT[resolve_quant(arr, quant)].encode(arr)


def dequantize(data: bytes, scale, shape, quant: str) -> np.ndarray:
    return QUANT[quant].decode(data, scale, shape)


# ------------------------------------------------------------- encoding ----

def _base64_encode(data: bytes) -> str:
    return _b64.b64encode(data).decode("ascii")


def _base64_decode(s: str) -> bytes:
    return _b64.b64decode(s)


def _external_pack(blobs):
    """Concatenate tensor blobs into one buffer; return (buffer, [(offset, length), ...])."""
    buf = bytearray()
    offsets = []
    for b in blobs:
        offsets.append((len(buf), len(b)))
        buf += b
    return bytes(buf), offsets


def _external_unpack(buffer: bytes, offset: int, length: int) -> bytes:
    return buffer[offset:offset + length]


ENCODING = {
    "base64": (_base64_encode, _base64_decode),
    "external": (_external_pack, _external_unpack),
}


def gguf_file_type(specs, fallback):
    """Validate resolved tensor encodings; a package policy is not a tensor type."""
    formats = {spec.quant or fallback for spec in specs}
    if not formats:
        raise ValueError('cannot export an empty tensor package')
    unsupported = sorted(q for q in formats if q not in QUANT or QUANT[q].ggml_type is None)
    if unsupported:
        raise ValueError(f'no GGUF mapping for tensor formats: {unsupported}; supported: '
                         f'{sorted(q for q, spec in QUANT.items() if spec.ggml_type is not None)}')
    # GGUF file-type metadata: all F32=0, mostly F16 (with F32 vectors)=1.
    return 1 if 'f16' in formats else 0
