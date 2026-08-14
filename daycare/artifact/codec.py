"""Tensor payload codecs: quant (f16 | int8) and encoding (base64 | external).

Works on numpy arrays + bytes only -- no tinygrad, no GPU. Each family is a
small dict of dispatch functions per the scope doc's "dispatch points, not
scattered branches" rule: adding q4k or exi later is a new entry, not a rewrite.
"""
import base64 as _b64
import numpy as np


# ---------------------------------------------------------------- quant ----

def _f16_quantize(arr: np.ndarray):
    """float32 array -> (raw f16 bytes, scale=None)."""
    return arr.astype(np.float16).tobytes(), None


def _f16_dequantize(data: bytes, scale, shape):
    return np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)


def _int8_quantize(arr: np.ndarray):
    """Per-tensor symmetric int8 quant: (int8 bytes, absmax scale)."""
    absmax = float(np.abs(arr).max())
    scale = (absmax / 127.0) if absmax > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def _int8_dequantize(data: bytes, scale, shape):
    q = np.frombuffer(data, dtype=np.int8).reshape(shape)
    return (q.astype(np.float32) * scale).astype(np.float32)


QUANT = {
    "f16": (_f16_quantize, _f16_dequantize),
    "int8": (_int8_quantize, _int8_dequantize),
}


def quantize(arr: np.ndarray, quant: str):
    """arr -> (payload_bytes, scale)."""
    fn, _ = QUANT[quant]
    return fn(arr)


def dequantize(data: bytes, scale, shape, quant: str) -> np.ndarray:
    _, fn = QUANT[quant]
    return fn(data, scale, shape)


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
