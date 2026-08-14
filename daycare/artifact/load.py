"""Load a flat <adapter> XML back to (manifest, {name: numpy array}). No GPU.

The inverse of save.write_adapter: read the manifest, parse the opaque <tensors>
block via index, base64-decode + dequantize each payload. Callers that need
tinygrad Tensors wrap the arrays at their boundary (keeps this GPU-free).
"""
from __future__ import annotations

import numpy as np

from . import codec, index
from . import manifest as mf


def load_adapter(path: str):
    man = mf.read(path)
    _, specs = index.from_element(man.tensors_el)
    tensors: dict[str, np.ndarray] = {}
    for s in specs:
        raw = codec.ENCODING["base64"][1](s.data)
        tensors[s.name] = codec.dequantize(raw, s.scale, s.shape, s.quant)
    return man, tensors
