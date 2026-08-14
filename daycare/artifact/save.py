"""Compose codec + index + manifest into a flat <adapter> XML (numpy, no GPU).

write_adapter takes plain numpy arrays (caller does tensor.numpy()), quantizes +
base64-encodes each into an inline <tensors> block, and wraps it in an
AdapterManifest. One self-contained .adapter.xml file, no safetensors, no JSON.
"""
from __future__ import annotations

import os

import numpy as np

from . import codec, index
from . import manifest as mf


def write_adapter(tensors: dict[str, np.ndarray], meta: dict, path: str, quant: str = "int8") -> None:
    specs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        payload, scale = codec.quantize(arr, quant)
        b64 = codec.ENCODING["base64"][0](payload)
        specs.append(index.TensorSpec(
            name=name, dtype=quant, shape=tuple(arr.shape),
            quant=quant, scale=scale, encoding="base64", data=b64,
        ))
    tensors_el = index.to_element(specs, encoding="base64")
    man = mf.AdapterManifest(
        subject=meta["subject"], base=meta["base"], r=meta["r"], alpha=meta["alpha"],
        last_k=meta["last_k"], targets=list(meta["targets"]), epochs=meta["epochs"],
        loss_start=meta["loss_start"], loss_end=meta["loss_end"],
        eval_metric=meta["eval_metric"], eval_before=meta["eval_before"],
        eval_after=meta["eval_after"], eval_gate=meta["eval_gate"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mf.write(man, path, tensors_el=tensors_el)
