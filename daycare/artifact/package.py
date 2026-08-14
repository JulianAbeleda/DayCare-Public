"""The .xpkg package: a loose directory with an XML control plane + raw .bin.

    <pkg>/
      manifest.xml                 <model> -- config, tokenizer ref, tensors ref
      weights/
        tensor-index.xml           <tensors encoding="external"> -- shapes/scales/offsets
        tensors-0000.bin           raw quantized bytes, concatenated

This is the end-product form (research/xml-artifact-format-scope.md): the weights
stay raw and uncompressed so a runtime can **mmap** them and use them in place
(zero decode, zero copy), while the XML stays readable -- you `cat` the index to
see what is in the blob without touching a byte of it. No base64 tax, no JSON.

Contrast with save.py's flat <adapter> XML (base64-inline), which is the
single-file convenience form for small artifacts.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import codec, index
from . import manifest as mf

BLOB = "tensors-0000.bin"
INDEX_REF = "weights/tensor-index.xml"


def _write_xml(el: ET.Element, path: str) -> None:
    ET.indent(el)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(el).write(path, encoding="utf-8", xml_declaration=False)


def write_package(pkg_dir: str, manifest: mf.ModelManifest, tensors, quant: str = "int8") -> None:
    """Write a .xpkg: raw .bin payload + XML index + manifest.

    `tensors` is a dict OR an iterable of (name, array) pairs -- the payload is
    streamed tensor-by-tensor straight to the .bin, so memory stays bounded by
    the largest single tensor (a 600M-param model must not be buffered whole).
    """
    items = tensors.items() if isinstance(tensors, dict) else tensors
    os.makedirs(os.path.join(pkg_dir, "weights"), exist_ok=True)
    specs, offset = [], 0
    with open(os.path.join(pkg_dir, "weights", BLOB), "wb") as f:
        for name, arr in items:
            arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
            payload, scale = codec.quantize(arr, quant)
            f.write(payload)                              # raw, uncompressed
            specs.append(index.TensorSpec(
                name=name, dtype=quant, shape=tuple(arr.shape),
                quant=quant, scale=scale, encoding="external", file=BLOB,
                offset=offset, length=len(payload),
            ))
            offset += len(payload)
    _write_xml(index.to_element(specs, encoding="external"),
               os.path.join(pkg_dir, INDEX_REF))
    manifest.quant = quant
    manifest.tensors_ref = INDEX_REF
    _write_xml(manifest.to_xml(), os.path.join(pkg_dir, "manifest.xml"))


def read_index(pkg_dir: str):
    """Read just the control plane -- manifest + tensor specs, no weight bytes."""
    manifest = mf.read(os.path.join(pkg_dir, "manifest.xml"))
    _, specs = index.read_xml(path=os.path.join(pkg_dir, manifest.tensors_ref))
    return manifest, specs


def read_package(pkg_dir: str, names: list[str] | None = None):
    """Read a .xpkg -> (manifest, {name: float32 array}).

    The blob is memory-mapped: nothing is copied until a tensor is actually
    dequantized, and `names` lets you pull only what you asked for.
    """
    manifest, specs = read_index(pkg_dir)
    blob_path = os.path.join(pkg_dir, "weights", BLOB)
    mm = np.memmap(blob_path, dtype=np.uint8, mode="r")   # zero-copy view
    out: dict[str, np.ndarray] = {}
    for s in specs:
        if names is not None and s.name not in names:
            continue
        raw = mm[s.offset:s.offset + s.length].tobytes()
        out[s.name] = codec.dequantize(raw, s.scale, s.shape, s.quant)
    return manifest, out
