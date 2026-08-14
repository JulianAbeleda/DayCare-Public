"""Tokenizer in the .xpkg: scalars as readable XML, big arrays as blobs.

GGUF carries the tokenizer as KV: a few scalars (model, pre, bos/eos/pad ids,
chat template) plus three *huge* arrays -- tokens (151,936 strings), token_type
(151,936 ints) and merges (151,387 strings). Those arrays are exactly the case
the thesis note calls out: they do not belong as XML text. So:

    tokenizer/
      tokenizer.xml     scalars + counts + blob refs + <chat-template> (readable)
      vocab.blob        length-prefixed UTF-8 tokens
      merges.blob       length-prefixed UTF-8 merge rules
      token-types.blob  int32 array

Blob format is length-prefixed (u32 LE + utf-8 bytes) rather than newline- or
NUL-separated, because byte-level BPE tokens can contain any byte.
"""
from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

TOKENIZER_REF = "tokenizer/tokenizer.xml"
VOCAB, MERGES, TYPES = "vocab.blob", "merges.blob", "token-types.blob"


def _write_strings(path: str, items) -> int:
    n = 0
    with open(path, "wb") as f:
        for s in items:
            b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
            f.write(struct.pack("<I", len(b)))
            f.write(b)
            n += 1
    return n


def _read_strings(path: str) -> list[str]:
    out = []
    with open(path, "rb") as f:
        blob = f.read()
    i = 0
    while i < len(blob):
        (ln,) = struct.unpack_from("<I", blob, i)
        i += 4
        out.append(blob[i:i + ln].decode("utf-8", errors="replace"))
        i += ln
    return out


def write_tokenizer(pkg_dir: str, kv: dict) -> str:
    """Split GGUF tokenizer KV into readable XML + blobs. Returns the manifest ref."""
    tdir = os.path.join(pkg_dir, "tokenizer")
    os.makedirs(tdir, exist_ok=True)

    n_vocab = _write_strings(os.path.join(tdir, VOCAB), kv["tokenizer.ggml.tokens"])
    n_merges = _write_strings(os.path.join(tdir, MERGES), kv.get("tokenizer.ggml.merges", []))
    types = np.asarray(kv.get("tokenizer.ggml.token_type", []), dtype=np.int32)
    types.tofile(os.path.join(tdir, TYPES))

    root = ET.Element("tokenizer", {
        "model": str(kv.get("tokenizer.ggml.model", "")),
        "pre": str(kv.get("tokenizer.ggml.pre", "")),
        "bos": str(kv.get("tokenizer.ggml.bos_token_id", "")),
        "eos": str(kv.get("tokenizer.ggml.eos_token_id", "")),
        "pad": str(kv.get("tokenizer.ggml.padding_token_id", "")),
        "add-bos": str(bool(kv.get("tokenizer.ggml.add_bos_token", False))).lower(),
    })
    ET.SubElement(root, "vocab", path=VOCAB, count=str(n_vocab))
    ET.SubElement(root, "merges", path=MERGES, count=str(n_merges))
    ET.SubElement(root, "token-types", path=TYPES, count=str(types.size))
    if (tmpl := kv.get("tokenizer.chat_template")):
        ET.SubElement(root, "chat-template").text = tmpl

    ET.indent(root)
    ET.ElementTree(root).write(os.path.join(pkg_dir, TOKENIZER_REF),
                               encoding="utf-8", xml_declaration=False)
    return TOKENIZER_REF


def read_tokenizer(pkg_dir: str) -> dict:
    """Reconstruct the GGUF tokenizer KV dict from the xpkg (for re-emit)."""
    path = os.path.join(pkg_dir, TOKENIZER_REF)
    root = ET.parse(path).getroot()
    tdir = os.path.dirname(path)

    def _p(el_name, attr="path"):
        el = root.find(el_name)
        return os.path.join(tdir, el.get(attr)) if el is not None else None

    kv = {
        "tokenizer.ggml.model": root.get("model"),
        "tokenizer.ggml.pre": root.get("pre"),
        "tokenizer.ggml.tokens": _read_strings(_p("vocab")),
        "tokenizer.ggml.merges": _read_strings(_p("merges")),
        "tokenizer.ggml.token_type": np.fromfile(_p("token-types"), dtype=np.int32).tolist(),
        "tokenizer.ggml.add_bos_token": root.get("add-bos") == "true",
    }
    for attr, key in (("bos", "bos_token_id"), ("eos", "eos_token_id"), ("pad", "padding_token_id")):
        if (v := root.get(attr)):
            kv[f"tokenizer.ggml.{key}"] = int(v)
    if (t := root.find("chat-template")) is not None and t.text:
        kv["tokenizer.chat_template"] = t.text
    return kv
