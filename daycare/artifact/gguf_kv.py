"""GGUF format layer: typed KV read + the inverse (write).

Port, don't rewrite (research/xml-artifact-format-scope.md sec 4): tinygrad's
gguf reader already holds every format constant -- the KV type codes, the string
framing, the array framing. We reuse its primitive readers for the read
direction and add only the reverse direction here.

Why a typed reader at all: tinygrad's `gguf_load` returns KV *values* but drops
their type codes, and a writer needs the exact type (28 as u32 is not 28 as str).
Types are what make the XML artifact self-describing, so we capture them.

The tinygrad import is confined to `read_gguf` (below), not module scope: only
the read direction needs tinygrad's primitive readers. The write direction and
the format constants (MAGIC, GGML_FOR_QUANT, ...) are pure struct/stdlib, and
daycare/artifact/export.py's preflight (can_export) needs exactly those without
paying the vendored-tinygrad dependency -- see trainer_env.use_train_tinygrad.
"""
from __future__ import annotations

import io
import struct

# GGUF KV type codes -- mirrors tinygrad's `readers` table.
NAMES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
         7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
CODES = {v: k for k, v in NAMES.items()}
_PACK = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
         7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

# ggml tensor data types we emit
GGML_F32, GGML_F16 = 0, 1
GGML_FOR_QUANT = {"f16": GGML_F16, "f32": GGML_F32}

MAGIC, VERSION, ALIGNMENT = b"GGUF", 3, 32


# ------------------------------------------------------------------ read ----

def read_gguf(path: str):
    """Header-only read: -> (kv, tensor_infos, data_start). Loads no weights.

    Upstream's `gguf_load` dequantizes every tensor onto a device; when you only
    want to know what a file *is* (or to verify one we emitted) that is far too
    much. This is the same parse, stopped before the data section -- the xpkg's
    `read_index` for GGUF.

    kv:           [(key, type_code, value)]; arrays as (item_type, [values])
    tensor_infos: [(name, dims, ggml_type, offset)]  -- dims as stored (reversed)

    Needs the vendored train tinygrad (daycare/nursery/setup_trainer.sh) for
    its primitive readers -- imported lazily here, not at module scope, so
    the rest of this module (write side + constants) stays usable without it.
    """
    from ..nursery.trainer_env import use_train_tinygrad

    use_train_tinygrad()
    from tinygrad.llm.gguf import read_int32, read_int64, read_str, read_uint64, readers

    def _read_arr_typed(r):
        item_typ, n = read_int32(r), read_uint64(r)
        return item_typ, [readers[item_typ](r) for _ in range(n)]

    read_uint32 = readers[4]
    with open(path, "rb") as fh:
        r = io.BufferedReader(fh, 1_000_000)
        magic, _ver, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
        if magic != MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        kv = []
        for _ in range(n_kv):
            k, typ = read_str(r), read_int32(r)
            kv.append((k, typ, _read_arr_typed(r) if typ == 9 else readers[typ](r)))
        infos = []
        for _ in range(n_tensors):
            name = read_str(r)
            dims = tuple(read_uint64(r) for _ in range(read_uint32(r)))
            infos.append((name, dims, read_int32(r), read_uint64(r)))
        alignment = next((v for k, _t, v in kv if k == "general.alignment"), ALIGNMENT)
        pos = r.tell()
        data_start = ((pos + alignment - 1) // alignment) * alignment
    return kv, infos, data_start


def read_typed_kv(path: str) -> list[tuple[str, int, object]]:
    """[(key, type_code, value)] -- arrays come back as (item_type, [values])."""
    kv, _infos, _start = read_gguf(path)
    return kv


# ----------------------------------------------------------------- write ----

def w_str(f, s: str) -> None:
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def w_i32(f, v: int) -> None: f.write(struct.pack("<i", v))
def w_u32(f, v: int) -> None: f.write(struct.pack("<I", v))
def w_u64(f, v: int) -> None: f.write(struct.pack("<Q", v))


def w_value(f, typ: int, val) -> None:
    if typ == 8:
        w_str(f, val)
    elif typ == 9:
        item_typ, items = val
        w_i32(f, item_typ)
        w_u64(f, len(items))
        for it in items:
            w_value(f, item_typ, it)
    else:
        f.write(struct.pack(_PACK[typ], val))


def w_kv(f, key: str, typ: int, val) -> None:
    w_str(f, key)
    w_i32(f, typ)
    w_value(f, typ, val)


# ----------------------------------------------------- text <-> value ----

def to_text(typ: int, v) -> str:
    return "true" if (typ == 7 and v) else "false" if typ == 7 else str(v)


def from_text(typ: int, s: str):
    if typ == 8:
        return s
    if typ == 7:
        return s.lower() == "true"
    if typ in (6, 12):
        return float(s)
    return int(s)
