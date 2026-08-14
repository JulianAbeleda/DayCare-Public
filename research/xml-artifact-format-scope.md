# Scope: XML-Native Model Artifacts (implementation)

Exhaustive implementation scope for making the XML control plane real in DayCare:
one `<model>`/`<adapter>` XML artifact as the source of truth, with **GGUF (and
later HF/ONNX) as export targets emitted from it** — not formats the artifact
passes through. Architecture and rationale:
[knowledge_base/notes/xml-native-model-artifacts.md] (the reusable thesis); this
doc is the DayCare build plan that applies it.

Status: **scope only.** The *format* already exists as a scaffold
(`examples/daycare-model.xpkg/`); the *code* to read/write/emit it does not.

## 0. What already exists (do not reinvent)

- **The package layout** — `examples/daycare-model.xpkg/`: `manifest.xml`,
  `weights/tensor-index.xml` + `tensors-*.bin`, `tokenizer/` (`tokenizer.xml` +
  `vocab.blob`/`merges.blob`), `training/` (`plan.xml`, `run-ledger.xml`,
  `checkpoints/checkpoint-*.xml` + `.tensors`), `evals/`, `data/`,
  `compatibility/{gguf,hf,onnx}-map.xml` (placeholders), `exports/{gguf,hf}/`.
- **The weight encoding is already "external"** — `tensor-index.xml` names each
  tensor with `dtype/shape/file/offset/length` into a raw `.bin`. This is the
  XOP-style raw-attached form; base64-inline is the *small-artifact* variant.
- **arkey already encodes the whole GGUF format** (`tinygrad-arkey/tinygrad/
  llm/gguf.py`): `_gguf_parse` reads the header (`magic=GGUF`, n_tensors, n_kv),
  the typed KV codes, tensor infos, alignment (32), and the quant-type -> bytes
  table + block layouts (`Q8_0` id 8, `Q4_K` id 12 = 256 elem/144 B, `Q6_K` id 14,
  `F16`). It has no *writer*, but **the writer is that parser inverted** -- so we
  **port the format layer from arkey and only write the reverse direction**, not
  re-derive the spec. The GGUF *reader* (for base-model import, M3) is reused
  from arkey as-is.
- **The current artifact to migrate** — `daycare/nursery/consolidate.py` saves the
  LoRA adapter as `state/adapters/ada.safetensors` (safetensors = JSON header).
  This is Phase 1's target.

## 1. Definition of done

1. A DayCare library reads/writes the xpkg format (manifest + tensor-index +
   raw/base64 payloads) and loads it into tinygrad tensors.
2. A LoRA adapter is saved as a flat `<adapter>` XML (no safetensors/JSON).
3. A `<model>` xpkg can be **emitted to a valid GGUF** that arkey loads and serves,
   with **logits matching** the source within tolerance (the acceptance gate).
4. Optionally: arkey serves the xpkg **directly** (no GGUF), same logit-match.

Each is independently shippable (they map to the phases in the thesis note).

## 2. Schema (extend the existing xpkg minimally)

Keep the existing elements; add only what weight-learning needs.

**Tensor index** (`weights/tensor-index.xml`) -- add quant + encoding:

```xml
<tensors encoding="external">                 <!-- or "base64" for flat/small -->
  <tensor name="blk.0.attn_q.weight" dtype="int8" shape="2048,2048"
          quant="int8" scale="0.0073"          <!-- per-tensor; or per-block ref -->
          file="tensors-0000.bin" offset="0" length="4194304"/>
  <!-- base64 variant: drop file/offset, inline a <data> child -->
</tensors>
```

**Adapter** (`<adapter>` -- new, small, flat by default):

```xml
<adapter subject="Ada" base="qwen3-0.6b-q8" method="lora">
  <lora r="16" alpha="32" last-k="8" targets="attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down"/>
  <training epochs="3" loss-start="1.64" loss-end="0.28">
    <eval metric="identity-consistency" before="0/2" after="2/2" gate="off"/>
  </training>
  <tensors encoding="base64" quant="int8"> ... </tensors>
</adapter>
```

**Reuse as-is:** `manifest.xml` (compatibility targets, refs), `tokenizer.xml`,
`training/*`, `evals/*`. The `encoding` attribute is the knob (external | base64 |
exi-later); `quant` is the orthogonal size lever (f16 | int8 | k-quant-later).

## 3. Core library (`daycare/artifact/`)

Substrate-independent, stdlib + the vendored train tinygrad for tensor I/O.

```text
daycare/artifact/
  index.py       tensor-index read/write: (name,dtype,shape,quant,scale) <-> payload
  codec.py       payload encoders behind one interface:
                   external (raw .bin + offsets) | base64 (inline) | [exi later]
                 and quant codecs: f16 (passthrough) | int8+scale (quantize/dequant)
  manifest.py    read/write <model>/<adapter> manifests (xml.etree)
  load.py        xpkg -> dict[name -> tinygrad Tensor]  (for serve / merge)
  save.py        tinygrad tensors + metadata -> xpkg (or flat <adapter> xml)
```

Design rules: `encoding` and `quant` are dispatch points (a dict of codec
functions), not branches scattered through the code -- so `external`->`exi` or
`int8`->`q4k` is a new function, not a rewrite. Every tensor round-trips
(`save` then `load` reproduces values within quant tolerance) -- a unit test.
**Reuse over write:** tensor bytes I/O leans on tinygrad's `nn.state`
(`safe_load`/save, or raw `Tensor` <-> bytes); GGUF read reuses arkey's
`gguf_load`; the GGUF quant/format tables are ported from arkey's `gguf.py`
(sec 4). Author only what genuinely doesn't already exist.

## 4. The GGUF emitter (`daycare/artifact/emit_gguf.py`) -- the "make a GGUF" deliverable

**Port, don't rewrite.** arkey's `gguf.py` `_gguf_parse` already holds every
format constant we need — the KV type enum, the read helpers (`read_str`,
`read_arr`, `_read_unpack`), the quant-type -> (n_elem, n_bytes) table, the block
layouts, the alignment rule. The emitter is the *inverse*: mirror `_gguf_parse`
into a `_gguf_write` (reuse its type codes and quant tables verbatim, invert the
struct packing). New code is only the write direction + the KV-key mapping — the
GGUF *knowledge* is copied, not re-derived. Likewise the base-model import (M3)
calls arkey's existing `gguf_load` directly.

Write a GGUF v3 file from an xpkg. GGUF is a serializer over the same objects:

```text
[header]   magic "GGUF" | version=3 | n_tensors | n_kv
[kv]       n_kv typed key/value pairs   <- from <config> + gguf-map.xml
[tinfos]   per tensor: name, n_dims, shape, ggml_type, offset
[pad]      align to general.alignment (32)
[data]     tensor blobs (ggml_type layout)
```

- **KV mapping** -- fill `compatibility/gguf-map.xml` (currently a placeholder)
  with the XML-attr -> GGUF-key table: `arch -> general.architecture`,
  `dim -> {arch}.embedding_length`, `n_heads -> {arch}.attention.head_count`,
  `n_kv_heads -> ...head_count_kv`, `rope_theta`, `qk_norm`, plus
  `tokenizer.ggml.*` from `tokenizer.xml`. Data-as-XML, per the ethos.
- **Quant on emit** (staged):
  - **F16 first** -- dequant int8->f16, emit `ggml_type=F16`. Simplest, lossless
    from f16, arkey already loads it. Proves the whole path.
  - **Q8_0 next** -- note the mismatch: GGUF Q8_0 is *per-block-of-32 int8 + one
    fp16 scale per block*, but our adapter stores *per-tensor* scale. Emitting true
    Q8_0 requires per-block re-quantization on export (a real transcode). Scope it
    as a codec in `codec.py`, gated by a logit-match.
  - **Q4_K/Q6_K** -- deferred; needs faithful super-block layout (Q4_K: 256 elem /
    144-byte block: d,dmin,scales,qs). Only if serving size demands it.
- **Acceptance gate:** emit -> `TinygradQwen3` loads the GGUF -> compare greedy
  logits / a few completions against the xpkg-native load. Promote only on match.
- Output lands in `exports/gguf/` (the dir already exists).

## 5. Other emitters (deferred, same shape)

`emit_hf.py` (safetensors + config.json -- ironically re-adds JSON, but only as an
*export* for the HF ecosystem) and `emit_onnx.py`. Same pattern: read xpkg, map
metadata via `hf-map.xml`/`onnx-map.xml`, write. Build only when needed.

## 6. Integration with the training flow

```text
Phase 1  consolidate.py: replace safe_save(...safetensors) with
         save.write_adapter(loras, meta) -> state/adapters/ada.adapter.xml
         (flat, base64+int8). Kills the safetensors JSON header.

Phase 2  merge.py: base <model> xpkg + <adapter> -> merged <model> xpkg
         (fold LoRA into the base tensors; write a new tensor-index + .bin).
         One-time base import: convert the Qwen3 Q8 GGUF -> xpkg via a
         read-arkey-gguf -> write-xpkg step (reuses arkey's gguf reader).

Phase 3a bridge: emit_gguf(merged xpkg) -> arkey serves the GGUF. Ships now;
         keeps the proven serve path while XML is already the source of truth.

Phase 3b native: load.py builds tinygrad tensors from the xpkg directly; a thin
         arkey entry point serves without GGUF. The endpoint.
```

Phase 3a is the pragmatic bridge; 3b is the purity endpoint. Both validated by
logit-match.

## 7. Validation strategy

- **Round-trip unit tests** (fast, no GPU): `save`->`load` reproduces tensors
  within quant tolerance; manifest attrs preserved.
- **GGUF emit gate** (GPU): xpkg -> GGUF -> arkey load -> greedy logits match the
  xpkg-native load on a fixed prompt set. Same discipline used to validate the
  training forward against arkey.
- **Adapter equivalence:** the `<adapter>` XML, loaded + merged, reproduces the
  post-training model that produced `identity 2/2` -- guards the Phase-1 swap.

## 8. Milestones (concretized from the thesis note's 0..4)

```text
M1  core lib (index/codec/manifest/load/save) + round-trip tests           [no GPU]
M2  Phase 1: adapter -> flat <adapter> XML in consolidate.py                [no GPU]
M3  base Qwen3 GGUF -> <model> xpkg importer (reuse arkey gguf reader)      [light]
M4  emit_gguf (F16): port arkey gguf format layer -> invert to a writer;
    logit-match gate against arkey                                         [GPU]
M5  merge adapter -> model xpkg; emit_gguf; serve via arkey (bridge)       [GPU]
M6  Q8_0 emit codec (per-block requant) + gate                             [GPU]
M7  native xpkg loader -> arkey serves without GGUF                        [GPU]
```

M1-M2 are pure Python and ship immediately (they're the XML win with zero GPU).
GGUF work (M4+) waits for a free GPU, per the training-run operational notes.

## 9. Open decisions / risks

1. **Default encoding per artifact size.** Adapters -> `base64` flat file
   (self-contained). Full models -> `external` (mmap-able, no base64 tax).
   Recommendation: pick by size, both behind the same schema.
2. **GGUF quant target.** F16 (simple, bigger) vs Q8_0 (compact, needs per-block
   requant). Recommendation: F16 to prove the path, Q8_0 as a gated codec.
3. **Native serve (3b) vs GGUF bridge (3a) priority.** The bridge delivers XML
   source-of-truth *now*; native serve is the purity payoff later. Recommend
   bridge first.
4. **Tokenizer representation.** `tokenizer.xml` currently refs `vocab.blob`/
   `merges.blob`. For GGUF emit we need `tokenizer.ggml.*` KV -- map blob -> KV.
5. **base64 vs raw for the flat adapter.** base64 (+33%) keeps one text file;
   raw-external is smaller but two files. For a ~2 MB adapter, base64 wins on
   single-file simplicity.
6. **Where the emitter/loader live.** `daycare/artifact/` (new) -- reusable by the
   nursery (save adapters) and a future serve path. Not in `substrate/` (that's
   arkey-serve) or `nursery/` (that's training).

## 10. Non-goals

- Modifying arkey or tinygrad (we *read* arkey's GGUF loader; we don't add a
  writer to it).
- Encoding tensor *numbers* as XML text (control plane only; see the thesis note).
- Building HF/ONNX emitters or EXI before a measurement demands them.
- A general model zoo -- this targets Qwen3-0.6B / models first.

## 11. First slice

**M1 + M2**, pure Python, no GPU: the core `daycare/artifact/` lib with round-trip
tests, then swap `consolidate.py`'s `safe_save` for a flat `<adapter>` XML. That
alone delivers the Phase-1 win (the artifact you author is XML, JSON gone) and
lays the rails every later phase — including the GGUF emitter — builds on.
