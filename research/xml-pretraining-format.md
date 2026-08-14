# XML Pretraining Format

This note maps a hypothetical XML-first format for pretraining. The goal is a
mental model, not a committed standard. The format should make model training
auditable without forcing expensive pretraining as the first move.

Concrete scaffolds live in:

- `examples/xml-pretraining-repo/`
- `examples/daycare-model.xpkg/`

The layout follows common GitHub ML conventions: `README.md`, `data/`,
`configs/`, `src/`, `scripts/`, `models/` or `artifacts/`, `docs/`, and
`tests/`. Hugging Face model repositories also commonly separate config,
tokenizer, weights, and README/model-card material. The XML version keeps that
separation readable while making XML the control plane.

## Why XML Over The Current Training Layout

The current training ecosystem is maintainable only when a project has strong
conventions. The actual state of a model is usually spread across config files,
Python code, dataset scripts, tokenizer files, README notes, binary weights,
training logs, and export-specific metadata. That works for active development,
but it is weak as an artifact format because the relationship between the data,
training recipe, tokenizer, architecture, checkpoints, evals, and exported
weights is implicit.

The XML argument is not that tensor bytes should become XML text. The argument
is that a model should have one inspectable control plane that names every
important object and records how those objects fit together.

For DayCare, the target shape is:

```text
one archive
  -> one XML manifest
  -> indexed data records
  -> indexed tokenizer blobs
  -> indexed weight tensors
  -> training plan and run ledger
  -> eval records and audit notes
```

That makes the artifact easier to maintain because the format has one place to
answer basic questions:

- what data was used;
- which tokenizer maps that data into tokens;
- which architecture consumed those tokens;
- which tensors belong to that architecture;
- which training run produced each checkpoint;
- which evals and audits justify keeping the artifact;
- how to export the artifact to GGUF, Hugging Face, or ONNX.

The maintainability win is the single-file or single-package contract. A tool can
inspect the model and the data through the same schema instead of reverse
engineering several repo conventions. The binary blobs can stay compact, but the
manifest makes their meaning, offsets, shapes, hashes, provenance, and training
context explicit.

Lifecycle comparison:

- [Current weight lifecycle](assets/current-weight-lifecycle.md)
- [DayCare flat XML weight lifecycle](assets/daycare-flat-xml-weight-lifecycle.md)

## Shared Concepts

Every design needs these XML records:

```text
dataset manifest
tokenizer manifest
architecture manifest
training plan
checkpoint/tensor index
evaluation plan
run ledger
release manifest
compatibility map
```

The key distinction:

```text
XML = control plane, metadata, schema, provenance
raw bytes = tensor/tokenizer/data payloads
```

## Option 1: GitHub Repo Layout

This is the easiest MVP. It is a normal repo with XML manifests.

```text
daycare-pretrain/
  README.md
  LICENSE
  pyproject.toml

  configs/
    model.xml
    tokenizer.xml
    optimizer.xml
    schedule.xml
    runtime-targets.xml

  data/
    raw/
      sources.xml
      web/
      repos/
      chats/
    interim/
      extraction-ledger.xml
      dedupe-ledger.xml
    processed/
      dataset.xml
      train.xml
      validation.xml
      holdout.xml

  src/
    daycare_pretrain/
      __init__.py
      ingest.py
      validate_xml.py
      train.py
      export.py

  scripts/
    ingest_sources.sh
    validate_dataset.sh
    train_sft.sh
    export_gguf.sh

  models/
    base/
      base-model.xml
    adapters/
      adapter.xml
    checkpoints/
      checkpoint-000100.xml
      checkpoint-000200.xml

  artifacts/
    inspections/
      model-archaeology.xml
    evals/
      eval-plan.xml
      eval-run-0001.xml
    exports/
      hf/
      gguf/

  docs/
    data-card.xml
    model-card.xml
    risk-card.xml

  tests/
    test_schema.py
    test_dataset_splits.py
    fixtures/
      tiny-dataset.xml
```

Core files:

```xml
<!-- configs/model.xml -->
<model id="daycare-small-v0">
  <architecture family="llama" layers="24" hidden-size="2048"/>
  <context length="4096"/>
  <tokenizer ref="configs/tokenizer.xml"/>
</model>
```

```xml
<!-- data/processed/dataset.xml -->
<dataset id="daycare-corpus-v0">
  <split name="train" path="train.xml" sha256="..."/>
  <split name="validation" path="validation.xml" sha256="..."/>
  <split name="holdout" path="holdout.xml" sha256="..."/>
  <source-policy>Keep final promotion gates out of train.</source-policy>
</dataset>
```

Best use:

- early research;
- hand editing;
- GitHub review;
- schema iteration;
- BoltBeam validation;
- training-tool export.

Tradeoff:

- many files, but every part is inspectable.

## Option 2: XML Model Package Directory

This is closer to a model repository. It is a canonical package that can be
loaded, audited, or exported.

```text
daycare-model.xpkg/
  manifest.xml
  README.md
  LICENSE

  data/
    dataset.xml
    records/
      shard-0000.xml
      shard-0001.xml

  training/
    plan.xml
    optimizer.xml
    run-ledger.xml
    checkpoints/
      checkpoint-000100.xml
      checkpoint-000100.tensors

  tokenizer/
    tokenizer.xml
    vocab.blob
    merges.blob

  weights/
    tensor-index.xml
    tensors-0000.bin
    tensors-0001.bin

  evals/
    eval-plan.xml
    eval-run-0001.xml
    failures.md

  compatibility/
    hf-map.xml
    gguf-map.xml
    onnx-map.xml

  exports/
    hf/
    gguf/
```

Package manifest:

```xml
<model-package id="daycare-model" version="0.1">
  <dataset ref="data/dataset.xml"/>
  <training ref="training/plan.xml"/>
  <tokenizer ref="tokenizer/tokenizer.xml"/>
  <weights ref="weights/tensor-index.xml"/>
  <evals ref="evals/eval-plan.xml"/>
  <compatibility>
    <target name="hf" map="compatibility/hf-map.xml"/>
    <target name="gguf" map="compatibility/gguf-map.xml"/>
  </compatibility>
</model-package>
```

Tensor index:

```xml
<tensors>
  <tensor name="layers.0.attn.q_proj.weight"
          dtype="F16"
          shape="2048,2048"
          file="tensors-0000.bin"
          offset="0"
          length="8388608"/>
</tensors>
```

Best use:

- canonical DayCare model artifact;
- BoltBeam import/export;
- audit trail attached to model;
- compatibility with GGUF/HF exporters.

Tradeoff:

- more formal than a repo layout;
- needs package validator and versioning.

## Option 3: Single-File XML Archive

This is the most aggressive design. It minimizes file sprawl.

```text
daycare-model.xmlpack
  [magic bytes]
  [version]
  [xml_header_length]
  [XML manifest]
  [binary blob region]
```

The XML manifest indexes every blob:

```xml
<xmlpack id="daycare-model" version="0.1">
  <architecture family="llama" layers="24" hidden-size="2048"/>

  <blobs>
    <blob id="tokenizer.vocab"
          media-type="application/octet-stream"
          offset="1048576"
          length="500000"/>
    <blob id="tensor.layers.0.attn.q_proj.weight"
          dtype="F16"
          shape="2048,2048"
          offset="1548576"
          length="8388608"/>
  </blobs>

  <training>
    <method>causal-lm</method>
    <dataset id="daycare-corpus-v0"/>
  </training>

  <compatibility>
    <target name="gguf" supported="true"/>
    <target name="hf" supported="true"/>
    <target name="native" supported="experimental"/>
  </compatibility>
</xmlpack>
```

CLI lifecycle:

```bash
boltbeam xmlpack inspect daycare-model.xmlpack
boltbeam xmlpack validate daycare-model.xmlpack
boltbeam xmlpack chat daycare-model.xmlpack --native
boltbeam xmlpack export daycare-model.xmlpack --to gguf
boltbeam xmlpack audit daycare-model.xmlpack
```

Best use:

- one-file distribution;
- long-term archival;
- direct native runtime experiments;
- strong hash/signature story.

Tradeoff:

- harder to diff in Git;
- harder to partially edit by hand;
- requires custom blob writer/reader;
- not the right first MVP.

## Recommended Path

Start with Option 1, move to Option 2, defer Option 3.

```text
Option 1: repo layout
  -> easiest to inspect, review, and change

Option 2: package directory
  -> best canonical model artifact

Option 3: single-file archive
  -> useful only after schema and runtime are stable
```

For DayCare/BoltBeam:

```text
XML repo/package = source of truth
BoltBeam = validator, compiler, runtime adapter, auditor
GGUF/HF/ONNX = compatibility targets
native XML runtime = future path
```

## Pretraining MVP

Even for pretraining, the first XML work should be metadata and audit, not a
massive training run:

```text
1. define dataset.xml and data-card.xml
2. define model.xml and tokenizer.xml
3. define training-plan.xml
4. run tiny causal-LM smoke training
5. write checkpoint tensor-index.xml
6. export to existing runtime format
7. audit and record eval-run.xml
```

This gives a full end-to-end lifecycle without paying full pretraining cost.
