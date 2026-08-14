# Model Archaeology

Model archaeology is the cheap alternative to pretraining first. Instead of
training a base model from scratch, inspect existing model artifacts, probe
their behavior, infer their strengths and gaps, and train small adapters only
where needed.

## Goal

```text
avoid expensive pretraining
  -> reverse-engineer useful existing models
  -> choose the best base
  -> train small DayCare/BoltBeam adapters
```

This is not exact provenance recovery. It is practical reverse engineering.

## Inputs

- GGUF models;
- Hugging Face model repos;
- safetensors checkpoints;
- LoRA/QLoRA adapters;
- tokenizer/config files;
- model cards and licenses;
- known base/tuned pairs when available.

## Artifact Inspection

First inspect what the model physically contains:

```text
format
architecture family
parameter count
tensor names
tensor shapes
tensor dtypes
quantization layout
tokenizer type
context length
RoPE/runtime metadata
chat template
license/provenance metadata
```

For the XML model-format idea, this becomes:

```bash
boltbeam model inspect model.gguf --to-xml model.inspect.xml
```

The XML should distinguish:

```text
known from artifact
inferred from tests
unknown / missing provenance
```

## Behavioral Probe Suite

Then test behavior directly:

- instruction following;
- summarization accuracy;
- evidence discipline;
- refusal to overclaim;
- tool-use planning;
- code reasoning;
- long-context stability;
- calibration: confidence vs correctness;
- safety/refusal profile;
- DayCare/BoltBeam-specific next-action choice.

Each probe should write a structured result:

```xml
<probe name="evidence_discipline" score="0.72">
  <pass count="36"/>
  <fail count="14"/>
  <notes>Over-promotes when speed evidence is absent.</notes>
</probe>
```

## Delta Analysis

If both base and tuned models are available:

```bash
boltbeam model diff base.gguf tuned.gguf --to-xml delta.xml
```

Useful outputs:

- changed tensor list;
- per-layer delta norms;
- attention vs MLP concentration;
- embedding/head changes;
- quantization-aware approximate deltas;
- suspected merged LoRA signatures;
- adapter target modules when adapter files are available.

This can show where training pressure landed, but it cannot prove the exact
dataset or optimizer.

## Inference Levels

Use confidence labels:

```text
known:
  directly present in artifact metadata

strong inference:
  repeated behavior across probes
  or exact base/tuned delta evidence

weak inference:
  behavior pattern resembles SFT/DPO/RLHF/etc.

unknown:
  cannot be recovered from artifact/tests
```

Example:

```xml
<inference confidence="strong">
  Model follows chat instructions well but lacks BoltBeam evidence discipline.
</inference>

<inference confidence="weak">
  Tuning resembles instruction SFT more than verifier-backed RL.
</inference>
```

## Selection Loop

The MVP loop:

```text
1. choose 5-10 candidate open models
2. inspect artifacts into XML
3. run the DayCare/BoltBeam probe suite
4. rank by gaps and license/runtime fit
5. pick the best base
6. train LoRA/QLoRA adapter only
7. audit against held-out tasks
8. export GGUF/HF as compatibility targets
```

## What This Avoids

This avoids starting with:

```text
crawl corpus
dedupe/filter massive data
pretrain base model
post-train
debug from scratch
```

That path is too expensive for the first DayCare pure LLM route.

## What It Cannot Do

Model archaeology cannot exactly recover:

- original training data;
- training order;
- optimizer settings;
- RLHF vs DPO vs ORPO with certainty;
- individual examples responsible for behavior;
- pretraining token count;
- hidden filtering policies.

It can still identify the model that is easiest to adapt.

## Promotion Criteria

Promote a base model for DayCare/BoltBeam adaptation only if:

- license allows intended use;
- runtime works locally;
- probe suite beats other candidates;
- weaknesses are adapter-addressable;
- held-out evidence-discipline failures are understood;
- the model does not need continued pretraining to understand the domain.

## Principle

```text
Do not pretrain to learn what the ecosystem already knows.
Use archaeology, probes, deltas, and small adapters first.
```
