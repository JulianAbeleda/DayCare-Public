# DayCare

A small, dependency-light training stack for local language models. No torch, no
transformers, no accelerate. Curriculum in, adapter out, with a measurement
harness that refuses to call an unproven run a success.

The point is to make the training loop **simple enough to read in an afternoon**:
every stage is one file you can run on its own, and every artifact is a single
self-contained XML file you can open in an editor.

```text
curriculum  ->  train  ->  evaluate  ->  rule  ->  promote
   XML          LoRA       gate +      verdict    merge -> GGUF
                           forgetting   vocabulary
                           probe
```

## Why it exists

Most training stacks assume a cluster and a research team. This one assumes one
GPU, one person, and a small model, and makes three things explicit that larger
stacks leave implicit:

1. **Not everything should be trained.** `nursery/route.py` classifies a lesson
   as *file*, *retrieval*, or *weights* before a GPU is spent. Teaching a model a
   fact that a text file already holds is the most common way to burn a run and
   damage the model.
2. **A run is not a success because the loss went down.** `harness/verdict.py`
   has six outcomes and only one of them promotes. A run that improves its target
   while a forgetting probe regresses is a failure with a name.
3. **Artifacts are readable.** An adapter is one `.adapter.xml` — int8 tensors,
   base64-inlined, with a manifest — not a directory of opaque blobs. It merges
   into a GGUF you can serve with llama.cpp.

## Quickstart

```bash
git clone https://github.com/JulianAbeleda/DayCare-Public && cd DayCare-Public

# training tinygrad (autograd + optim), pinned, no torch
./daycare/nursery/setup_trainer.sh

# end to end: SFT + LoRA + before/after gate, on the built-in curriculum
DAYCARE_MODEL_NAME=Ada DAYCARE_EPOCHS=6 python -m daycare.nursery.consolidate

# or: train against a teacher-written curriculum and keep the adapter
python -m daycare.nursery.distill curriculum.xml --save out.adapter.xml > run.xml

# did it learn the target without losing anything else
python -m daycare.nursery.evaluate ~/models/Qwen3-0.6B-Q8_0.gguf Ada

# merge the adapter into a servable package
python -m daycare.artifact.merge base.xpkg out.adapter.xml out.xpkg
```

Nothing above needs a network. Training and merging need a GPU; the harness and
the stub substrate run anywhere.

## The pipeline

| stage | module | what it does |
|---|---|---|
| route | `nursery/route.py` | decide file vs retrieval vs weights before training anything |
| curriculum | `nursery/curriculum.py` | build the training set plus a held-out eval set (library) |
| generate | `nursery/generate.py` | produce candidate examples from a teacher model |
| judge | `nursery/judge.py` | score candidates; the teacher grades, never the student |
| distill | `nursery/distill.py` | supervised training against teacher completions (the fast path) |
| consolidate | `nursery/consolidate.py` | end-to-end SFT + LoRA + before/after gate |
| lora | `nursery/lora.py` | the adapter itself |
| evaluate | `nursery/evaluate.py` | target gate **and** forgetting probe, against any GGUF |
| score | `nursery/score.py` | turn raw answers into numbers |
| rule | `harness/verdict.py` | one of six verdicts; only `PASS_PROMOTE` ships |
| campaign | `harness/campaign.py` | many attempts across a knob, never pooled into one average |
| merge | `artifact/merge.py` | adapter + base -> servable GGUF |

## Measurement

The harness is not a wrapper around a loss curve. It implements a fixed verdict
vocabulary, and the rules that decide between them are plain code:

```text
PASS_PROMOTE           passed whole-system authority; may be promoted
LOCAL_PASS_SYSTEM_FAIL looked good in isolation, failed end to end
FAIL_LOCAL_AB          did not beat the comparator
FAIL_CORRECTNESS       failed the required gate
MEASUREMENT_UNSTABLE   noise or environment made the result unusable
REFUTED                closed by prior evidence
```

Two rules it will not let you break: a run with no baseline cannot report an
improvement, and a `simulated` run can never be promoted no matter how good its
numbers look (`app/mode.py`).

## Substrates

Training and serving use different runtimes on purpose:

- **Train** — a pinned upstream tinygrad (autograd + `nn/optim.py`), fetched by
  `nursery/setup_trainer.sh`, wired through `nursery/trainer_env.py`.
- **Serve** — `substrate/tinygrad_qwen3.py` (inference-only fork),
  `substrate/llama_cpp.py` (spawns a local `llama-server`),
  `substrate/remote.py` (a GPU host over ssh), or `substrate/stub.py`
  (deterministic, model-free, for tests).

Everything speaks one protocol: `generate(prompt, temperature) -> str`
(`substrate/base.py`), so a test suite runs against the stub with no model
present.

## Artifacts

One file per artifact, no safetensors, no JSON sprawl:

- `.adapter.xml` — int8-quantized tensors, base64-inlined, plus a manifest
  (`artifact/save.py`)
- `.xpkg` — a model package directory with tokenizer and weights
  (`examples/daycare-model.xpkg/`)
- `.gguf` — emitted for llama.cpp and NVIDIA serving (`artifact/emit_gguf.py`)

## Research notes

The reasoning behind the code, under `research/`:

- [Training Map](research/training-map.md) — the whole territory
- [Fact vs Weight](research/fact-vs-weight.md) — what should never be trained
- [State vs Weights](research/state-vs-weights.md) — where knowledge belongs
- [Catastrophic Forgetting](research/catastrophic-forgetting.md) — the failure this stack is built around
- [SFT, LoRA, and QLoRA](research/sft-lora-qlora.md)
- [Preference and RL Training](research/preference-and-rl-training.md) — DPO, PPO, GRPO, RLVR
- [Pretraining and Continued Pretraining](research/pretraining.md)
- [Data Curation and Evaluation](research/data-and-evaluation.md)
- [XML Pretraining Format](research/xml-pretraining-format.md) · [XML Model Artifacts](research/xml-artifact-format-scope.md)
- [JEPA and World Models](research/jepa-and-world-models.md)
- [Model Archaeology](research/model-archaeology.md) · [Pretraining History](research/pretraining-history.md)
- [Open-Source Tooling](research/open-source-tooling.md) · [Bibliography](research/bibliography.md)

## Status

Research code, single-maintainer, tested at 0.6B on one GPU. The measurement
discipline is the mature part; the training paths are small and readable rather
than fast or general. No model weights, datasets, training outputs, or run state
belong in this repo (see [notes/repo-principles.md](notes/repo-principles.md)).

## License

[MIT](LICENSE).

## Research records

What was learned, with the numbers, the predictions written before each run, and the runs that were rejected:

- [LoRA training standard](research/lora-training-standard.md): the rules, each one paid for by a rejected run
- Nemotron 3 Nano 4B tool calling: [the run that aced the quiz and was rejected](research/nemotron-tool-calling/RESULTS.md), [the capability check that confirmed the rejection](research/nemotron-general-capability/RESULTS.md), [the adopted run](research/nemotron-tool-selection/RESULTS.md), [category metadata](research/nemotron-category-metadata/RESULTS.md), [tool retrieval](research/nemotron-tool-retrieval/RESULTS.md), [retrieval training, rejected](research/nemotron-retrieval-training/RESULTS.md)
- [XML model package](research/xml-pretraining-format.md), with a working scaffold in [`examples/daycare-model.xpkg`](examples/daycare-model.xpkg)
- [Bibliography](research/bibliography.md): the papers this stack was built from

The hybrid Mamba-2 / attention model these runs trained through is [`daycare/model/nemotron_h.py`](daycare/model/nemotron_h.py).
