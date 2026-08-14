# daycare (package)

The implementation. Every stage is a module you can run or import on its own.

```text
daycare/
  nursery/     the training pipeline
    route.py         file / retrieval / weights -- decide before spending a GPU
    curriculum.py    build a training set and a held-out eval set
    generate.py      draw candidate examples from a teacher model
    judge.py         the teacher grades; the student never marks its own work
    score.py         raw answers -> numbers
    distill.py       supervised training on teacher completions (fast path)
    consolidate.py   end-to-end SFT + LoRA + before/after gate
    lora.py          the adapter
    evaluate.py      target gate AND forgetting probe, against any GGUF
    fixtures.py      deterministic inputs for tests
    trainer_env.py   import the training tinygrad, not the serving one
  train/
    optimizer.py     lr/decay update rule over a small trainable adapter
  harness/       measurement, not training
    verdict.py       the six-outcome vocabulary; only PASS_PROMOTE ships
    campaign.py      many attempts across one knob, never pooled
    artifact.py      the claim-bearing record a run leaves behind
  substrate/     inference only -- generate(prompt, temperature) -> str
    base.py          the protocol
    stub.py          deterministic, model-free (tests run with no model)
    tinygrad_qwen3.py  Qwen3 through the inference-only tinygrad fork
    llama_cpp.py     spawns a local llama-server
    remote.py        a GPU host over ssh
    tinygrad_train.py  the trainable forward
  artifact/      single-file model artifacts
    save.py          write one .adapter.xml (int8 + base64 + manifest)
    merge.py         adapter + base -> merged package
    emit_gguf.py     GGUF for llama.cpp / NVIDIA serving
    package.py       .xpkg layout, tokenizer, index
  model/
    qwen_forward.py  cache-free differentiable forward used for training
  app/
    mode.py          DAYCARE_MODE: the one switch between demo and real
    run.py           start / poll / stream / cancel a training run
  image/           image-model training project (separate track)
```

## Two tinygrads (train vs serve)

Upstream tinygrad does too much to depend on wholesale, so it is split by role:

- **Serve** — an inference-only fork: model load and generation, no autograd.
  Used by `substrate/tinygrad_qwen3.py`.
- **Train** — a pinned upstream copy, git-ignored, providing autograd,
  `nn/optim.py`, and `nn/state.py`. Fetch it with:

  ```bash
  ./daycare/nursery/setup_trainer.sh    # pip install --no-deps tinygrad==0.13.0
  ```

  Import it through `nursery/trainer_env.py::use_train_tinygrad()`; override the
  path with `DAYCARE_TRAIN_TINYGRAD_PATH`. No torch, no transformers.

## Run

```bash
# end-to-end training run on the built-in curriculum
DAYCARE_MODEL_NAME=Ada DAYCARE_EPOCHS=6 python -m daycare.nursery.consolidate

# train on a teacher-written curriculum, keep the adapter
python -m daycare.nursery.distill curriculum.xml --save out.adapter.xml > run.xml

# gate + forgetting probe against any GGUF
python -m daycare.nursery.evaluate ~/models/Qwen3-0.6B-Q8_0.gguf Ada

# rule on a run
python -m daycare.harness --help

# tests (no model required)
python -m pytest tests/
```

## Environment

| variable | default | meaning |
|---|---|---|
| `DAYCARE_MODE` | `debug` | `live` runs real backends; a simulated run can never be promoted |
| `DAYCARE_MODEL_NAME` | `Ada` | the target the built-in curriculum teaches |
| `DAYCARE_EPOCHS` / `DAYCARE_LR` / `DAYCARE_LORA_R` | `6` / `1e-3` / `16` | training knobs |
| `DAYCARE_TRAIN_HOST` | `linuxbox` | ssh host for the remote backend |
| `DAYCARE_MODELS_DIR` | `~/models` | where base and output GGUFs live |
| `DAYCARE_LLAMA_BIN` | `~/env/llama.cpp/build-cuda/bin/llama-server` | llama.cpp server binary |
| `DAYCARE_LLAMA_MODEL` | `~/models/Qwen3-0.6B-Q8_0.gguf` | model the llama substrate serves |
| `DAYCARE_LLAMA_CTX` / `DAYCARE_LLAMA_GPU_LAYERS` / `DAYCARE_LLAMA_PORT` | `2048` / `-1` / `0` | llama.cpp serving knobs |

## Status

Tested at 0.6B on a single GPU. The harness and artifact formats are the settled
parts; the training paths favour being readable over being fast or general.
