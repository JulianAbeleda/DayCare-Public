# DayCare

A small, dependency-light training stack for local language models, built on
[tinygrad](https://github.com/tinygrad/tinygrad). No torch, no transformers, no accelerate. Curriculum in,
adapter out, with a measurement discipline that refuses to call an unproven run a success.

**Current focus:** can a small local model use tools well? The active work trains **Nemotron 3 Nano 4B** (thinking
on) with RLOO on the turn *after* a tool call, inside [GameTerm](https://gameterm.arkey.ai): the
calculator returned a result, or rejected the call, or the policy refused a file read. What does the model do next?
The full record, including every failed run, is in the
**[post-tool RL index](research/post-tool-rl-index.md)**.

## The discipline

The code is small. The part worth copying is how a run earns the right to be believed
([RL run playbook](docs/rl-run-playbook.md)):

1. **Predeclare.** Hypothesis, numeric predictions per gate, and stop triggers are written and committed before
   the first counted update. Gates are paired stock-vs-adapter bootstrap intervals, clustered by source task.
2. **Prove the outputs first.** One update, export, reload in a fresh process: the adapter must reproduce
   bit-exactly before the long run starts.
3. **Stop on triggers, not on feelings.** KL, entropy, length, capped-turn rate and reward drift are checked on a
   rolling window; the loop stops itself and keeps the adapter.
4. **Replay before changing a recipe.** A new loss or mask is replayed on the previous run's saved records first,
   and each term's share of the gradient is measured.
5. **Score exactly as declared.** A failed gate ends the experiment; no rescue runs. A run that improves its target
   while retention regresses is a failure with a name.
6. **Stop spinning.** Two consecutive runs failing the same gate for the same reason means diagnose, not a third
   variant.

## Post-tool RL status

Why this turn: calling the calculator was already workable (tool selection was a separate adapter track), and
relaying a result that answers the question works. The weak spot is the turn after a result that does not settle
it. With thinking on, stock is right 33/64 after a miss and 16/64 after a rejection, against 63/64 for relay. So
training scores only the answer that follows the result ([why](docs/writeup.md#why-the-post-tool-turn)).

| Run | Outcome |
|---|---|
| 1 | not adopted: held-out success +7.0 points (pass), but blank answers 60 -> 95 (fail) |
| 2 | failed: training collapsed (~update 40); the audit found a lost length brake and a KL-weight bug |
| 3 | not adopted: +12.9 points (pass), blanks 53 against a limit of 50 (fail) |
| 4 | stopped by the entropy trigger at update 98; information-only gates strong, retention -2 |
| 5 | **adopted by owner exception**: every gate passed except one retention check (see below) |
| 6 | replication of run 5 on a new seed: stopped by the entropy trigger at update 66 |

**The adopted adapter (run 5)**, held-out against stock: repair after a rejected calculator call **+15.5 points**
[+11.2, +20.0]; blank answers in the relay family 97 -> 16 of 2,620; blocked calls handled correctly **+12.3**
[+7.7, +16.9]; an out-of-distribution blocked probe **+10.6** [+5.6, +15.4]. It is an exception, not a pass: the
numeric retention check (one greedy sample per item) failed by one item. A follow-up
[diagnosis](research/posttool-g3-numeric-diagnosis.md) found that loss fragile (level with stock at temperature 1
over all 68 numeric items), with known costs: a lower calculator call rate on numeric items (-5.5 points) and one
real reading shift. Run 6 did not reproduce it on a second seed because its entropy trigger fired first.

**Outside GameTerm**, on public tool-use benchmarks, the adapter shows no measurable change against stock, neither a
regression nor a gain: BFCL v4 non-live AST pooled +1.0 [-1.0, +3.0], multi_turn_base +4.5 [-1.5, +10.5]; When2Call
macro F1 -2.5 [-6.4, +1.2] ([public benchmarks](research/public-benchmarks-r5.md)). The short version of the whole
series, with a chart: **[write-up](docs/writeup.md)**.

## What is in this repository

- the base training pipeline (SFT + LoRA + before/after gate), the verdict vocabulary and campaign harness, the
  single-file artifact format, and the substrates (`daycare/`);
- the RL loop behind the post-tool work (`rloo_tinygrad`, `rloo_posttool`, the stop triggers, the GameTerm
  episode rules and graders) and its tests: [RL training](docs/rl-training.md);
- the research records, including the post-tool RL series (`research/`) and the run playbook (`docs/`);
- GameTerm's calculator as a headless runner (`tools/calculate-runner`, Rust, fend-core MIT).

**Not published:** the one-off gate and analysis scripts, the captured GameTerm envelope, the frozen task states,
model weights and run outputs.

## Quickstart

```bash
git clone https://github.com/JulianAbeleda/DayCare-Public && cd DayCare-Public
python -m venv .venv && . .venv/bin/activate
pip install -e ".[train,test]"   # train = the pinned tinygrad-arkey fork; drop it for harness/tests only
# or without cloning: pip install "daycare[train] @ git+https://github.com/JulianAbeleda/DayCare-Public"

# tests: no GPU, no model
python -m pytest tests

# training tinygrad (autograd + optim), pinned, no torch
./daycare/nursery/setup_trainer.sh

# end to end: SFT + LoRA + before/after gate, on the built-in curriculum
DAYCARE_MODEL_NAME=Ada DAYCARE_EPOCHS=6 python -m daycare.nursery.consolidate

# or: train against a teacher-written curriculum and keep the adapter
python -m daycare.nursery.distill curriculum.xml --save out.adapter.xml > run.xml

# did it learn the target without losing anything else
python -m daycare.nursery.evaluate path/to/Qwen3-0.6B-Q8_0.gguf Ada

# merge the adapter into a servable package
python -m daycare.artifact.merge base.xpkg out.adapter.xml out.xpkg

# rule a run and emit its ledger (simulated backend: runs anywhere, never promotes)
python -m daycare.harness --workload "repo idiom" --repeats 3 --out ledger.xml
```

Training and merging need a GPU and a local model; none of the commands above needs a network after setup.

## Reproduce run 5

The adopted adapter's recipe, on [tinygrad-arkey](https://github.com/JulianAbeleda/tinygrad-arkey) `exp` at
`bbf307f8347c166ec93b40d81c1b0bd3ce728563` and
[Nemotron 3 Nano 4B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) converted to GGUF:

```bash
export DAYCARE_TRAIN_TINYGRAD_PATH=/path/to/tinygrad-arkey DAYCARE_BASE_GGUF=/path/to/nemotron-3-nano-4b-bf16.gguf
(cd tools/calculate-runner && cargo build --release --locked)   # GameTerm's calculator, Rust 1.88
export DAYCARE_CALCULATE_RUNNER=$PWD/tools/calculate-runner/target/release/gameterm-calculate-runner
python -m daycare.nursery.rloo_posttool train --root out/r5 --states $S --envelope $E --updates 102 \
  --categories repair relay blocked --mix repair=0.75,relay=0.15,blocked=0.1 --mask none --reward graded \
  --wrong -1 --blank -1.5 --abstain relay=-0.5,repair=0,miss=0,empty=0,blocked=0 --length-weight 1.0 \
  --repetition n=40,penalty=-0.05 --kl-aggregation matched --kl-beta 0.03 --lanes 32 --prompts-per-update 4 \
  --compact 8,16 --keep-every 10 --train-seed 20260930 --protocol rloo-posttool-calculator-r5.md
```

Not everything is public yet. The GameTerm calculator runner is: [`tools/calculate-runner`](tools/calculate-runner/)
builds it, and its replies are byte-identical to run 5's runner. The envelope (`$E`) and the frozen task states
(`$S`) are not published either. That page covers the one-update bit-exact check to run first, how the states are
built, and the tests.

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

## Verdict vocabulary

The rules that decide between these are plain code (`harness/verdict.py`):

```text
PASS_PROMOTE           passed whole-system authority; may be promoted
LOCAL_PASS_SYSTEM_FAIL looked good in isolation, failed end to end
FAIL_LOCAL_AB          did not beat the comparator
FAIL_CORRECTNESS       failed the required gate
MEASUREMENT_UNSTABLE   noise or environment made the result unusable
REFUTED                closed by prior evidence
```

A run with no baseline cannot report an improvement, and a `simulated` run can never be promoted no matter how
good its numbers look (`app/mode.py`). The post-tool runs add one more rule: an owner may adopt an adapter that
failed a gate only in writing, recorded as an exception with its evidence and known costs, never as a pass.

## Substrates and artifacts

- **Train**: a pinned tinygrad (autograd + `nn/optim.py`) fetched by `nursery/setup_trainer.sh`, wired through
  `nursery/trainer_env.py`. The post-tool work samples and trains on one stack,
  [tinygrad-arkey](https://github.com/JulianAbeleda/tinygrad-arkey) (`exp`).
- **Serve**: `substrate/tinygrad_qwen3.py`, `substrate/llama_cpp.py` (a local `llama-server`), `substrate/remote.py`
  (a GPU host over ssh), or `substrate/stub.py` (deterministic, model-free, for tests). All speak
  `generate(prompt, temperature) -> str` (`substrate/base.py`).
- **Artifacts**: one `.adapter.xml` per adapter (int8 tensors, base64-inlined, with a manifest; `artifact/save.py`),
  `.xpkg` model packages (`examples/daycare-model.xpkg/`), and `.gguf` for llama.cpp (`artifact/emit_gguf.py`).

## Research notes

The reasoning behind the code, under `research/`:

- **[Write-up](docs/writeup.md)**: one app, one inference stack, one GPU; the post-tool RL result in ~1,000 words
- **[Post-tool RL index](research/post-tool-rl-index.md)**: runs 1-6, the side investigations and the literature
  report, in order
- [Public benchmarks, run 5 vs stock](research/public-benchmarks-r5.md): BFCL v4 and When2Call
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

## Research records

What was learned, with the numbers, the predictions written before each run, and the runs that were rejected:

- [LoRA training standard](research/lora-training-standard.md): the rules, each one paid for by a rejected run
- Nemotron 3 Nano 4B tool calling: [the run that aced the quiz and was rejected](research/nemotron-tool-calling/RESULTS.md), [the capability check that confirmed the rejection](research/nemotron-general-capability/RESULTS.md), [the adopted run](research/nemotron-tool-selection/RESULTS.md), [category metadata](research/nemotron-category-metadata/RESULTS.md), [tool retrieval](research/nemotron-tool-retrieval/RESULTS.md), [retrieval training, rejected](research/nemotron-retrieval-training/RESULTS.md)
- [XML model package](research/xml-pretraining-format.md), with a working scaffold in [`examples/daycare-model.xpkg`](examples/daycare-model.xpkg)
- [Bibliography](research/bibliography.md): the papers this stack was built from

The hybrid Mamba-2 / attention model these runs trained through is [`daycare/model/nemotron_h.py`](daycare/model/nemotron_h.py).

## Status

Research code, single-maintainer, tested at 0.6B on one GPU. The measurement
discipline is the mature part; the training paths are small and readable rather
than fast or general. No model weights, datasets, training outputs, or run state
belong in this repo (see [notes/repo-principles.md](notes/repo-principles.md)).

## License

[MIT](LICENSE). Research code, single maintainer. `tools/calculate-runner` links fend-core (MIT); its notice is in
[`tools/calculate-runner/THIRD_PARTY_NOTICES.md`](tools/calculate-runner/THIRD_PARTY_NOTICES.md).
