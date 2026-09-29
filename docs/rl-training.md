# RL training: RLOO on the post-tool turn

The code behind the [post-tool RL series](../research/post-tool-rl-index.md): one-stack RLOO on
Nemotron 3 Nano 4B, sampling and training in the same tinygrad process, scored on the turn after a [GameTerm](https://gameterm.arkey.ai) tool
call. The adopted recipe is [run 5](../research/rloo-posttool-calculator-r5.md). This page lists the modules, the
external pieces they need, and the commands. It also says which inputs are **not** published.

## Modules

| Module | Role |
|---|---|
| `daycare/nursery/rloo_tinygrad.py` | the loop: sample with the tinygrad-arkey rollout sampler, recompute the tail, RLOO + KL + entropy, one Adam step in place; export and bit-exact `verify` |
| `daycare/nursery/rloo_posttool.py` | post-tool episodes (multi-turn, GameTerm's wire format), rewards, `sample` / `train` / `verify` / `force` / `resume` |
| `daycare/nursery/rlvr.py`, `rlvr_update.py` | the RLOO objectives (leave-one-out baseline, token policy loss, KL/entropy aggregation) and the reference tail |
| `daycare/nursery/rl_triggers.py` | the rolling stop triggers (`DEFAULTS` = run 5's, with the composition-adjusted entropy test) and their replay |
| `daycare/nursery/posttool_tasks.py` | task generation, the fixed train/held-out split, state freezing, the training mix |
| `daycare/nursery/selection_probe.py` | the out-of-distribution `blocked` probe (gate G6) |
| `daycare/nursery/lora.py`, `provenance.py` | LoRA adapters with exact target maps; file digests and the source revision a run records |
| `daycare/harness/posttool.py`, `posttool_blocked.py` | GameTerm's later-turn request rules, the calculator client, the graded reward, and the blocked grader (rule (c)) |
| `daycare/harness/countdown.py`, `countdown_reward.py`, `gameterm_wire.py` | Countdown arithmetic, the Countdown verifier, GameTerm's tool-result bytes |
| `daycare/model/chat_format.py`, `gguf_tokenizer.py` | the chat template renderer (Jinja, sandboxed) and the GGUF tokenizer |
| `daycare/artifact/record_xml.py`, `native_lora.py`, `lora_gguf.py` | XML run records, lossless XML adapter checkpoints, LoRA GGUF export for llama.cpp |

## External dependencies

### tinygrad-arkey (sampler and trainer)

Run 5 ran on [tinygrad-arkey](https://github.com/JulianAbeleda/tinygrad-arkey), branch `exp`, commit
**`bbf307f8347c166ec93b40d81c1b0bd3ce728563`**. That tree has the Nemotron-H model, the chunked prefill and the
rollout sampler with step compaction (`--compact`). The pinned upstream tinygrad from `setup_trainer.sh` does not
have them.

```bash
git clone https://github.com/JulianAbeleda/tinygrad-arkey && git -C tinygrad-arkey checkout bbf307f8347c166ec93b40d81c1b0bd3ce728563
export DAYCARE_TRAIN_TINYGRAD_PATH=$PWD/tinygrad-arkey
```

No pip install is needed: `trainer_env` puts the tree first on `sys.path`. Python needs only `numpy` and `jinja2`.
Use a checkout, not `pip install` of tinygrad-arkey: the installed package leaves out the repository's `extra/`
tree, which its own model and attention modules import, so the loop fails with `No module named 'extra'`.
Run 5 trained on one 32 GB NVIDIA GPU (tinygrad picks the device; `DEV` overrides it).

### Model

Nemotron 3 Nano 4B, BF16: [`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
(NVIDIA Open Model License). Convert it to a BF16 GGUF with llama.cpp:

```bash
python convert_hf_to_gguf.py NVIDIA-Nemotron-3-Nano-4B-BF16 --outtype bf16 --outfile nemotron-3-nano-4b-bf16.gguf
export DAYCARE_BASE_GGUF=$PWD/nemotron-3-nano-4b-bf16.gguf   # or pass --model
```

Run 5's GGUF had sha256 `2126a5e8056f4178200f3c30dcac11ac0ebae200fdb8d2b8ff86e2fa2930fcb8`, and its
`model.safetensors` had sha256 `55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`. Each run records
the model digest, so a different conversion shows up in the run record. No weights are published here.

### GameTerm calculator runner

Episodes run `calculate` calls through GameTerm's own calculator (argument normalisation, then evaluation with
[fend](https://github.com/printfn/fend)). Its headless runner is in this repo: [`tools/calculate-runner`](../tools/calculate-runner/)
(Rust 1.88, fend-core pinned at 1.5.8, `Cargo.lock` committed):

```bash
cd tools/calculate-runner && cargo build --release --locked
export DAYCARE_CALCULATE_RUNNER=$PWD/target/release/gameterm-calculate-runner   # or pass --runner
```

Its replies are byte-identical to the runner run 5 used: 2,204 unique argument strings from run 5's records and 71
edge cases (malformed and wrong-field arguments, bounds, rounding, percentages, division by zero, huge numbers,
timeouts), and all 1,709 recorded `calculate` results re-render byte for byte. The interface:

- an executable that takes no arguments and stays alive for the whole run;
- **stdin:** one line per call, holding the `calculate` tool's arguments as compact JSON, e.g.
  `{"expression":"12 / 5"}` or `{"expression":"2/3","decimals":2}`;
- **stdout:** one JSON line per call:
  - `{"ok":true,"outcome":"12 / 5 = 2.4"}`: an answer, shown to the model as stdout;
  - `{"error":"arguments: MalformedArguments","ok":false}` (also `MissingField`, `OutOfRange`): a schema error.
    DayCare turns it into a `rejected` result with GameTerm's fixed one-line text (`posttool.REJECTIONS`);
  - `{"error":"could not calculate ...","ok":false}`: a calculator refusal (including the 1 s evaluation timeout),
    shown to the model as stdout.

The argument rules, rejection texts and timeout are in the runner's [README](../tools/calculate-runner/README.md).

### Inputs that are not published

- **The GameTerm request envelope** (`envelope-002-default.xml`): the llama-server chat-completions body that
  GameTerm sends, captured and stored as an XML record (`record_xml`). Its keys are `messages` (the system prompt,
  the user context, and a `{"role": "user", "content": "{{REQUEST}}"}` placeholder), `tools` (the `calculate`,
  terminal and file tools, among others), `chat_template_kwargs: {"enable_thinking": true}`, `logit_bias`,
  `temperature`, `top_p` and `max_tokens`. It was captured from the non-public GameTerm build, so it is not
  included.
- **The frozen task states** (`posttool-tasks-003/states.xml`, 427 trained states). `posttool_tasks` builds them in
  three steps: `freeze` (the task list and a fixed split), `rloo_posttool sample --tasks` (the stock model's own
  first call on each task), then `states`. `freeze` checks the new tasks for overlap with earlier evaluation suites
  that are not published. It reads them from `DAYCARE_RUNS`, so it cannot rebuild tasks-003 exactly outside the
  original setup. You can build your own task set with the same generators:
  `python -m daycare.nursery.posttool_tasks freeze --root R --no-private-suites` skips those exclusions and the
  historical `empty` states (their probe history is unpublished too), and fetches the Countdown tasks over the
  network (`--countdown-per-size 0` to skip them). `freeze` builds in a temporary folder and renames it to `R`
  only on success.
- The records of runs 1-6 (used by the trigger replays in `tests/test_rl_triggers.py`, which skip without them).

## Reproduce run 5

With the envelope `E` and states `S` above, the recipe from the
[run-5 record](../research/rloo-posttool-calculator-r5.md) (Process 1) is:

```bash
RECIPE="--states $S --envelope $E --categories repair relay blocked
 --mix repair=0.75,relay=0.15,blocked=0.1 --mask none --reward graded --wrong -1 --blank -1.5
 --abstain relay=-0.5,repair=0,miss=0,empty=0,blocked=0 --length-weight 1.0 --repetition n=40,penalty=-0.05
 --kl-aggregation matched --kl-beta 0.03 --lanes 32 --prompts-per-update 4 --compact 8,16 --keep-every 10
 --train-seed 20260930 --protocol rloo-posttool-calculator-r5.md"

# P2: one update, export, then verify the reload bit-exactly in a fresh process
python -m daycare.nursery.rloo_posttool train --root out/r5-check $RECIPE --updates 1
python -m daycare.nursery.rloo_posttool verify --run out/r5-check --envelope $E

# P3: 102 updates from stock; the stop triggers are rl_triggers.DEFAULTS (not passed)
python -m daycare.nursery.rloo_posttool train --root out/r5 $RECIPE --updates 102
python -m daycare.nursery.rloo_posttool verify --run out/r5 --envelope $E
```

`train` refuses to start when the checkout has uncommitted tracked changes, because the run records the exact
revision. The adapter is written as `adapter.xml` (lossless) and `adapter.gguf` (a llama.cpp LoRA). The held-out
gates are `rloo_posttool sample` arms: stock, and adapter (with `--run`). Their states and flags are listed in the
run-5 record, section 5.

## Tests

```bash
python -m pytest tests                                               # CPU only; tinygrad tests skip
DAYCARE_TRAIN_TINYGRAD_PATH=/path/to/tinygrad-arkey python -m pytest tests   # + the tiny Nemotron-H loop on CPU
```

Tests that need the calculator runner (`DAYCARE_CALCULATE_RUNNER`) or the unpublished run records (`DAYCARE_RUNS`)
skip when those are not set. There is no CI; with the runner built, run them locally:

```bash
(cd tools/calculate-runner && cargo build --release --locked && cargo test --release --locked)
DAYCARE_CALCULATE_RUNNER=$PWD/tools/calculate-runner/target/release/gameterm-calculate-runner python -m pytest tests
```

That un-skips `tests/test_posttool.py`'s calculator test; the two episode tests in `tests/test_rloo_posttool.py`
also need `DAYCARE_TRAIN_TINYGRAD_PATH` (tinygrad-arkey `exp`).
