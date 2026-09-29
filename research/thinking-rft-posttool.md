# Thinking RFT with post-tool turns and a recurrent tail

Date: 2026-09-24. Predeclared before any data, training or evaluation.

## Why

The first thinking RFT adapter (`thinking-rft-countdown.md`)
gained +6.3 points on 150 fresh four-number Countdown tasks (interval +0.3 to
+12.7) but failed retention: three of its five real losses were an empty answer
after a calculator call. Its training rows were all tool-free, so no row showed
the turn after a tool result. This experiment adds that turn, and trains
through the recurrent tail now that it is possible
(`recurrent-tail-training.md`).

## Fixed settings

- Model: Nemotron 3 Nano 4B BF16 GGUF, thinking on, GameTerm wire-format tool
  results (`daycare/harness/gameterm_wire.py`).
- Training: `native_tool_sft`, LoRA rank 32, alpha 64, learning rate 5e-5,
  two epochs, `last_k=2` (Mamba block 40 and MLP block 41, segment gradients,
  `--segment 512`), `LRU=0`. Targets at most 4,500 characters.
- Two changes against the first adapter: post-tool rows and depth. The gates
  decide adoption; they do not attribute the result to one change.

## Checklist

### A. Protocol
- [x] A1. This file committed before any data is built (a private commit).

### B. Serving check (before training)
- [x] B1. An adapter with `blk.40.ssm_in` and `blk.40.ssm_out` exports to GGUF
  and llama-server applies it: logits with the adapter differ from without it
  on a fixed prompt, and a zero-B adapter matches the base. If llama.cpp
  ignores the Mamba tensors, stop and train `last_k=1` instead.
  **Passed** (`serving-check-001`): a zero-B adapter gave logprobs identical to
  the base; a Mamba-only adapter (MLP deltas zeroed, Mamba deltas scaled)
  changed them (top token -0.0062 to -0.0042). llama.cpp applies both.

### C. Data (all targets are the model's own thinking-on output, verified)
- [x] C1. Fresh efficacy holdout: 150 four-number tasks from TinyZero's public
  test region at offset 350000, excluding every earlier split and both earlier
  holdouts. Frozen before training: `countdown-holdout-4num-003`, 150 tasks,
  zero overlap.
- [x] C2. Countdown post-tool rows: for training-partition tasks, prefill a
  verified solver call and its real calculator result in wire format; sample
  two thinking-on continuations; keep completed, tool-free, correct answers; at
  most one per task.
- [x] C3. Arithmetic post-tool rows: GSM8K train rows 0 to 31 with their verified
  calculator expressions (`calculator-verified-sft-001`), real receipts, wire
  format; keep continuations whose final number equals the gold answer; at
  most one per question. None of these are retention-suite items.
  C2 and C3 (`rft-posttool-001/posttool-samples.xml`): every sample was
  accepted, 96/96 Countdown (48 tasks) and 64/64 arithmetic (32 questions), no
  empty answers. Stock thinking already handles this turn.
- [x] C4. Assemble: Countdown traces (the existing 50), post-tool rows (C2, C3),
  plain rows (31), at least 25% plain; the loader accepts it; counts recorded.
  Deviation, decided by counts alone: all post-tool rows would put plain at
  19%. Arithmetic post-tool rows go first (the retention failure), then
  Countdown ones in task order until plain is exactly 25%.
- [x] C5. Overlap check: no training request appears in the retention suite or
  in any holdout. **Caught one:** the plain row "What is the chemical symbol
  for gold?" is retention item `main::ordinary-3` (it was also in the first
  RFT adapter's data). Assembly now excludes every request in the retention
  suite (`--exclude`). Final set: 120 rows, 50 Countdown, 40 post-tool (32
  arithmetic, 8 Countdown), 30 plain (25%); zero overlap.

### D. Train
- [ ] D1. Clean tree; train; no NVIDIA Xid; no non-finite loss; run record
  shows `gradient: segment` and the four target tensors.
  Attempt 1 (`sft`) crashed in feature caching at row 77: tinygrad's NV
  backend wrote a 40-bit value into the 32-bit QMD prefetch-address field
  (patched in `setup_trainer.sh`, a private commit). Attempt 2 (`sft2`) completed but
  is **invalid**: every row's epoch-2 loss equalled its epoch-1 loss exactly.
  LoRA's B started as a lazy zero constant that TinyJit baked into the
  segment kernels, so the forward never saw an update. A second tinygrad
  pitfall surfaced while fixing it: two same-shaped lazy A matrices received
  one merged gradient. `apply_lora` now creates realized buffers; a GPU test
  trains from the default initialization and requires the loss to fall.
  The equivalence tests had passed only because they set B to realized random
  values. Earlier last-block adapters were unaffected (no JIT, A shapes differ).
- [ ] D2. Export `adapter.gguf` losslessly.

### E. Gates
- [ ] E1. Efficacy: C1 holdout, two seeds, stock vs adapter, first turn,
  thinking on, 4,096 tokens, temperature 1.0. Pass only if the task-clustered
  bootstrap 95% interval of the paired difference has a lower bound above zero.
- [ ] E2. Retention (only if E1 passes): the 132-item suite, adapter vs stock,
  both thinking on, temperature 0. Pass only with zero plain-answer losses and
  no net real loss after hand audit of every disagreement.
- [ ] E3. The specific failure: count answers that are empty after a calculator
  call; the first adapter had three.
- [ ] E4. Stop rule: a failed gate ends the experiment; no rescue runs.

### F. Record
- [ ] F1. Results here, committed.

## Result: stopped at D1 by decision

Attempt 3 (`sft3`, with the LoRA initialization fix) was stopped on
2026-09-24 during feature caching (5 of 120 rows), before any update, when
the plan switched to RLOO with llama.cpp sampling. No gate was run; this is
neither a pass nor a failure. The dataset (`rft-posttool-001/dataset.xml`),
the fresh holdout (`countdown-holdout-4num-003`) and the checked code remain
usable if this treatment is resumed.
