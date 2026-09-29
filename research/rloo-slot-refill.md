# RLOO sampling: stop paying for finished lanes (slot refill / step compaction)

Date: 2026-09-27. Predeclared: the hypothesis, the bound and the A/B process are committed before any GPU
measurement. Goal (the maintainer): similar results, faster, with the training math unchanged.

## Hypothesis

**H.** An update spends most of its sampling time on lanes whose episode has already finished, because the
decode step always runs all 32 lanes. Removing that waste *without changing which episodes are sampled or how
they train* cuts the 32-lane post-tool update to ~0.64x its time (1.56x faster; range 0.58-0.70x). The
parity check stays bit-exact (max logprob error 0), and peak VRAM stays at the 32-lane run's ~22.1 GB (+0.3 GB at
most). Falsified if s/update (updates 1-7, same seed) is above 0.80x baseline, parity is not exact, or peak VRAM
exceeds 22.5 GB.

## Why the waste exists, and what is measured already

- The sampler (`NemotronHRolloutSampler`, tinygrad-arkey `exp`) already refills free lanes *within* a sampling
  call: follow-up turns join the running batch (4322ae9e2). But one update is exactly 4 prompts x 8 = 32 episodes
  on 32 lanes, so once an episode ends there is nothing left in that update to put in its lane. The lane idles
  until the slowest episode (thinking turns run up to 4,096 tokens) finishes.
- **Idle lanes cost full price.** The step graph always runs `batch` rows. r3-speed runs: 22.35-22.58 ms/step at
  32 lanes, 39.30-39.34 ms/step at 64 lanes, with the *same* 32 episodes (the 64-lane run's extra 32 lanes were
  idle the whole time). So step time is ~5.4 ms + 0.53 ms per row, and an idle row costs as much as a busy one.
- **How much is idle.** From the logged per-update episode turn lengths (r1 101 updates, r2 53, r3-speed 2;
  update 0 dropped because it includes JIT capture), a window simulator (32-step windows, one lane per episode,
  follow-ups on the next window boundary) reproduces the logged `decode_steps` on **156/156** updates. The per-episode token totals it uses reproduce the logged
  `active_lane_steps` on 154/154 (r1+r2), and window rounding adds 2% occupancy on top of that. Lane
  utilization is **0.31** (1 - mean/max episode tokens = 0.69). Decode is **88%** of update wall time (82 s/update
  mean). Script: `<scratch>/refill/bound.py`.

## Options

| option | what it does | same training math? | predicted update time |
|---|---|---|---|
| A. oversubscribe groups (sample > 8 per prompt, keep the first 8 to finish) | fills lanes with same-update work | **no**: keeps short episodes and drops long ones, which biases every group toward short episodes (and the Kimi length term sees a truncated distribution) | rejected |
| B. pipeline across updates (start update n+1's groups in freed lanes) | real refill | **no**: n+1's episodes are sampled with weights n and continue after Adam writes the adapter in place, so they come from a mix of two policies. The 0.1-nat parity hard stop has to become off-policy TIS (cap 2) that actually corrects; PipelineRL-style one-step lag | x0.39 ideal (x0.31 decode) |
| **C. compaction (recommended)** | between windows, move running lanes down to lanes 0..n-1 (and their prompts to slots 0..m-1) and run a step graph over the smallest of {8, 16} rows that covers them (all 32 otherwise) | **yes**: same episodes (same prompts, same groups, same policy, same sampling distribution); only the lane and graph that serve a rollout change | **x0.64** (8/16), x0.61 (4/8/16), x0.60 (pow2 to 1) |

Reasoning for C's number: per window, cost = 32 x T(k) with k the smallest listed row count covering the running
lanes, T(k) = 5.4 + 0.53k ms. The prediction scales only decode by the simulated ratio (x0.59 for 8/16) and leaves
prefill, scoring and learn unchanged. Caveats, both named before measuring: T(k) below 32 rows is extrapolated
from two points, and the compacted step still runs the sampling tail (final MLP block + output head) on all 32
rows (below). Both push the real number up, which is why the H band stops at 0.80.

Recommendation: **C**. It is the only option that keeps each update's training data distribution and update
math identical. B is ~1.6x faster again on paper, but it changes the math (off-policy by up to one step, mixed
policies within an episode, a relaxed parity stop), so it would be a new experiment, not a speed-up.

## Design of C (tinygrad-arkey `exp`, `NemotronHRolloutSampler(compact=(8, 16))`)

- **Lane packing.** `pack()` runs after each window's retire. It moves the highest running lane into the lowest
  free lane (`_move_lane`, a TinyJit copy of the lane's Mamba conv/state and replay ring, its generated-key ring
  rows, its length and its next input token), and does the same for prompt slots (`_move_slot`, the slot's
  prefix keys/values). The ring rows are global step rows (`step % capacity`), so a moved lane's keys keep their
  meaning. Refill already takes the lowest free lane and slot, so new rollouts land contiguously.
- **Sized step graphs.** For row count s < 32 the step and flush graphs are captured inside `_rows(s)`. That
  context swaps every per-lane buffer for an **alias of its first s rows** (`_leading`: a `Buffer.view` of the
  same memory, not a shrink view). A shrink view silently broke read-then-write ordering: the Mamba step reads its
  conv tail and then assigns it, and through a view the read could see the new value. Unit tests caught this
  (logprobs wrong from token 0); aliases fixed it. Sized graphs plan into the same shared arenas, and no buffer is
  allocated, so there is no new VRAM beyond the move graphs' one-lane temporaries.
- **Parity is kept by construction.** In a compacted step, the frozen blocks (0..capture-1) run on s rows. The
  capture (the input to the LoRA block) is written for those rows, padded with zero rows to 32 in its own buffer,
  and the tail (final MLP block, norm, head, sampling) runs on all 32 rows. That is the exact kernel set
  `Tail`/`tail_logprobs` recomputes in the update, which already packs tokens into arbitrary rows at 0 error. The
  frozen blocks at s rows may round differently than at 32. That changes nothing trained: the capture is by
  definition what the tail read. It is the same kind of batch-shape rounding that already differs between any two
  schedules.
- **Sampling noise.** Gumbel noise is drawn per step over 32 rows, as before. A lane's position differs, so the
  same seed gives different (equally distributed) samples than the uncompacted run. The A/B therefore compares
  distributions, not tokens.
- **Scope.** With a capture, compaction requires the tail from `capture` to be stateless MLP blocks (LoRA on the
  final block). It also requires `max(compact) <= prompts` slots. The post-tool loop passes `prompts >= lanes`; the
  Countdown `Loop` defaults to 10 slots, so `compact=(8, 16)` is refused there. Otherwise the constructor refuses. `compact=()` (the default)
  leaves every existing graph, key and behaviour unchanged.
- **DayCare.** `rloo_posttool --compact 8,16` (default off; `cfg["compact"]` goes to the `Loop`). The episode
  stats add `row_steps`, `row_utilization`, `lane_moves`, `window_rows` (windows per row count) and
  `window_rows_s` (wall seconds per row count, which gives T(k) directly).

CPU tests (tiny Nemotron-H, `DEV=CPU`):
- `test/unit/test_nemotron_h_sampler_compact.py` (4 tests), all pass:
  - compacted rollouts match the full recompute
  - captures match the reference inputs
  - `tail_logprobs` on 32-row packing reproduces every sampled logprob bit for bit, with lane moves > 0 and some windows < full rows
  - follow-ups + shared prefix + prefill
  - no-capture mode
  - argument checks
- The existing sampler/model suites were rerun for regressions.
- DayCare `tests/test_rloo_posttool.py` (only with `DAYCARE_TRAIN_TINYGRAD_PATH` pointing at a tree that has
  `compact`; skipped otherwise) adds a compacted multi-turn run through `Loop.learn`: 17 turns, windows at
  4 and 8 of 8 rows, `max_logprob_error` 0, exact == checked == sampled tokens.

## Process (GPU A/B, fixed before measuring)

- After run 3 finishes, with `gpu-run time` (exclusive) for both arms. Code: refill worktrees (DayCare experiment branch +
  this change, tinygrad-arkey `exp` + this change). Protocol name `rloo-slot-refill.md`, not counted.
- Both arms use run 3's exact flags and seed (20260924), `--updates 8`, `--lanes 32`: baseline `--compact off`,
  refill `--compact 8,16`. VRAM is sampled by `nvidia-smi` every 1 s.
- Report per arm:
  - s/update over updates 1-7 (update 0 carries JIT capture; its time is reported separately, since compaction
    captures 2x more step graphs)
  - s/episode, sample_s, decode ms/step
  - `window_rows_s`, giving T(8), T(16), T(32)
  - peak VRAM
  - parity (max logprob error, exact/checked) on every update
- **Confound and its control** (adversarial review): once lanes move, the arms sample different episodes, and
  s/update is set by each update's straggler. Raw s/update over 7 updates therefore mixes speed with episode-length
  noise. The primary speed number is schedule-normalised: each arm's decode_s is divided by *its own*
  full-width cost, i.e. its logged decode_steps x the baseline arm's measured ms/step at 32 rows. This measures
  "compacted decode / uncompacted decode on the same schedule". The ratio is then applied to the 156-update
  historical decode share to predict s/update. Raw s/update is reported next to it.
- Same-math check:
  - both arms train on the same states per update (same shuffled picks)
  - both show 0 parity error on every token
  - reward, episode-length and turn distributions are compared per update and pooled (tokens are not expected to
    match: the noise stream differs). Pooled mean reward and mean episode tokens should agree within their
    sampling noise (2 standard errors over 7 x 32 episodes).

## Finding (2026-09-27, measured after run 3 finished; `gpu-run time`, same seed, run-3 flags, 8 updates)

Runs: `refill-ab-base` / `base4` (compact off) and `refill-ab-compact` / `compact3` (`--compact 8,16`), under
`<runs>/`. Each arm was run twice and reproduced itself exactly (every loss identical).
`base3` hung at update 4 in a 30 s GPU wait. A 3.6 GB foreign process was on the GPU during that "exclusive" run,
and the rerun `base4` was clean and bit-identical to `base`, so `base3` is excluded.

| | baseline (32 rows) | compaction (8/16/32) | ratio |
|---|---|---|---|
| s/update, updates 1-7 | 83.1 (83.8 rerun) | 63.0 (63.3 rerun) | **x0.76 (1.32x faster)** |
| s/episode, updates 1-7 | 2.60 | 1.97 | x0.76 |
| decode vs its own full-width cost (schedule-normalised) | 1.00 | **0.684** | predicted 0.59 |
| update 0 (includes JIT capture) | 219 s | 525 s | +306 s once (30 more step graphs) |
| step ms at 8 / 16 / 32 rows | - / - / 22.6 | 12.25 / 18.1 / 21.3 | linear model said 9.6 / 13.9 / 22.4 |
| parity: max logprob error, exact/checked | 0, all tokens, all updates | 0, all tokens, all updates | same |
| peak VRAM (nvidia-smi, own process) | 22.16 GB, then **20.56 GB** after the planner fix | 23.29 GB, then **20.67 GB** after the planner fix | +0.11 GB |

**Speed.** H's speed claim holds (x0.76 is under the 0.80 bound), but the point prediction (x0.64) was too
optimistic. Small-row steps cost more than the linear extrapolation: T(8) = 12.3 ms, not 9.6. They still run the
32-row sampling tail (final block + 131k-vocab head) and the fixed per-step work. Re-running the 258-update
historical bound (r1+r2+r3) with the measured T(k) gives x0.72 per update, consistent with the x0.76 measured on
these 7 updates, whose straggler mix differs. The one-time capture cost (+306 s) is repaid after ~15 updates. A
102-update run like run 3 saves ~29 min of ~2.4 h.

**Same math.**
- Both arms trained on the same states every update.
- Every sampled token was recomputed bit for bit (max error 0, exact == checked on all 16 updates).
- Pooled over updates 1-7 (7 x 32 episodes per arm), the distributions agree within noise:

| | baseline | compaction |
|---|---|---|
| tokens per episode | 889 ± 68 | 905 ± 74 |
| mean reward | 0.781 ± 0.028 | 0.772 ± 0.028 |
| turns per episode | 1.74 | 1.75 |

- Tokens differ between the arms, as predeclared (different noise stream once lanes move).

**VRAM** (the first compacted run failed H's VRAM clause; fixed).
- The first compacted run peaked at 23.29 GB, +1.1 GB, which fails H's <= 22.5 GB clause.
- The cause was not the lane moves. It was the memory planner's shared arena pool, which keyed arenas by each
  graph's lane colouring: 135 -> 264 arenas, 2.05 -> 2.88 GB (`GlobalCounters`, `LRU=0`).
- Graphs in one pool never run at once, so tinygrad-arkey now maps each graph's lanes onto pool arenas by size rank.
  Any injective map keeps a graph's own lanes apart, which is all the colouring guarantees.
- With the fix the compacted sampler warms into the same arena count, and the baseline itself drops to 20.56 GB,
  bit-identical results.
- Compaction now peaks at 20.67 GB, below the old 32-lane baseline.
- The lane/slot move copies also plan into the pool (a test checks moves allocate nothing).

**Not done / next.**
- B (pipelining across updates, ~x0.39 on paper) stays unbuilt: it changes the training math.
- Cheaper wins on top of C:
  - capture the compacted graphs lazily (only buckets a run reaches), to cut the +306 s capture
  - run the sampling tail at the compacted row count with a row-count-matched Tail replay (T(8) floor)
  - a 4-row graph

Code:
- tinygrad-arkey `exp`: `NemotronHRolloutSampler(compact=...)`, moves inside the pool, `shared_arenas` rank
  mapping; tests `test/unit/test_nemotron_h_sampler_compact.py`.
- DayCare: `rloo_posttool --compact 8,16`; `tests/test_rloo_posttool.py` compact block. The default stays off:
  a counted run opts in by flag.
