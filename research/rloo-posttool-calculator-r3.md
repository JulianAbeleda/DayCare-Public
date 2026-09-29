# RLOO on the post-tool calculator turn, run 3: run 1 on repair + relay, with a length term and a working KL

Status: **done: not adopted (G2 fail, 53 > 50; G1 +12.9 pass)**; predeclared (the maintainer, 2026-09-27 ~15:20 EDT, conditional on the repo principles; checked
against the maintainer's principles notes (private): aligned). Approved as drafted, including open question 3: trigger limits 1.5x
finished-episode length and a 5% floor on the capped-rate ratio (window 10). Open questions 1-2 stand as drafted
(length weight 1.0 from update 1; abstain repair 0.0 / relay -0.5). No design change after this line.
Revised after an adversarial review returned NO-GO on the first draft: hypothesis overclaimed the graded
reward, KL trigger calibrated on an inert KL, beta chosen by an estimate, lanes chosen by a VRAM measurement.
This is a DayCare LoRA RLVR feature for Nemotron 3 Nano 4B on the GameTerm harness. Earlier runs:
[run 1](rloo-posttool-calculator.md) (G1 pass +7.0, G2 fail 60 -> 95 blanks),
[run 2](rloo-posttool-calculator-r2.md) (overlong filter; runaway length, stopped at u54),
[audit](rloo-posttool-audit-20260927.md), [playbook](../docs/rl-run-playbook.md).

## Hypothesis

**What run 1 showed.** Run 1's G2 blanks were mostly turns cut off at the 4,096-token cap (adapter: 82 truncated,
1 voluntarily empty, 12 reasoning never closed, of 95). By category the increase over stock was miss +20 (27 -> 47),
relay +10 (4 -> 14), repair +6 (21 -> 27), empty -1. Run 1's length brake was the capped episodes' tokens times a
negative advantage, and the replay below shows **most of that brake came from miss (Countdown) episodes**: on run
1's own records, repair + relay alone pushed toward length in 3 of the first 4 blocks. Run 2 removed the capped
tokens from the loss and length ran away.

**What the graded reward does not do.** Scoring a blank below a wrong answer (the first draft's mechanism) barely
moves the length push: run 1 as run (0/1) -0.14 / -0.12 / -0.08 / -0.29 in blocks u10-u40, graded -0.17 / -0.16 /
-0.08 / -0.31. The blank value moves it by <= 0.05 over -1 to -3.

**H (what run 3 tests).** Run 1's recipe, minus miss and empty in training (repair 0.8 + relay 0.2), plus four
changes: (1) a per-group length term (Kimi k1.5, arXiv:2501.12599), weight 1.0, the replacement for the brake that
miss supplied; (2) a KL term whose coefficient counts (matched aggregation, beta 0.03 by measured gradient share);
(3) a dense penalty on repeated 40-token thinking n-grams (arXiv:2502.03373); (4) blank (-1.5) below wrong (-1), so
a voluntarily empty or unclosed reply no longer ties a wrong answer. **Predicted mechanism:** the length term makes
the sequence term push toward shorter episodes from the first update on this mix (replayed: negative in every
10-update block of both runs, below), before any looping starts, so capped turns stay near stock; the KL term damps
the late drift (run 1 u90+); the repetition penalty acts only on learned verbatim loops (it never fires on stock).
Prediction: run 1's repair gain is kept, relay does not regress, and blanks after a tool result fall below stock.

The honest give-up tier (relay -0.5, others 0.0) stays in the reward but is **latent, not claimed**: stock gave up
in 0 of 7,120 episodes (2,128 evaluation, 4,992 run-1/2 training), and RLOO reinforces only what is sampled.

Predictions (against the same stock arms as runs 1-2):

| Gate | Prediction |
|---|---|
| G1 held-out miss + repair + empty (93 x 4) | paired difference >= +5 points, CI lower bound > 0 (run 1: +7.0 [+1.4, +12.8]); repair >= 0.80 (stock 0.697, run 1 0.861) |
| G1b held-out relay (85 x 4) | within +/-3 points of stock 0.953; upper bound >= 0 |
| G2 blanks after a tool result, G1 + G1b (712 episodes) | adapter <= 50 (stock 60, run 1 95); miss blanks <= stock's 27 + 5 (miss is untrained) |
| G4 give-ups on relay | <= 2% of relay episodes (<= 6 of 340); expected 0 (vacuous if so, see G4) |
| G3 retention | zero plain-answer losses |
| Training | no stop trigger fires in 102 updates; capped-turn rate <= 5% per 10-update block |

Why it might fail: (a) the length term also shortens correct repair answers; replay says the cost is small
(finished repair length vs correctness: r = -0.01 to -0.12; accuracy 0.85 vs 0.86 for the shorter vs longer half
of run 1 u0-39), but a shorter correct answer that drops a check could lose G1. (b) Kimi warm up the length penalty
because it slows early training; we apply it from update 1 (replay requires a negative push from u0). (c) Miss is
not trained; run 1 lost 14 held-out miss points while training it, and whether it regresses without training is
open (G1 still contains miss at 30%). (d) Audit D5: 4 prompts x 8 per update under a constant-size Adam step
(~5e-5 RMS); one or two mixed groups set each step's direction, and lr 2e-4 was chosen by a 10-update Countdown
sweep that cannot see drift from update ~25 on. Run 3 does not change lr or batch size; the stop triggers are the
only guard.

## Process

### Reward (`posttool.graded_reward` + `posttool.group_length_term`, policy `graded`; evaluation scores the plain outcome)

| Final turn | Reward | Source |
|---|---:|---|
| correct by the task's rule (tagged answer; hedged or not) | +1 | run 1 rule |
| honest give-up (below; latent) | relay **-0.5**, repair/miss/empty **0.0** | 2601.20126, 2509.25760, 2607.10738 |
| wrong, unreadable, verifier error, another tool | -1 | 2601.20126, 2509.25760 (+1/0/-1) |
| blank: capped (`incomplete`), reasoning never closed, empty, calculate call at the turn cap | **-1.5** | below wrong (2502.03373 r_e) |
| **plus, per group**: length term x 1.0 | lambda = 0.5 - (len - min)/(max - min); correct: lambda, other: min(0, lambda) | Kimi k1.5, arXiv:2501.12599 Sec. 2.3.3 |

- **Length term.** `len` is the episode's sampled tokens (all turns); min and max are over the group of 8; equal
  lengths give 0. A correct episode earns +0.5 (shortest) to -0.5 (longest); any other outcome earns 0 to -0.5, so
  a short wrong answer is never rewarded for being short. With weight 1.0 the ordering holds within a group: the
  longest correct (+0.5) is above any give-up (<= 0) and any wrong (<= -1). Kimi add it "with a weighting parameter"
  without stating the value and warm it up; weight 1.0 is set by replay (next section), applied from update 1.
- **Give-up detection is deterministic.** It needs a completed final turn (eos, reasoning closed, no call),
  non-empty content, **no `<answer>` tag anywhere**, and a match of `posttool.GIVEUP` ("I'm not sure", "I couldn't
  find/verify/finish ...", "I don't know", "no solution found", "my best guess is"). A tagged answer is always scored
  by its tag, so hedging a wrong answer earns nothing. The detector fired on none of the 7,120 stock and run-1/2
  episodes, and never on "since we cannot have a fraction of a bus".
- **Wire envelope.** A give-up is plain assistant content: no system-prompt change and no new format; GameTerm
  parity holds.
- **Abstain values, refined from the maintainer's ~0.25.** arXiv:2601.20126 (Jha et al.) sweeps r_abs from -0.95 to +0.3;
  the best value is model-dependent and a high r_abs collapses to 100% abstention. With +1/-1 a calibrated policy
  commits when p > (1 + r_abs)/2: at 0.25 that puts 19 of 61 held-out repair states in the abstain region (a give-up
  scores 0 at evaluation, so that could cost G1); at 0.0, 7 of 61. Relay at -0.5 sets p > 0.25 (0 of 85 relay
  states). AWA-RL (arXiv:2607.10738): smaller refusal reward on easy items. TruthRL (arXiv:2509.25760): +1/0/-1.
- **Blank value.** -1.5 is set by the ordering alone (wrong > blank); the push is insensitive to it (<= 0.05).

### Replay: the length push on runs 1-2 saved records (playbook rule)

Scripts: `<scratch>/r3/{replay,norm,lenterm,lenterm2}.py` (episodes regrouped from `update-*/update.xml`, all 8
per group). Net length push = sum(A x length) / sum(|A| x length) per 10-update block; scale-free because Adam
normalizes the step; negative = toward shorter. "r3 mix" reweights each category's sums to repair 0.8 / relay 0.2.

| Replay | u0 | u10 | u20 | u30 | u40 | u50 | u60 | u70 | u80 | u90 | u100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run 1 as run (outcome 0/1, all categories) | +0.04 | -0.14 | -0.12 | -0.08 | -0.29 | -0.32 | -0.35 | -0.32 | -0.24 | -0.17 | -0.25 |
| run 1, graded, all categories | +0.01 | -0.17 | -0.16 | -0.08 | -0.31 | -0.34 | -0.36 | -0.34 | -0.25 | -0.21 | -0.27 |
| run 1, miss groups alone (graded) | -0.17 | -0.27 | -0.30 | -0.27 | -0.36 | -0.49 | -0.39 | -0.33 | -0.27 | -0.57 | -0.32 |
| run 1, graded, r3 mix | +0.22 | -0.05 | +0.01 | +0.24 | -0.27 | -0.19 | -0.27 | -0.41 | -0.19 | +0.05 | -0.29 |
| run 1, graded, r3 mix + miss 0.2 | +0.02 | -0.15 | -0.04 | +0.04 | -0.29 | -0.33 | -0.32 | -0.38 | -0.23 | -0.25 | -0.30 |
| run 1, graded, r3 mix + miss 0.3 | -0.03 | -0.18 | -0.07 | -0.03 | -0.30 | -0.37 | -0.33 | -0.37 | -0.24 | -0.34 | -0.30 |
| run 1, r3 mix + length term w 0.5 | +0.10 | -0.22 | -0.08 | +0.10 | -0.35 | -0.31 | -0.34 | -0.48 | -0.27 | -0.15 | -0.36 |
| **run 1, r3 mix + length term w 1.0 (run 3)** | **-0.03** | **-0.33** | **-0.16** | **-0.04** | -0.40 | -0.40 | -0.38 | -0.52 | -0.32 | -0.27 | -0.41 |
| run 2 as run (filter + soft penalty) | +0.19 | +0.06 | +0.06 | +0.05 | -0.29 | -0.65 | | | | | |
| run 2, graded, r3 mix | +0.12 | -0.15 | -0.12 | -0.28 | -0.57 | -0.70 | | | | | |
| **run 2, r3 mix + length term w 1.0 (run 3)** | **-0.08** | -0.31 | -0.33 | -0.43 | -0.62 | -0.71 | | | | | |

- **Graded alone is not a length brake on this mix.** On run 1's calm repair + relay episodes it pushes toward
  length in 3 of the first 4 blocks. Few of those episodes were capped; the brake engages only once capping appears.
- **Decision: a direct length signal is needed, and the per-group length term at weight 1.0 is the minimal one.**
  Weights 0.05-0.5 leave u0 and u30 positive; 1.0 is the smallest tried (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
  that is negative in every block of both runs. Keeping miss in the mix also works at a 0.3 share (u0-u30: -0.03 /
  -0.18 / -0.07 / -0.03) but trains Countdown search again, which run 1 lost 14 held-out points on and which gave the
  largest blank increase (+20); it is rejected. Miss 0.2 alone leaves u30 positive.
- The length term does not trade against accuracy in these records: among finished repair episodes, corr(length,
  correct) = -0.01 (run 1 u0-39), -0.09 (u40+), -0.05 / -0.12 (run 2).
- The repetition penalty (upper bound) moves these sums by <= 2.5% before u40 and 4-6% after it.

### Repetition penalty (`posttool.thinking_repetition`; `--repetition n=40,penalty=-0.05`)

- Demystifying Long CoT (Yeo et al., arXiv:2502.03373, Algorithm 1, Table 7: N = 40, P = -0.05, dense). Every
  thinking token (before the turn's `</think>`; all of a turn that never closed) inside a 40-token n-gram seen
  earlier in that turn gets -0.05 added to its own advantage (`token_bonus` -> `rloo_tinygrad.flatten`). Immediate,
  not discounted (a deviation: the paper discounts it in PPO), not baselined. **`</think>` must be token 13:**
  `rloo_posttool.think_end_id` raises otherwise (it used to fall back to None, which would penalize answer tokens
  as thinking; `test_think_end_id_is_asserted_not_guessed`).
- n = 40 on the ~188-token tails the records keep: stock evaluation 0.00 of capped (42) and finished (670) episodes
  repeat; run-1 adapter capped 0.33 (repeated share 0.244), run-2 training capped 0.63 (0.558); finished <= 0.02.
  n = 40 detects the **learned** verbatim loop with near-zero false positives; at n = 10, 48-65% of finished episodes
  would be penalized. At the upper bound a looping capped episode carries about -115 advantage-token units, ~2% of its
  sequence term: a targeted push on the repeating tokens, not the brake.

### Capped episodes: in the loss and the baseline (`--mask none`)

No filtering (DAPO's premise fails here; audit D2). A capped episode keeps all its sampled tokens in the loss,
counts in its group's leave-one-out baseline, scores -1.5 and is the group's longest (length term min(0, lambda)).

### KL: matched aggregation, beta 0.03 by measured gradients (`--kl-aggregation matched --kl-beta 0.03`)

**Measured, not estimated** (`<scratch>/r3/beta_grad.py`, `gpu-run check`, no optimizer step). Run 1's raw
checkpoint at u30 (`checkpoint-030.npz`; KL vs stock 1.1e-2 token mean, as run 1 logged at u30) as the policy, the
stock adapter as reference, 3 real batches of 4 run-3-mix states x 8 sampled at 4,096 tokens, scored with the
run-3 reward (graded + length term + repetition bonus). On each batch, the adapter gradient of the PG term alone and
of the KL term alone (matched, per unit beta), before clipping:

| batch (tokens, mixed groups) | KL | \|g_PG\| | \|g_KL\| per beta | cos(g_PG, g_KL) | beta\|g_KL\|/\|g_PG\| at 0.01 / 0.03 / 0.05 / 0.1 / 0.3 / 1.0 | cos(total, g_PG) at 0.03 / 0.05 / 0.3 |
|---|---:|---:|---:|---:|---|---|
| 0 (14,226, 4/4) | 1.09e-2 | 34.0 | 141 | +0.27 | 0.04 / 0.12 / 0.21 / 0.41 / 1.24 / 4.1 | 0.993 / 0.983 / 0.746 |
| 1 (21,750, 4/4) | 1.07e-2 | 16.3 | 215 | -0.45 | 0.13 / 0.40 / 0.66 / 1.32 / 3.97 / 13.2 | 0.918 / 0.765 / -0.221 |
| 2 (23,826, 4/4) | 1.19e-2 | 57.3 | 260 | -0.35 | 0.05 / 0.14 / 0.23 / 0.45 / 1.36 / 4.5 | 0.991 / 0.974 / 0.384 |

- **The first draft's beta 0.3 would have made KL the dominant direction**: 1.2-4.0x the PG norm, and the update
  direction's cosine with the PG fell to 0.75 / -0.22 / 0.38. The draft's estimate (share ~0.1 at this KL) was off by
  13-42x because it assumed a per-token score norm of ~1; the measured KL gradient is 4-13x the PG's per unit beta.
- **Rule:** beta is the largest grid value at which, on every measured batch at run 1's u30 KL, the KL gradient is
  at most 0.4x the PG's and the summed gradient keeps cosine >= 0.9 with the PG alone (KL bends the step, never
  turns it). **beta = 0.03** (0.12 / 0.40 / 0.14; cosine 0.993 / 0.918 / 0.991); 0.05 fails batch 1 (0.765).
- It is a brake: in 2 of 3 batches cos(g_PG, g_KL) < 0, so the KL term opposes the PG's move away from stock, and its
  gradient grows with drift (0 at the reference). Near stock it is negligible; at the trigger KL (0.025, ~2.3x) it
  grows ~1.5x, to ~0.2-0.6x the PG.
- Clipping (1.0) and Adam act on the sum; Adam normalizes the step size, so the ratio above (the direction), not the
  absolute norm, is what matters (audit D3). P2 logs `kl_weight_ratio` and `kl` every update.

### Stop triggers: ON at run-3 defaults (`rl_triggers.DEFAULTS`); drift rule

With a working beta the KL trigger may stay quiet, so run 3 adds the audit's length triggers. Means of the last 10
stepped updates vs the first 10 (window 10; the capped-rate floor and 1.5x length are set by the replay below):

| Trigger | Limit |
|---|---|
| KL (token mean) | > 0.025 |
| entropy | < 0.85x baseline |
| capped-turn rate | > 15%, or > max(2x baseline, 5%) |
| finished-episode length | > 1.5x baseline |
| outcome reward (0/1, not the graded reward) | < baseline - 0.2 |
| `advantage_length_sum` (the sequence term's length push, logged) | net positive over the window (checked from u20), or positive 10 updates in a row |
| skipped updates | 3 in a row |

Replay (`python -m daycare.nursery.rl_triggers RUN --run3-reward --categories repair relay`, pinned in
`tests/test_rl_triggers.py`): rates and push recomputed from repair + relay episodes under the run-3 reward.

| Trigger alone (window 10) | run 1 trips at | run 2 trips at (collapse u45) |
|---|---|---|
| KL > 0.025 | u98 | u35 |
| entropy < 0.85x | u102 | u38 |
| capped > 15% | never | u47 |
| capped > max(2x, 5%) | never | **u32** |
| capped > 2x, no floor (audit as written) | u17 (0.6% vs a 0% baseline) | u12 |
| finished length > 1.5x | never | u41 |
| finished length > 1.3x (audit as written) | **u25** | u39 |
| outcome reward < base - 0.2 | never | u51 |
| push net positive over 10 / 10 in a row, run-3 reward | never / never | never / never |
| push net positive over 10, as-trained reward | never | **u20** |
| **all, run-3 defaults** | **u98 (KL)** | **u32 (capped rate)** |
| all, KL and entropy silenced | never | u32 |

- Run 2 trips at u32, 13 updates before its collapse, on a length trigger that does not depend on KL. Run 1 trips at
  u98 on KL (its end-of-run drift: KL 3.9-4.9e-2 at u95-101); without KL and entropy it never trips.
- The audit's 1.3x length and plain 2x capped limits would have stopped run 1 (which passed G1) at u25 and u17;
  that is noise on a 0-3% capped baseline and a transient length spike (817 tokens at u20-24), hence 1.5x and the
  5% floor. This deviation from the audit is deliberate and replayed.
- The push triggers never fire under the run-3 reward on either run, because the replayed push is negative there
  (the point of the length term). Their job is the case run 2 was: as trained, run 2's push was net positive from
  u10 and the window trigger fires at its first check (u20); run 1 as trained never. The audit's 10-in-a-row form is
  weaker (run 2's longest positive streak was 9) and is kept only because the audit specified it.
- **Drift rule (the maintainer, 2026-09-27):** on any trip the run stops itself (adapter kept, `stopped` in the record).
  Before any new run, a diagnosis is written from the saved records: which term moved, and why. No resume, no rescue.

### Training mix: repair 0.8 + relay 0.2 (`--categories repair relay --mix repair=0.8,relay=0.2`)

- Repair carries the signal: in run 1, 80% of repair groups were mixed (49/61, first half) and repair rose 0.775
  -> 0.842, matching the held-out +16.4.
- Relay stays at 0.2: only 29% of relay groups were mixed, but relay is the counter-pressure against abstaining on
  easy items (relay abstain -0.5 needs relay in the batch), and run 1's G1b lost 2.9 points with 11 relay truncations.
- **No miss or empty** (replay above for why not as a brake). Both stay in the G1 exam (30% + 4%).
- `training_mix` gives 125 repair + 31 relay = 156 training states (seeded).

### Rejected alternative: budget forcing / AnytimeReasoner at the cap

AnytimeReasoner (arXiv:2505.13438), s1 (arXiv:2501.19393) and Elastic Reasoning (arXiv:2505.05315) force or budget
the end of thinking. Rejected as a training term: GameTerm (llama-server) does not force-close thinking at
max_tokens, so rewarding forced answers trains a path deployment never takes, and each capped episode would need a
second prefill + decode. As an inference-only GameTerm change it is being measured separately
([posttool-budget-forcing.md](posttool-budget-forcing.md)).

### Fixed settings (as run 1 unless listed); one configuration, predeclared

- **Starts from stock** (fresh zero-initialized LoRA). Nemotron 3 Nano 4B BF16, thinking on, LoRA rank 32 /
  alpha 64, final MLP block; sequence RLOO, group 8, **4 states per update, 32 lanes, 102 updates** (3,264
  episodes, 102 Adam steps, as run 1), Adam lr 2e-4, entropy 1e-3 (matched), clip 1.0, TIS cap 2, 0.1-nat parity
  stop, temperature 1.0, 4,096 tokens per turn, 8-turn cap, seed 20260924, `posttool-tasks-002`.
- **Lanes are not a treatment choice.** 64 lanes x 51 updates would halve the Adam steps and double the batch, a
  different optimizer regime (D5). Run 3 runs 32 lanes regardless of any VRAM or speed measurement.
- Stack: tinygrad-arkey exp 4322ae9e2 (`<worktree>`); DayCare from a detached worktree at the
  commit recorded in the run.
- Command: `rloo_posttool train --categories repair relay --mix repair=0.8,relay=0.2 --mask none --reward graded
  --wrong -1 --blank -1.5 --abstain relay=-0.5,repair=0,miss=0,empty=0 --length-weight 1.0
  --repetition n=40,penalty=-0.05 --kl-aggregation matched --kl-beta 0.03 --lanes 32 --prompts-per-update 4
  --updates 102 --protocol rloo-posttool-calculator-r3.md` (triggers at `rl_triggers.DEFAULTS`).
- Logged per update: `giveup_rate`, `advantage_length_sum` (trained set, run-3 reward incl. length term),
  `repeated_thinking_tokens`, `kl_weight_ratio`, plus the run-1/2 metrics. Runs 1-2 remain reproducible (policies
  `outcome`/`soft_overlong`, masks `none`/`overlong_filter`, `legacy_token_mean`, `rl_triggers.LEGACY`).

### Evaluation (unchanged; stock arms reused)

Same held-out exam, 4 episodes per state and arm, seeds (stock 20260925, adapter 20260926), outcome scoring,
task-clustered bootstrap (50,000 resamples, rng 20260923). Stock arms: `posttool-g1-stock-001`,
`posttool-g1b-stock-001`, `posttool-g3-stock-001`. Give-ups score 0 and are reported per category and arm.

### Checklist and gates

- [x] P1. The maintainer reviews this draft and marks it predeclared (after the adversarial review). Done 2026-09-27 ~15:20 EDT.
- [x] P2. Unit and tiny-model tests (`tests/test_posttool.py`, `tests/test_rloo_posttool.py`,
  `tests/test_rl_triggers.py`: graded ordering, give-up detector, length term, repetition positions, `</think>` = 13
  asserted, zero bonus bit-neutral, capped episodes in the loss, run-1 path bit-exact, trigger replays). Then one
  real-model update with the run-3 policies, export + `verify` bit-exact, and report `kl_weight_ratio`.
  Separate speed note, **not counted and not a treatment input**: 64 vs 32 lanes, 3 updates each (`gpu-run time`).
- [x] P3. Train (102 x 32). Stop triggers on at defaults; any trip ends the run under the drift rule.
- [x] G1 (pass). Held-out miss + repair + empty (93 x 4): pass if the lower bound of the 95% interval is above zero.
- [x] G1b (pass). Held-out relay (85 x 4): fail if the interval's upper bound is below zero.
- [x] G2 (**fail**: 53 > 50). Blank answers after a tool result (final turn with no content and no call, truncated included), G1 +
  G1b episodes: **pass if adapter <= 50** (the prediction; stock 60) **and** the paired task-clustered bootstrap
  95% CI of (adapter - stock) does not lie entirely above 0. The CI is reported. Why not "significantly fewer": the
  CI half-width is ~20 blanks (run 1: +35 [+16, +55], 117 tasks), so a CI-below-zero gate would fail a 10-blank
  improvement. Reported per category with **miss separately** (stock 27; run 1 47; miss is untrained here): repair
  21, relay 4, empty 8 for stock. Give-ups are not blanks and are reported next to G2.
- [x] G4 (vacuous: 0 give-ups). Give-ups on held-out relay episodes <= 2% (<= 6 of 340). **If the adapter gives up 0 times, G4 is
  vacuous** (it tests nothing about abstention) and is reported as "vacuous: 0 give-ups", not as a pass of the
  abstention design. Per-category give-up rates reported for both arms.
- [x] G3 (not gated: G2 failed; run post-stop as information only, net real -2). Retention (only if G1, G1b, G2, G4 pass): the 132-item suite through HT-020, thinking on, temperature 0;
  zero plain-answer losses, no net real loss after hand audit.
- [ ] Stop rule: a failed gate ends the experiment; no rescue runs. Stop spinning: if run 3 fails G2 again for the
  same reason as run 1, stop and bring the maintainer the diagnosis; no fourth variant.

## Open questions for the maintainer (resolved 2026-09-27: all as drafted)

1. Length term weight 1.0 from update 1 (replay-chosen) vs Kimi's warm-up (length term only after an initial phase).
   Warm-up would leave u0-u9 with a positive push on this mix (+0.22 in run 1's replay).
2. Abstain values repair 0.0 / relay -0.5, or ~0.25 (19 of 61 repair states in the abstain region)? Moot if give-ups
   stay at 0.
3. The trigger deviations from the audit (1.5x length, 5% capped floor, window 10), each replayed above.

## Log

- **P2** (2026-09-27, not counted; tinygrad-arkey 4322ae9e2, `gpu-run time`). Tests: `test_posttool`,
  `test_rloo_posttool`, `test_rl_triggers` 25 passed. `posttool-rloo-r3-check-001` (DayCare (private commit), 1 update, 32
  lanes, 4,096 tokens): 156 training states (125 repair + 31 relay); the record stores mix repair 0.8 / relay 0.2,
  reward graded (+1 / wrong -1 / blank -1.5 / abstain relay -0.5, others 0), length weight 1.0, repetition n 40,
  -0.05, `think_end` 13, KL matched, beta 0.03, entropy 1e-3, lr 2e-4, seed 20260924, mask none, triggers at
  `rl_triggers.DEFAULTS`. Update 1: reward 0.844, 4/4 groups mixed, parity exact (max gap 0), `kl_weight_ratio`
  **0.040**, `advantage_length_sum` -828 (toward shorter), 0 repeated thinking tokens, give-up rate 0. Export +
  `verify` in a fresh process bit-exact (4,194,304/4,194,304).
  - **Defect found and fixed before the counted run:** `update.xml` wrote the length-shaped group rewards as
    `np.float64(x)` (NumPy 2 repr), which `record_xml.read` cannot parse; the drift rule's diagnosis reads these
    records. Fix (`record_xml` writes `repr(float(v))`, test added); training numerics untouched. Rerun after
    the fix, `posttool-rloo-r3-check-002`: identical update (same loss, raw adapter sha256 c7bb9179... equal to
    check-001), `verify` bit-exact, the record reads back (rewards 1.5, 0.999, -1.5, ...).
  - Speed/VRAM note (not counted, **not a treatment input**; 3 updates each, same seed, `gpu-run time`): 32 lanes
    378.8 s total (217 / 52 / 110 s), peak VRAM 22.2 GB; 64 lanes 530.2 s (240 / 115 / 175 s), peak 31.8 GB of
    32 GB. At 4 prompts x 8 per update, 64 lanes was not faster here and leaves no VRAM headroom. Run 3 stays at
    32 lanes as predeclared.
- **P3 started** (`posttool-rloo-r3-001`, counted, DayCare (private commit) detached worktree `<worktree>`,
  from stock (zero adapter), 32 lanes x 102 updates, `--keep-every 10`, export on, triggers on at defaults).
  - u1-20 (per 10: u1-10 / u11-20): mean reward 0.788 / 0.822, capped-turn rate 1.9% / 3.0%, KL 1.1e-3 / 2.7e-3
    (token mean; 2.7e-3 at u20), entropy 0.613 / 0.591, finished-episode length 666 / 526 tokens,
    `advantage_length_sum` -6,691 / -6,091 (toward shorter every block), `kl_weight_ratio` 0.045 / 0.053, repeated
    thinking tokens ~25 per update, give-ups 0, empty-after-tool 8.4% / 7.5%; parity exact; no trigger.
  - u21-40 (per 10): mean reward 0.828 / 0.810, capped-turn rate 1.6% / 3.6%, KL 2.2e-3 / 2.8e-3, entropy 0.570 /
    0.598, finished-episode length 619 / 644, `advantage_length_sum` -5,645 / -3,392 (toward shorter), `kl_weight_ratio`
    0.040 / 0.066, repeated thinking tokens ~50 per update, give-ups 0, empty-after-tool 4.7% / 8.4%; ~85 s per update;
    no trigger.
  - u41-60 (per 10): mean reward 0.850 / 0.850, capped-turn rate 2.2% / 1.9%, KL 4.4e-3 / 4.1e-3, entropy 0.591 /
    0.568, finished-episode length 539 / 496, `advantage_length_sum` -4,594 / -5,454, `kl_weight_ratio` 0.060 / 0.045,
    repeated thinking tokens ~20 per update, give-ups 0, empty-after-tool 4.4% / 4.7%; raw checkpoints to u60; no
    trigger.
  - u61-80 (per 10): mean reward 0.838 / 0.860, capped-turn rate 1.6% / 3.3% (u71-75 alone 6.7%), KL 4.0e-3 /
    5.6e-3, entropy 0.547 / 0.577, finished-episode length 509 / 511, `advantage_length_sum` -4,501 / -6,206,
    `kl_weight_ratio` 0.045 / 0.054, repeated thinking tokens ~31 / ~47 per update, give-ups 0, empty-after-tool 4.1% /
    6.3%; no trigger.
  - u81-102 (per 10: u81-90 / u91-100 / u101-102): mean reward 0.838 / 0.909 / 0.938, capped-turn rate 3.2% / 1.6% /
    0%, KL 6.2e-3 / 5.8e-3 / 5.4e-3, entropy 0.583 / 0.551 / 0.504, finished-episode length 472 / 429 / 497,
    `advantage_length_sum` -4,232 / -5,608 / -2,264, `kl_weight_ratio` 0.052 / 0.059 / 0.063, give-ups 0.
- **P3 complete** (`posttool-rloo-r3-001`, counted; DayCare (private commit), tinygrad-arkey 4322ae9e2): 102 updates, **no stop
  trigger**, parity exact throughout (max gap 0, rho 1.0), 2.02 h (71 s per update). Over the run: mean reward 0.788
  (u1-10) -> 0.909 (u91-100); finished length 666 -> 429 tokens; capped-turn rate 0.5-6.7% per 5 updates; KL peak
  6.2e-3 (limit 0.025); entropy 0.613 -> 0.551; the length push negative in every block; 0 give-ups in 3,264
  episodes. Export + `verify` in a fresh process bit-exact (4,194,304/4,194,304); raw adapter backed up to
  `<backups>/posttool-rloo-r3-001/` (sha256 checked). G1/G1b adapter arms next
  (`posttool-r3-{g1,g1b}-adapter-001`, seed 20260926; stock arms reused as predeclared).
- Gates (2026-09-27): adapter arms `posttool-r3-g1-adapter-001` (93 states x 4, 604 s) and
  `posttool-r3-g1b-adapter-001` (85 x 4, 271 s), DayCare (private commit), seed 20260926, 4,096 tokens, cap 8, `gpu-run time`;
  stock arms reused (`posttool-g1-stock-001`, `posttool-g1b-stock-001`). Scored by `posttool-rloo-r3-001/gates.py`
  (run 1's procedure plus the G2 bootstrap and G4); it reproduces run 1's numbers exactly (G1 +7.0 [+1.4, +12.8],
  G2 +35 [+16, +55]) when pointed at run 1's arms.

## Finding

**Verdict: not adopted.** G1 and G1b pass; **G2 fails its predeclared threshold (53 blanks, limit 50)**, which by the
stop rule ends the experiment. G3 was not run. G4 is vacuous (0 give-ups). Training met its predictions (no stop
trigger in 102 updates; capped-turn rate <= 3.6% per 10-update block).

| Gate | Prediction | Result | Verdict |
|---|---|---|---|
| G1: held-out miss + repair + empty (93 x 4) | >= +5 points, lower bound > 0; repair >= 0.80 | stock 0.669, adapter **0.798**; **+12.9 points [+7.9, +18.2]** (75 task clusters, 50,000 resamples); 41 states gained, 11 lost; repair **0.865** | pass |
| G1b: held-out relay (85 x 4) | within +/-3 of 0.953; upper bound >= 0 | 0.953 -> 0.962; **+0.9 [-2.1, +3.8]**; 9 gained, 7 lost | pass |
| G2: blanks after a tool result, G1 + G1b (712 episodes) | adapter <= 50 and CI not all above 0; miss <= 32 | stock 60 -> adapter **53**; difference **-7 [-21, +7]** (115 clusters); miss 27 -> 24 | **fail** (53 > 50) |
| G4: give-ups on held-out relay | <= 6 of 340 | 0 of 340 (stock 0) | vacuous: 0 give-ups |
| G3: retention | zero plain-answer losses | not run (G2 failed) | not run |
| Training | no trigger; capped <= 5% per 10-update block | no trigger in 102 updates; capped 1.6-3.6% per block | met |

**Reading.** G2 failed by 3 blanks against a threshold we set at the prediction; the change in blanks (-7) has a
CI that includes zero, so the adapter neither clearly cut nor added blanks. That is a different failure from run 1
(blanks +35 [+16, +55], driven by capped turns). The mechanism predicted for G2 did not show on the held-out
exam: turns cut at the 4,096 cap were 42 (stock) vs 43 (adapter) in G1 and 0 vs 0 in G1b. The 7 fewer blanks came
from reasoning never closed (13 -> 8) and voluntarily empty replies (5 -> 2), not from fewer capped turns. The
length term shortened finished answers (repair 614 -> 579 tokens, relay 244 -> 205) but not the Countdown-style
turns that run out the budget (miss and empty are untrained here).

### What changed, as GameTerm scenarios

- **Relay**: the calculator result is (or leads directly to) the answer; map it back to the question.
- **Repair**: GameTerm rejected the call's arguments, or the calculator refused the expression; fix and retry, or
  answer another way.
- **Miss**: Countdown; the result is not the target yet; keep searching (not trained in run 3).
- **Empty**: the probe's states where stock had answered nothing after a tool result (not trained in run 3).

| Scenario | Questions | Share of exam | Episodes | Correct, stock -> adapter | Score, stock -> adapter (paired 95% CI) |
|---|---:|---:|---:|---:|---|
| relay | 85 | 48% | 340 | 324 -> 327 | 0.953 -> 0.962 (+0.9 [-2.1, +3.8]) |
| repair | 61 | 34% | 244 | 170 -> **211** | 0.697 -> **0.865** (+16.8 [+9.8, +24.2]) |
| miss | 28 | 16% | 112 | 75 -> 81 | 0.670 -> 0.723 (+5.4 [-0.9, +11.6]) |
| empty | 4 | 2% | 16 | 4 -> 5 | 0.250 -> 0.312 (+6.2 [+0.0, +12.5]; 4 states) |

What the GameTerm user sees on a failure (episode counts, stock -> adapter):

| Scenario | Blank after a 4,096-token think (cut off) | Blank: reasoning never closed or empty | Another tool called | Wrong or unreadable answer |
|---|---|---|---|---|
| relay | 0 -> 0 | 4 -> 1 | 2 -> 0 | 10 -> 12 |
| repair | 8 -> 10 | 13 -> 8 | **19 -> 2** | 34 -> 13 |
| miss | 27 -> 23 | 0 -> 1 | 4 -> 2 | 6 -> 5 |
| empty | 7 -> 10 | 1 -> 0 | 1 -> 0 | 3 -> 1 |

Blanks after a tool result (G2) by category, stock -> adapter: repair 21 -> 18, relay 4 -> 1, **miss 27 -> 24**
(reported separately: untrained, and within its <= 32 limit), empty 8 -> 10; total 60 -> 53.

Side effects: capped turns G1 42/575 -> 43/560, G1b 0/401 -> 0/396; tool turns per episode 1.55 -> 1.51 (G1), 1.18
-> 1.16 (G1b); finished-episode length repair 614 -> 579, relay 244 -> 205, miss 1,026 -> 1,041, empty 1,655 ->
1,368 tokens; give-ups 0 in both arms. Compared with run 1's adapter (same exam): repair 0.861 -> 0.865 (kept), relay
0.924 -> 0.962, miss 0.527 -> 0.723, blanks 95 -> 53, relay capped turns 11 -> 0.

Worked examples, from the evaluation rollouts (four episodes per arm):

- **Repair, a win** (`repair:relay-jugs-09:rejected`). "A tank holds 1233.3 liters. How many 8.67-liter jugs can be
  filled completely?" (142). GameTerm rejected the call `1233.3 / 8.67` ("malformed arguments: see the tool schema").
  - Stock: 1 of 4. Two episodes called another tool, one never closed its reasoning (blank).
  - Adapter: 4 of 4, `<answer>142</answer>` (one after a retried call, three reasoning directly; 236-593 tokens).
- **Miss, a win (untrained)** (`miss:miss-4-440015`). "Using [8, 22, 31, 46], make 46"; the call `8+22+31+46 = 107`
  missed.
  - Stock: 1 of 4; three episodes thought to the 4,096 cap and the user got a blank.
  - Adapter: 3 of 4 (`46 / (31 - 22 - 8)`, or 46 x (31 - 22 - 8)), in 873-3,399 tokens; one reused a number.
- **Relay, a win** (`relay:relay-vans-10:own`). "7,691 people, buses seat 51; how many buses?" (151). The model's
  own call was `(7691 + 50) / 51` and returned 151.78.
  - Stock: 1 of 4; three relayed the rounded result as `<answer>152</answer>`.
  - Adapter: 3 of 4 answered 151; one relayed 152.
- **Relay, a regression** (`relay:relay-download-10:own`, also run 1's example). The call `111872 / 89.9 + 60`
  returned 1304.4.
  - Stock: 4 of 4 answered 21.
  - Adapter: 2 of 4; two relayed the tool output as the answer (`<answer>1304</answer>`, `<answer>1305</answer>`) in
    ~60 tokens, with no blank.

Per the stop rule this experiment ends here: no rescue run, no G3. The adapter (`posttool-rloo-r3-001`, raw sha256
in `adapter-raw.sha256`, backed up) is not adopted. Stop-spinning check: run 3 did not fail G2 for run 1's reason
(blanks fell instead of rising; capped turns unchanged instead of doubling), but it is the second gated run to fail
G2, so the next step is the maintainer's call, not a fourth variant. Open for the maintainer: (1) the G2 threshold was set at the
point prediction (<= 50) while its CI half-width is ~14-20 blanks, so a pass needed a large effect; (2) the
held-out blanks are dominated by capped Countdown turns (miss + empty: 34 of 53 adapter blanks), which run 3 does
not train; (3) G1 +12.9 [+7.9, +18.2] is the strongest post-tool result so far, with relay and miss not regressing.

### G3 retention, post-stop, INFORMATION ONLY (does not change the verdict; not an adoption)

Run after the stop at the maintainer's request (2026-09-27 ~18:50 EDT), only to decide whether run 3's adapter is a safe
starting point for a round 2. The verdict above (not adopted, G2 fail) stands. Procedure as predeclared:
`scripts/thinking_retention_gate.py` freeze + run, the frozen 132-item suite through HT-020, thinking on,
temperature 0, 4,096 tokens, server seed 42, adapter `posttool-rloo-r3-001/adapter.gguf` served with the base model
(`posttool-r3-g3-adapter-001`, DayCare (private commit)); stock arm reused (`posttool-g3-stock-001`).

| Family | Items | Stock | Adapter | Adapter wins / losses (automatic) |
|---|---:|---:|---:|---|
| plain | 40 | 40 | 40 | 0 / **0** |
| math | 56 | 53 | 54 | 1 / 0 |
| oracle | 8 | 8 | 7 | 0 / 1 |
| selection | 16 | 15 | 13 | 0 / 2 |
| smoke | 4 | 3 | 2 | 0 / 1 |
| clarification | 8 | unmeasured | unmeasured | 2 replies differ; neither arm asks (see below) |

- **Plain-answer losses: 0** (40/40 both arms). **Empty answers after a calculator call: 0 of 62** items that called
  it (stock 0 of 68). No truncated reply in either arm. 84 of 132 replies are byte-identical to stock.
- **Hand audit of every disagreement** (5 automatic flips + 2 changed clarification replies):
  - `gsm8k-3018` (math, win, real): adapter "160 ounces." (correct, no calculator); stock called the calculator with
    a wrong expression and answered 244.
  - `gsm8k-3004-expression` (oracle, loss, **not real**): both answer "$22"; the adapter appends "... is $22 total
    for 4 batches", so the last-number rule reads 4. The answer is right; the extra sentence is sloppy.
  - `smoke-1-natural` (smoke, loss, **real**): the tool returned 121932631112635269; the adapter relayed
    12193263112635269 (dropped a digit). A relay copying error.
  - `files-fresh-08` (selection, loss, **real, minor**): read request under a blocked policy; both arms end with a
    correct refusal, but after the rejections the adapter also tried `terminal_open`, a tool outside the allowed set.
  - `files-fresh-12` (selection, loss, **real**): same blocked read; after rejections the adapter tried
    `write_file` twice (creating `test.txt` and `input.xml`, both rejected) and other reads, then ended in an
    inference failure (LimitExceeded) with no reply. Stock tried two allowed tools and explained the refusal.
  - `clarify-02`, `clarify-04`: both arms answer without asking (11.36 vs 11.356 liters; 30 students, the adapter
    shows the union arithmetic). No change in clarification behavior.
- **Net real change: 1 win, 3 real losses (net -2)**, none on plain answers. By the predeclared G3 rule (zero
  plain-answer losses, no net real loss after hand audit) this would **fail on net real loss**; it is informational
  only. For a round-2 start: calculator relay/plain behavior is intact, but the adapter is looser under repeated tool
  rejection outside the calculator (an unallowed tool, attempted writes on a read-only request) and made one
  digit-copy error.

## Sources (ids verified on arxiv.org, 2026-09-27)

- Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs", arXiv:2501.12599, Sec. 2.3.3 (per-group length
  reward lambda = 0.5 - (len - min)/(max - min); correct: lambda, incorrect: min(0, lambda); added with a weight;
  warmed up).
- Jha et al., "Rewarding Intellectual Humility: Learning When Not To Answer in Large Language Models",
  arXiv:2601.20126 (+1 / r_abs / -1; r_abs sweep; over-abstention collapse).
- Wei et al., "TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning", arXiv:2509.25760 (ternary +1/0/-1).
- Zhang et al., "To Answer or to Abstain: Mitigating Search-Agent Hallucinations via Abstention-Aware Reinforcement
  Learning" (AWA-RL), arXiv:2607.10738 (refusal reward lower on easy items).
- Yeo et al., "Demystifying Long Chain-of-Thought Reasoning in LLMs", arXiv:2502.03373 (Algorithm 1 n-gram
  repetition penalty, N 40, P -0.05).
- Qi et al., AnytimeReasoner, arXiv:2505.13438; Muennighoff et al., s1, arXiv:2501.19393; Xu et al., Elastic
  Reasoning, arXiv:2505.05315.
- Yu et al., DAPO, arXiv:2503.14476 (why run 2's filter failed here: audit D2).
