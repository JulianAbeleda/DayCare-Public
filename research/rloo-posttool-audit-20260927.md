# Audit: why post-tool RLOO run 2 failed, and what else in the RL loop is wrong (2026-09-27)

Scope: run 2 (`posttool-rloo-r2-001`, stopped by the maintainer at update 53) against run 1 (`posttool-rloo-r1-001`,
102 updates), and the whole update path: `daycare/nursery/rloo_tinygrad.py`, `rloo_posttool.py`, `rlvr.py`,
`daycare/harness/posttool.py`. Run-2 result itself: [rloo-posttool-calculator-r2.md](rloo-posttool-calculator-r2.md).
Code unchanged by this audit. Analysis scripts: `<scratch>/{ext,an*,analyze*}.py`, `data.json`
(every `update-*/update.xml` of both runs).

## Hypothesis

Candidate causes of run 2's collapse (outcome reward 0.82 at u21-40 -> 0.43 at u47-53; episodes ending at the
4,096-token cap ~5% -> 45-53%; finished-episode length 676 -> 1,112 tokens), and candidate defects in the loop:

- H1. Overlong filtering removed run 1's only length brake (capped episodes' negative-advantage tokens).
- H2. The soft overlong penalty (DAPO Eq. 13, buffer 819) had almost nothing to act on.
- H3. The KL (and entropy) terms are inert because of their normalization relative to the policy gradient.
- H4. Gradient clipping (clip 1.0 vs grad-norm 13-34, clip scale 0.03-0.29) is the effective learning rate.
- H5. Our leave-one-out baseline under filtering differs from DAPO/TRL and that is the cause.
- H6. Other defects: TIS, Adam, sign/reference of KL, reward for truncation, batch size.

## Process

- Read the update path end to end (line numbers below are at the audited commit).
- Recomputed per-update metrics from both runs' records (KL, entropy, step RMS, grad norm, clip scale, capped
  rate, finished length, trained episodes, rho) in 5- and 10-update windows.
- Counterfactual on run 2's own episodes: for each group, recomputed leave-one-out advantages A under three
  treatments of capped episodes and summed A x length (the token-sum loss's net push toward longer outputs):
  (a) filtered from loss and baseline (run 2 as run), (b) kept in loss and baseline (run 1's treatment, with the
  run-2 shaped reward), (c) kept in the baseline, masked from the loss (TRL `mask_truncated_completions`).
- Per-token weight ratio of the KL term to the PG term, from each update's |A|, trajectories and token count.
- Reference implementations, source read (cloned 2026-09-27): verl `verl/trainer/ppo/core_algos.py`
  (`agg_loss`, `compute_rloo_outcome_advantage`, `kl_penalty`), `verl/workers/utils/losses.py` (`ppo_loss`: KL
  and entropy aggregated with the PG's own `loss_agg_mode`), `verl/workers/reward_manager/dapo.py` (overlong
  buffer), `verl/utils/reward_score/math_dapo.py` (+1/-1 reward), verl-recipe `dapo/run_dapo_qwen2.5_32b.sh`;
  TRL `trl/trainer/rloo_trainer.py` (`_generate_and_score_completions`, `_compute_loss`) and `rloo_config.py`.

## Finding

### Ranked causes of the run-2 collapse

1. **H1 confirmed: filtering capped episodes from the loss removed the length brake.** Under the token-sum PG
   loss a capped episode contributes ~4,096 tokens x a negative advantage. Net sum(A x length) on run 2's
   episodes, per 10-update block (u0, u10, u20, u30, u40, u50):

   | treatment of capped episodes | u0 | u10 | u20 | u30 | u40 | u50 |
   |---|---|---|---|---|---|---|
   | (a) filtered, loss + baseline (as run) | +12.4k | +4.0k | +3.3k | +2.6k | -23k | -9k |
   | (b) kept in loss + baseline (run-1 style) | -1.3k | -36k | -28k | -69k | -323k | -89k |
   | (c) baseline only, loss masked (TRL) | +23k | +30k | +22k | +38k | +83k | +16k |

   Run 1 (treatment b, outcome reward) was net negative in 9 of 10 blocks after u10; the capped part alone was
   -13k to -53k per block. With it gone, the trained set pushed toward length for 40 updates: among trained
   finished episodes u0-39, corr(A, length) = +0.08 (n = 1,210): small, but nothing opposed it. Row (c) shows
   the cause is **masking the loss at all**, not our baseline choice (H5 rejected as the cause; see D4).
2. **H2 confirmed: the soft penalty had no mass to act on.** Finished episodes whose longest turn passed 3,277
   tokens: 1.2% in run 2 (p95 1,955, p99 3,405). Our length distribution is bimodal: turns either finish short
   or loop to the cap. The ramp grades the empty middle; the one place it reaches -1 (the cap) is filtered out.
3. **Contributing, H3 confirmed: no KL brake.** KL (token mean vs the stock reference) rose 4.7e-4 (u0-9) ->
   1.3e-2 (u20-29) -> 7.4e-2 (u40-49) in run 2 while nothing pushed back; details in D1.

H4 rejected (D3). Displacement: step RMS ~5e-5 per update in both runs, so the failure is direction, not size.

### Defects and deviations in the update path

- **D1 (bug, severity high): KL and entropy are ~1e-5 of the PG in weight.** `TailSteps._grad`
  (`rloo_tinygrad.py:282-298`): PG = sum over tokens of `weight x A / trajectories` (`scale`, l.344); KL and
  entropy use `mask = 1 / token_count` (l.351), a mean over all ~20k batch tokens, times beta 1e-3 (`CONFIG`,
  l.60). Per-token weight ratio KL:PG = (beta/T)/(mean|A|/N): median 8.1e-6 (run 1), 5.5e-6 (run 2), max
  6.3e-5. verl aggregates KL and entropy with the PG's own `loss_agg_mode` (`workers/utils/losses.py`, `ppo_loss`),
  so their coefficient means what it says; TRL RLOO subtracts beta x **sequence-summed** KL from the reward
  (`rloo_trainer.py`, `rewards = rewards - self.beta * kl`, beta default 0.05). Ours is weaker than verl's
  parity by ~T/N (~700x) and than TRL's by ~4 orders. KL form (exact full-vocab KL(pi||ref)), sign (`+ beta*kl`)
  and reference (`frozen_lora_tail`, the zero-init adapter = stock; `rlvr_update.py:11`) are correct. The
  mixed normalization is inherited from `rlvr.regularized_token_loss` (`rlvr.py:69-94`), whose docstring flags
  it as "our adaptation, not an inferred paper default". Entropy bonus is inert for the same reason; harmless.
- **D2 (design, severity high): masking capped episodes discards the only anti-loop signal.** `masked()`
  (`rloo_posttool.py:265-284`). DAPO's filtering premise (arXiv:2503.14476 Sec. 3.4) is that a truncated
  sample may be sound-but-long reasoning, so a penalty is noise. Ours are degenerate self-verification loops
  (run 1 finding: blanks are mostly truncation loops), so the penalty was signal. Also note verl's public DAPO
  recipe does **not** implement Overlong Filtering at all: truncated samples keep their +/-1 score plus the
  buffer penalty (reaching -2) in both loss and group advantage (`reward_manager/dapo.py`, `__call__`;
  `math_dapo.compute_score`).
- **D3 (not a bug): clipping is not the effective learning rate.** Adam divides by the RMS of the (clipped)
  gradients, so a uniform clip scale cancels; step RMS is flat ~5e-5 while clip scale ranges 0.03-0.29 (e.g.
  run 2 u40-49 clip 0.138, u50-53 0.035, RMS 5.6e-5 and 5.3e-5). First step RMS 1.4e-4 = lr 2e-4 / sqrt 2 (B = 0
  zeroes A's first gradient). Consequence worth knowing: because Adam normalizes, only the *relative* weights
  of terms matter, which is exactly why D1 bites; the PG's absolute scale (sum vs mean) is irrelevant.
- **D4 (deviation, low): our filter also drops capped episodes from the baseline.** TRL
  (`mask_truncated_completions`) masks their tokens but keeps them in the group's leave-one-out baseline;
  DAPO computes group statistics over all G outputs. Row (c) above: it would not have braked length either.
- **D5 (design, medium): tiny, noisy batches under a constant-size step.** 4 prompts x 8 per update; mixed
  groups averaged 1.4-3.1 of 4; after filtering run 2 trained on 16.8 of 32 episodes at u50+. Adam moves
  ~5e-5 RMS regardless, so one or two groups set each step's direction. DAPO: 512 prompts x 16 with dynamic
  sampling (`filter_groups`) that drops zero-variance groups; lr 1e-6 full fine-tune, KL off, clip-higher 0.28.
  Our lr 2e-4 was chosen by a 10-update Countdown sweep (`rloo-tinygrad-countdown-v2.md`), which cannot see
  drift that starts at update ~25.
- **D6 (fine): TIS never active.** rho max 1.000, clipped 0.0000 in every update of both runs (one on-policy
  step, bit-exact parity); `parity_weights` (`rloo_tinygrad.py:305-328`) is correct and dormant.
- **D7 (design, low): truncation scores the same as a wrong answer** (0, `posttool.episode_reward`,
  `harness/posttool.py:191-202`); run 1's brake came only from token count x negative advantage. DAPO scores
  wrong -1 and truncated -1 + penalty.
- **D8 (latent): `masked()` raises when no group keeps two finished episodes** (`rloo_posttool.py:282-283`), so a
  deeper collapse would have crashed the run rather than stopping it cleanly with a record.
- Soft penalty implementation (`harness/posttool.py:205-224`) matches DAPO Eq. 13 and verl's `dapo.py`
  (factor 1.0, applied to the longest turn); correct, just mis-regimed (cause 2).

### Process: why the design failed despite the literature

- **Regime did not transfer.** DAPO: 32B base, 20,480-token cap, buffer 4,096, truncation rare and lengths grow
  smoothly (a feature). Ours: 4B, 4,096 cap, truncation is a discrete loop failure. The predeclaration copied
  the buffer *ratio* (0.2) without checking where our length mass sits (p99 3,405 would have shown the ramp is
  empty). It also cited filtering + soft penalty as "used together" from the paper's cumulative ablation, while
  the released verl recipe ships only the penalty.
- **Survivorship bias was not anticipated** because the predeclaration reasoned about what filtering removes
  (noisy penalties) but not about what the removed tokens had been doing in run 1's gradient. A two-minute
  counterfactual on run 1's saved records (row b vs a above) would have shown the brake before run 2 started.
- **The inert KL went unnoticed since run 1** because the predeclaration's "checked, not changed" noted "only
  the KL and entropy terms are token means" without computing the ratio, the log prints `kl` (a value) but
  never the KL term's share of the loss or gradient, and no test pins that share.
- **No in-run stop trigger for collapse.** P3 stops only on parity, non-finite or FloatingPointError; gates run
  after training. Signals were visible ~15 updates before the collapse (u45): KL 1.8e-2 at u25-29 and 3.4e-2 at
  u30-34 vs run 1's 1.2-1.3e-2 at the same point; entropy 0.60 -> 0.50 by u30-34 (run 1 flat 0.50-0.56); finished
  length 965 at u35-39 vs 549-794 before. Run 1 itself was starting the same drift at its end (u95-101: KL
  3.9-4.9e-2, entropy 0.42, finished length ~790).

### Recommendations

Code (fix before run 3):
1. D1: aggregate KL and entropy with the PG's normalization (per-token weight `beta / trajectories`, i.e.
   summed over tokens / N), or move KL into the reward as TRL does; add a unit test pinning the KL:PG gradient
   share for a known batch, and log `beta*kl / |pg|` per update.
2. D8: when the mask leaves nothing to learn, skip the step and record it instead of raising.
3. Log per update: finished-episode mean/p95 length, capped-episode rate, sum(A x length) over the trained set,
   trained-episode count, KL term share.

Predeclaration template, stop/alarm triggers (checked every update, rolling 5):
- capped-episode rate > 2x the first-10 mean, or > 15%;
- finished-episode mean length > 1.3x the first-10 mean;
- entropy < 0.85x the first-10 mean; KL > 2x the prior run's envelope at the same update;
- outcome reward < first-10 mean - 0.15; sum(A x length) positive for 10 consecutive updates.
Any trigger: pause, save, report; resume only on the maintainer's call. And before any recipe change: replay it as a
counterfactual on the previous run's saved episodes (as in cause 1) and state which terms move.

Run 3 direction (not a predeclaration): keep capped episodes in the loss and baseline (run 1's brake), fix the
KL normalization so its coefficient is real, drop or re-regime the soft penalty (ramp where length mass is,
e.g. starting near finished p95), consider scoring wrong/truncated below zero, and more prompts per update or
zero-variance-group resampling so each Adam step is less noisy. Add the triggers above.
