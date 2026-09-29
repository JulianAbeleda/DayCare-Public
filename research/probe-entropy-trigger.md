# Probe entropy: an entropy drift check that batch composition cannot move

Status: **H + P committed before any measurement; measured, Finding below. Proposed rule not wired in.** Code: `daycare/nursery/probe_entropy.py` (+
`tests/test_probe_entropy.py`). Outputs: `<runs>/probe-entropy-001/`. Analysis scripts outside Git:
`<scratch>/probe/`. Nothing is wired into the training loop.

## Why

Run 6 stopped at u66 on the composition-adjusted batch-entropy trigger (`rl_triggers.DEFAULTS`, window/baseline
ratio < 0.85). The diagnosis (`<scratch>/r6/diag6.md`, [run 6 Log](rloo-posttool-calculator-r6.md)) found the
batch token entropy moved by which states and samples a batch drew: the baseline's 0.12 blocked share alone decided
whether run 6 tripped, and healthy runs 4, 5 and 6 reached window ratios 0.87-0.89 against the 0.85 limit, so the
trigger sits inside its own noise band. A drift check has to hold the states and the tokens fixed.

## Hypothesis

Mechanism: the LoRA sits on the final block only, so the hidden state entering that block is a function of the
tokens alone. Scoring a fixed set of stored stock episodes through each checkpoint's tail gives the policy's
teacher-forced next-token entropy on identical inputs: composition and sampling noise are removed, and what is left
is the policy's change plus the probe's own state-sampling error.

Predictions (made before measuring):
1. The stock tail reproduces the stored sampler log-probs (max error <= 1e-6; it is the parity check's forward).
2. Probe noise: the checkpoint ratio (policy / stock state-mean entropy) differs between two independent stock
   completion sets by SE <= 0.02, and the state-bootstrap SE of a ratio is <= 0.03.
3. Healthy runs (r3 to u102, r4 to u98, r5 to u102, r6 to u66) stay at ratio >= 0.85 at every checkpoint.
4. Run 2 (runaway length, collapse ~u45, stopped by hand at u54) falls below every healthy checkpoint by u40.
5. The mean log-prob of the stored stock tokens (a teacher-forced -KL(stock || policy) estimate, relative to stock)
   separates run 2 at least as well as entropy.

Falsified if no threshold separates run 2 at or before u50 from every healthy checkpoint with the stated margin; then
the Finding says so plainly and proposes no entropy trigger.

## Process

### 1. Probe (frozen, `freeze`)

- States: the run 4-6 training pool (`posttool-tasks-003` states.xml, TRAIN split, `training_mix` with repair .75 /
  relay .15 / blocked .10), stratified seeded sample (`select`, seed 20261002): **24 repair, 6 relay, 4 blocked**.
- Completions: stock (zero adapter), the training sampler (`rloo_posttool.Episodes`, capture on, `--compact 8,16`),
  envelope-002 default, T = 1, 4,096-token turns, up to 8 turns (calculator calls run through GameTerm's runner).
  **Two independent completion sets** (seeds 20261003, 20261004; one sampling call each, one episode per state).
  Every sampled token of every turn is scored (tool results and prompts are prefilled, not scored, as in training).
- Stored: `probe.xml` (states, seeds, tokens, stops, rewards, input hashes), `hidden.npz` (final-block inputs and
  sampler log-probs per token).

### 2. Metric (`score`)

- Per policy and set: per state, the mean over its scored tokens of the next-token entropy (-sum p log p of the
  tail's T = 1 distribution, the training entropy's formula) and of the stored token's log-prob; then the mean over
  states (each state weighs equally), overall and per category. **Primary statistic: R = H_policy / H_stock on the
  same set** (overall; per category reported). Secondary: dLP = mean log-prob policy - stock.
- Policies: stock; every raw checkpoint of r2 (010-050), r3 (010-100 + final u102), r4 (010-090, 098), r5 (010-100 +
  final u102), r6 (010-060, 066). r2/r3 trained on `posttool-tasks-002` states (repair/relay only); the probe is
  from 003, so for them it is partly off-distribution (reported, not corrected).

### 3. Noise

- Between-set: for each checkpoint, R on set A vs set B; SE_set = SD(R_A - R_B) / sqrt(2) over all checkpoints.
- Within-set: paired state bootstrap of R (resampling states within category, 20,000 draws), per checkpoint.
- sigma for the rule = the larger of the two at the healthy checkpoint nearest the threshold.

### 4. Rule selection (fixed now)

- Level rule: trip when R < X. X = (lowest healthy R) - 2 sigma, rounded down to 0.01: every healthy checkpoint is
  >= 2 sigma above X, P(false alarm per healthy checkpoint) <= Phi(-2) ~= 2.3% at the closest one (reported with the
  per-checkpoint sum over all healthy checkpoints). Run 2 is **caught** if a checkpoint <= u40 has R < X (at collapse
  if only u50).
- Step rule: trip when R drops by > Y between consecutive checkpoints (10 updates); Y = largest healthy drop +
  2 sigma*sqrt(2); same detection criterion.
- The same two rules on dLP. The rule reported as proposed is the one that catches run 2 earliest with the healthy
  margin intact; if none does, no rule is proposed.
- Decision set A is the primary; set B is the replicate the rule must also hold on.

### 5. Direct collapse signals from the saved records (no GPU)

For r2..r6, per block of 10 updates, from every `update-*/update.xml`: (a) **within-group distinct final answers**
(the numeric answer parsed from each episode's final content, `posttool.numeric_answer`; no answer = one bucket;
word-problem groups only), and (b) **repeated-thinking rate**: share of episodes whose final-turn reasoning tail
(600 chars, the record's only thinking text) repeats a word 12-gram; plus the logged `repeated_thinking_tokens` per
1k where the run logged it (r3+). Separation judged the same way: run 2 by u40 outside the healthy runs' range.

### 6. GPU

`gpu-run check` only: freeze (model load + 2 x 34 episodes, est. ~10 min), score (44 checkpoints + stock, forward
only, est. ~15 min).

## Log

- 2026-09-28 17:01-17:11 EDT `freeze` (`gpu-run check`, ~10 min incl. compile): set A 13,861 tokens, set B 18,226
  (34 episodes each; 26/27 correct; no capped turn). Tinygrad-arkey `exp` bbf307f83 (`<worktree>`).
- 17:11-17:15 `score` (~4 min, 4.4 s per policy for both sets): stock + 44 checkpoints. Stock tail reproduces the
  stored sampler log-probs exactly (max error 0.0). Total GPU ~14 min.
- Records pass (`<scratch>/probe/records.py`, no GPU): r5 `update-101` (u102) unreadable (the pre-fix
  control-character bug), so r5 blocks end at u101.
- The maintainer (during the run): keep an entropy alarm but fix it; acceptance bar = (1) zero healthy trips with
  the margin in probe-noise SE and P(false alarm | healthy run) <= 10%, (2) trips on run 2 no later than the replayed
  length/capped triggers (u32), (3) composition-free by construction; else report-only. Answered below.

## Finding

**A fixed-probe entropy ratio separates run 2 from every healthy checkpoint, with margin. Proposed rule: stop when
R = H_probe(policy) / H_probe(stock) < 0.96, evaluated at every raw checkpoint (every 10 updates). It meets all three
acceptance conditions.** Prediction 1 held (exact parity), 2 held (SE 0.003 << 0.02), 3 held (healthy min 0.971),
4 held (run 2 below every healthy checkpoint from u30), 5 failed (log-prob drift does not separate; below).

### Calibration table (overall R; set A, set B in brackets where it differs by > 0.005)

| update | r2 (bad) | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|
| 10 | 0.988 | 0.997 | 0.993 | 0.991 | 1.001 |
| 20 | 0.962 | 0.990 | 0.986 (0.993) | 0.999 | 1.006 |
| 30 | **0.936** (0.942) | 0.986 | 0.982 (0.992) | 0.999 | 1.001 (1.008) |
| 40 | **0.892** | 0.987 | 0.991 (1.001) | 0.996 | 0.997 (1.005) |
| 50 | **0.847** | 0.992 | 1.009 | 0.987 | 0.980 (0.989) |
| 60 | - | 0.998 | 1.007 | 0.990 | 0.971 (0.976) |
| 66 / 70 | - | 1.004 | 1.002 | 0.999 | 0.973 (u66) |
| 80 | - | 1.003 | 0.995 | 1.002 | - |
| 90 | - | 0.989 | 0.986 (0.992) | 0.991 (0.984) | - |
| 98-102 | - | 0.982 (0.988) | 0.990 (0.997) | 0.987 (0.979) | - |

Stock: H = 0.604 (set A) / 0.595 (set B). Per category, run 2 falls fastest in repair (u30 0.927, u40 0.878,
u50 0.829), then relay (0.948/0.906/0.864), blocked least (0.972/0.950/0.920); healthy runs stay within 0.968-1.022
in every category (r6 u60 repair 0.968 is the lowest). Full table: `<scratch>/probe/calib.out`.

### Noise

- Between completion sets: SD(R_A - R_B) = 0.0043 over 44 checkpoints -> **SE 0.003** per set (mean offset -0.003).
- State bootstrap within a set (paired, stratified): SE 0.001-0.007 per checkpoint (0.003 at r6 u60).
- sigma = 0.003. The rule's X = floor(0.971 - 2 x 0.003) = **0.96**, as fixed in P4.

### The rule against the acceptance bar

1. **Healthy runs: zero trips.** Closest: r6 u60 0.971, **3.7 SE** above the line (set B 0.976, 5.3 SE); r3 min
   0.980 (3.7 SE), r4 0.982 (3.3 SE, its bootstrap SE is 0.007), r5 0.979-0.987 (6.4 SE). P(false alarm) from the
   probe's own noise: 5e-4 at the closest checkpoint, 1e-3 summed over all 39 healthy checkpoints. That ignores
   seed-to-seed variation of the trajectory itself, which is larger: the four healthy runs' minimum R is 0.980 +- 0.007
   (A) / 0.984 +- 0.007 (B); a new healthy run's minimum falls below 0.96 with P ~ 4-5% (t, 3 df, probe noise added;
   ~0.3% under a normal). **Estimated P(false alarm | healthy run) ~ 5%** (<= 10%). Four runs is a thin base.
2. **Run 2 trips at u30** (0.936 / 0.942, ~8 SE below the line), before its replayed capped trigger (u32) and its
   collapse (~u45). With a check every 10 updates it cannot trip between checkpoints; u20 (0.962) is just above.
3. **Composition-free by construction:** the states, the tokens and the stock reference are frozen; nothing depends
   on a baseline window or on what a batch drew. What remains is the policy's tail on fixed inputs.

Replayed existing triggers on run 2 (playbook replay: run-3 reward, repair + relay rates): `DEFAULTS` u32 (capped
rate 0.056 > 0.05). One at a time: capped u32, KL u35, batch entropy (composition-adjusted) u38, finished length u41,
reward u51, length push never, skips never. The probe (u30) is the earliest single signal.

### Rejected alternatives

- **Step rule** (drop > Y between checkpoints): largest healthy drop 0.017 (r6 u40-50) -> Y = 0.026; run 2 drops
  0.026 at u20 and u30 (not beyond Y), 0.044 at u40. Trips at u40: later than the level rule.
- **Log-prob of the stored stock tokens (dLP)**: healthy runs drift down steadily (r4 u98 -0.059, r5 u102 -0.054 on
  set A) while run 2 is only -0.036 at u30 and -0.082 at u40; it trips no earlier than u40 and its between-set SE is
  3x larger (0.009). It measures distance travelled, not collapse.
- **Within-group distinct final answers** (records, 10-update blocks): run 2 1.60-1.98, healthy 1.30-2.11 (1.00 for
  r5's one-update tail): no separation at any point.
- **Repeated-thinking rate** (final-turn reasoning tail repeats a word 12-gram): healthy blocks 0.000-0.052 (r6 u61-66
  is the max), run 2 0.022/0.025/0.037/**0.062**/0.306/0.602. It separates only at u31-40 by 0.01 (inside noise) and
  clearly at u41-50, i.e. at the collapse. A good confirmation signal, not an early one. The logged
  `repeated_thinking_tokens` (r3+ only) has no run-2 value to compare.

### Caveats

- 34 states and ~14-18k tokens per set; blocked has 4 states (its per-category ratio is noisy, report-only).
- r2 and r3 trained on `posttool-tasks-002` states (repair/relay only); the probe is 003. Run 2 was still caught.
- Run 2 is the only bad run: the rule catches one known failure mode (runaway length with an inert KL). It is not
  shown to catch a collapse that keeps stock-token entropy (e.g. confident wrong answers).
- The margin on r6 is 0.011 in raw R; r6 was still falling at u60-66 (0.980 -> 0.971). A longer run of that seed could
  have crossed it; that would be a true drift of 3-4% below stock, which is what the line means.

### Proposed predeclarable rule (not wired in)

At every raw checkpoint (`--keep-every 10`), score the checkpoint on `probe-entropy-001` set A with
`probe_entropy score` (forward only, ~5 s on the loaded model): **stop if R < 0.96**; report set B, the per-category
ratios and dLP. Replaces the batch-entropy test (`entropy_ratio`), which stays report-only. Wiring it in (scoring
inside the loop from the live adapter, no reload) is a separate, predeclared change, replayed on these checkpoints.
