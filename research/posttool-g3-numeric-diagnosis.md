# Post-tool G3 numeric loss: robust or a coin flip, and is it the calculator?

Status: **Finding written 2026-09-28** (H + P committed first, a private commit). Diagnosis only: no training, no adoption. The maintainer approved
the three-step proposal from [run 5](rloo-posttool-calculator-r5.md)'s Finding.
This is a DayCare LoRA RLVR feature for Nemotron 3 Nano 4B on the GameTerm harness.

Run 5 failed G3 on numeric net real -1: losses `gsm8k-3023` ("Dora got 1 pack", answer 2; run 4 lost it too) and
`gsm8k-3005` ($560, answer 960), win `gsm8k-3018` (160, stock 244). G3 samples each item once at temperature 0.
Numeric items with a `calculate` call: stock 63/68, run 4 48/68, run 5 56/68.

## Hypothesis

H1 (robust). At the exam temperature (1.0) the run-5 adapter fails `gsm8k-3023` and `gsm8k-3005` clearly more often
than stock: pooled over the two items (40 samples per arm), the Newcombe 95% CI of (adapter - stock) failure rate lies
above 0. **Fragile** if the CI covers 0; then the G3 verdict was decided by one greedy path, as the selection items were
([selection transfer](posttool-selection-transfer.md) P2).

H2 (mechanism). The adapter calls the calculator less often than stock on these items, and its failures sit mostly in
episodes with no `calculate` call. Falsified if the adapter's call rate is not lower, or failures are spread evenly
across called and uncalled episodes.

`gsm8k-3018` is reported the same way (the win), as a control of the same size.

## Process

- `scripts/selection_transfer_sample.py` (the G3 path exactly: `calculator_harness_pilot.run`, llama-server +
  GameTerm's live calculator study, G3 envelope `countdown-recipe-prep-003/envelope.json`, thinking on, 4,096 tokens) at
  temperature 1.0, llama-server `--seed` = replicate 1..20 (a restart per replicate), `--items gsm8k-3023 gsm8k-3005
  gsm8k-3018`. Arms: stock, run-5 adapter (`posttool-rloo-r5-001/adapter.gguf`). 20 samples per item per arm.
  `gpu-run check`. Outputs `<runs>/g3num-diag-001/`.
- Scoring: `thinking_retention_gate.judge` (last number = label, no non-calculator tool). Every failure is read by
  hand; a failure whose stated final answer equals the label is an **artifact** and counts as a pass (reported). Per
  episode: did it call `calculate`, and with what expression.
- Reading, set now: H1 as above. H2: call rate adapter vs stock (Newcombe CI) and failures split by called / not
  called.

## Log

- **2026-09-28 13:40-14:05 EDT** (`g3num-diag-001/`, `gpu-run check`, 20 replicates x 2 arms, no run errors). Per item
  (pass by `judge` / `calculate` called / failures without a call / with a call): stock 3023 8/20, 1, 11, 1; 3005 19/20,
  16, 0, 1; 3018 17/20, 15, 0, 3. Run 5: 3023 8/20, 0, 12, 0; 3005 13/20, 15, 1, 6; 3018 15/20, 14, 3, 2. Hand read of
  failures: one run-5 3023 failure is an artifact ("Dora got two packs": right, the last-number judge finds no digit), so
  run 5 3023 is 9/20; the rest are real (3023: "1 pack", "0.5"; 3005: "$560", one "$400 together"). Rows:
  `<scratch>/r5/g3num_rows.json`.

## Finding

**H1 fragile (pooled), with one real loss inside it. H2 falsified: the calculator is not the cause here.**

- `gsm8k-3023` is a coin flip for both models: stock 8/20, run 5 9/20 (after the artifact). Stock's temperature-0 pass,
  which made it a G3 "loss" in runs 4 and 5, was the lucky path. Both arms mostly answer "1 pack" without a tool.
- `gsm8k-3005` is a real loss: failures stock 1/20, run 5 7/20, difference CI [+0.05, +0.52]. It is a **reading
  shift**, not tool skipping: 6 of the 7 failures call the calculator, on the wrong expression (`400 + 400 * (2/5)`:
  "2/5 times more" read as "plus 2/5"; stock mostly writes `400 * (1 + 2/5)` then adds 400).
- `gsm8k-3018`, run 5's G3 "win", is noise: stock 17/20, run 5 15/20.
- Pooled 3023 + 3005 (the declared H1 test): failures stock 13/40, run 5 19/40 (18/40 after the artifact), Newcombe CI
  [-0.06, +0.34]: covers 0, so **fragile** as predeclared. The G3 numeric verdict (net -1) was decided by single greedy
  samples on two of its three items.
- Calculator use on these items is the same (stock 17/40 episodes, run 5 15/40 on 3023 + 3005). The T=0 G3 drop in
  call rate (63 -> 56 of 68 numeric items) is real in that one sample but does not explain these losses.

**What this changes.** (1) The proposed mechanism (length term -> skip the calculator) is not supported by these items;
it stays a hypothesis for the whole numeric set, untested. (2) The one robust drift is on problem *reading*, which the
post-tool training never scores (states begin after the model's own call, so a wrong expression is never
corrected). (3) G3's numeric check cannot gate at one sample per item: the next step is a powered numeric retention
measure on all 68 items at temperature 1 (stock vs run 5), which answers both whether run 5 regressed math overall and
whether call rate fell. Run 5's verdict stands as declared (not adopted); this is information for run 6.

## P2: powered numeric retention (H + P, committed before measuring)

H3 (overall regression). On all 68 G3 numeric items (math 56, oracle 8, smoke 4) at temperature 1, the run-5 adapter's
pass rate is lower than stock's: item-clustered bootstrap 95% CI of (adapter - stock) entirely below 0. **No
regression** if the CI covers 0 or lies above it. H4 (calculator): the adapter's `calculate` call rate over the same
episodes is lower than stock's (CI entirely below 0).

Process: the P1 path unchanged (`selection_transfer_sample.py`, G3 envelope, thinking on, 4,096 tokens, T=1.0, server
seed = replicate), `--items` = the 68 numeric source ids, 8 replicates per arm (544 episodes each), arms stock and
run 5, `gpu-run check`, outputs `g3num-diag-002/`. Scoring: `judge`, paired per item, bootstrap over items (50,000),
and the same for call rate. Artifacts: failures whose reply states the label in words or before a later number are
counted by a hand-checked rule on the stored replies (reported with and without). The per-item stock variance sizes a
non-inferiority margin for a numeric retention gate (P(pass | unchanged) >= 0.9), proposed for run 6.

### P2 Log and Finding

- Timing note (review): P2's H + P was committed at 14:00:25, after the stock arm had started (13:59); its first
  output was written at 14:04, so no P2 data was seen before the commit.
- **2026-09-28 13:59-15:00 EDT** (`g3num-diag-002/`, 8 replicates x 2 arms, 544 episodes each, no run errors).
  Item-clustered bootstrap (50,000, seed 20260928), paired per item:

  | Measure (68 numeric items, T=1) | Stock | Run 5 | Diff, 95% CI |
  |---|---:|---:|---|
  | pass, `judge` as is | 85.7% | 88.2% | +2.6 [-0.9, +6.2] |
  | pass, artifact-tolerant (label stated in the last sentence, digits or words) | 92.1% | 92.8% | +0.7 [-1.8, +3.7] |
  | `calculate` call rate | 84.2% | 78.7% | **-5.5 [-9.4, -1.8]** |

  Largest item drops (artifact-tolerant, of 8): gsm8k-3018 7 -> 5, gsm8k-3029 8 -> 6, four clarify items 8 -> 7 or
  7 -> 6; largest gains gsm8k-3022 1 -> 6, gsm8k-3023 2 -> 3.

- **H3 not supported: no overall math regression.** Run 5 is level with stock (+0.7 points, CI [-1.8, +3.7]; +2.6 under
  the unmodified judge). G3's numeric fail at one greedy sample per item was noise plus one real item (`gsm8k-3005`, P1).
- **H4 supported: the adapter calls the calculator less** (-5.5 points, CI entirely below 0) **without losing
  accuracy** at this sample size. The calculator-skipping drift is real; its cost is not measurable here.
- **Gate sizing for run 6.** Stock's per-item variance gives SE 1.36 points at k=8 (0.96 at k=16) for two arms of an
  unchanged model; non-inferiority with P(pass | unchanged) >= 0.9 needs M >= 4.4 points at k=8 (3.1 at k=16). Proposed
  powered numeric retention gate: 68 items x 8 at T=1, artifact-tolerant scoring, paired CI lower bound >= -5 points;
  call rate reported (or gated) beside it.
