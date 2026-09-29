# RLOO on the post-tool turn, run 6: run 5 replicated with a new seed and a powered numeric retention gate

Status: **stopped by the entropy trigger at u66, not adopted; information-only gate arms cancelled after the maintainer adopted run 5 by owner exception (16:55 EDT; the G2 arm was ~40% done, relay only, level with stock)**; PREDECLARED 2026-09-28 15:30 EDT (under the maintainer's go-ahead, quoted below, after the adversarial review;
text below the Log is frozen except the Log, Finding and checklist ticks). The maintainer, 2026-09-28: hold run 5 (not
adopted) and go ahead with run 6. This is a DayCare LoRA RLVR feature for Nemotron 3 Nano
4B on the GameTerm harness. Earlier: [run 5](rloo-posttool-calculator-r5.md) (all gates pass except G3 numeric, net
real -1, at one greedy sample per item), [G3 numeric diagnosis](posttool-g3-numeric-diagnosis.md) (at T=1 on all 68
numeric items run 5 is level with stock, +0.7 [-1.8, +3.7]; calculator call rate -5.5 [-9.4, -1.8]).
Playbook: [rl-run-playbook.md](../docs/rl-run-playbook.md). Scripts outside Git: `<scratch>/r6/` (`g7.py`).

## Why a run 6 and not a re-score of run 5

The playbook forbids rescue runs: run 5's verdict stands, and its adapter is not re-scored under a new gate. Run 6 is
a new predeclared run: the same recipe from stock with a **new seed** (a replication that must pass on its own), and
G3's numeric check replaced by a powered gate sized from measured stock variance (the diagnosis). Nothing else in the
training changes, so any difference from run 5 is seed variation, and a pass is evidence the recipe, not one lucky
seed, works.

**The gate change is the maintainer's decision (stop-spinning rule).** Run 5's G3 was the second consecutive G3 failure, and this
is the second G3 sub-check retired after a failure (selection became the OOD probe G6 after run 4; numeric becomes G7
after run 5). The maintainer approved the swap explicitly, 2026-09-28: the three-step proposal in run 5's Finding, step 3
(replace G3's one-sample numeric check with a powered one); then, after the diagnosis
showed no overall regression, approved run 6. The margin rule
(M >= 3.24 SE for P(pass | unchanged) >= 0.9) was fixed in the diagnosis P2 before it was measured, the same
rule as G5/G6.

## Hypothesis

Mechanism (as run 5): RLOO on the model's own post-tool turns, reward = a correct non-empty final answer, raises repair
and cuts blanks where the trained states sit, while matched KL and the stop triggers bound drift elsewhere; prior art
as run 5 (RLOO arXiv:2402.14740, Kimi k1.5 length term arXiv:2501.12599, ProRL arXiv:2505.24864).

From stock, run 5's recipe with training seed 20261001 passes every gate below, and trips no trigger. Predictions (run
5 in brackets): G1 repair >= +10 [+15.5]; G1b relay no drop [+0.8]; G2a blanks fall [-3.1 pts]; G2b wrong not up [-2.9];
G5 long no worse [+10.8]; G5 blocked no worse [+12.3]; G6 OOD probe no worse [+10.6]; G7 numeric no worse [+0.7];
plain retention 40/40; audit within margin [0/40 vs 1/40]. **Calculator call rate** on G7's episodes: predicted to fall
about as in run 5 (-5.5 [-9.4, -1.8]); reported, not gated. Declared reading: a drop whose CI lies entirely below 0 is
recorded as a real loss of tool use for the next design (the post-tool goal is tool competence), even if G7 passes.
Falsified by any failed gate or a trigger trip.

## Process

### 1. Fixed setup (run 5's, verbatim unless listed)

- Training: run 5's `TRAIN` flags (`<runs>/posttool-r6-run.sh`, diff vs `posttool-r5-run.sh`: seed,
  roots, protocol name, the G7 arm; stock arms that are reused are not re-run; no checkpoint-050 arms):
  `posttool-tasks-003`, mix repair .75 / relay .15 / blocked .1, envelope-002 default, graded reward, length term 1.0,
  repetition n 40 / -0.05, matched KL 0.03, `--compact 8,16`, 102 updates, blocked rule (c) grader, triggers
  `rl_triggers.DEFAULTS` (composition-adjusted entropy). **Only change: `--train-seed 20261001`** (run 5: 20260930).
- Code: DayCare at the PREDECLARED commit in a pinned worktree `<worktree>` (the scorers `g7.py` and
  `gates6.py` import from it; the scoring commit is recorded in the Log); tinygrad-arkey `exp`
  bbf307f83 (`<worktree>`). Code changes since run 5's pinned commit: `record_xml` stores XML-illegal
  characters losslessly (a private commit, records only), the sampler's `--items` / arm entries (scripts only). Neither touches
  sampling, grading or the update.
- Outputs: `<runs>/posttool-rloo-r6-001/` (P2: `posttool-rloo-r6-check-001/`).
- P2 before P3: one update, export, `verify` bit-exact in a fresh process. On a trigger trip: stopped, not adopted,
  diagnosis first, gates as information only (as run 5).

### 2. Gates (vs stock; all adapter arms seed 20260928 unless listed, `gpu-run time|check`)

Stock arms reused where envelope, states, seed, flags, grader and code path are identical to the adapter arm (all from
run 5 or its diagnosis): G2 / G1 / G1b (`posttool-r4-g2-stock-001`), G1c (`posttool-r4-g1c-stock-001`), G5 long
(`posttool-r4-g5l-stock-001`), G5 blocked (`posttool-r5-g5b-stock-001`, rule (c)), G6 probe (`posttool-r5-g6-stock-001`),
G3 plain (`posttool-r4-g3-stock-{a,b}`), G7 (`g3num-diag-002/stock`). Scored by run 5's `gates5.py` (validated on run
4) with the adapter paths swapped, and `<scratch>/r6/g7.py`.

| Gate | States x samples | Pass rule |
|---|---|---|
| G1 word-problem repair | 46 x 20 | CI lower bound of (adapter - stock) > 0 |
| G1b relay | 85 x 20 | CI upper bound >= 0 |
| G2a relay-family blanks | 131 x 20 | paired CI entirely below 0 |
| G2b relay-family wrong answers | 131 x 20 | CI upper bound <= +2.5 points |
| G5 long-number | 18 x 20 | CI lower bound >= -8 points |
| G5 blocked (rule c) | 31 x 20 | CI lower bound >= -8 points |
| G6 OOD selection probe | 78 x 8 | CI lower bound >= -9 points |
| **G7 numeric retention (new)** | 68 x 8, T=1 | CI lower bound >= **-5** points |
| G3 plain retention | 40 x 1, T=0 | zero plain-answer losses (two-stock counting, hand audit) |
| G5 audit | 40 + 40 replies, blind | adapter - stock fabricated rate <= +15 points |

- **G7** (replaces G3's numeric families as a gate): the 68 G3 numeric items (math 56, oracle 8, smoke 4) through the
  G3 path (`selection_transfer_sample.py`: llama-server + GameTerm's live calculator study, G3 envelope, thinking on,
  4,096 tokens) at temperature 1.0, llama-server seed = replicate 1..8, adapter `posttool-rloo-r6-001/adapter.gguf`;
  stock arm `g3num-diag-002/stock` (same path, seeds 1..8). Scoring (`g7.py`, sha256 d92c630d, imports the pinned
  worktree): an episode passes if `thinking_retention_gate.judge` passes it, or (tolerant) it completed, was not
  truncated, called no tool but `calculate`, and the last sentence or line of its reply states the label (digits, or a
  number word zero..twenty). **Every tolerant-only pass, both arms, goes to a blind hand audit** (shuffled, arm hidden, a
  LLM reviewer marks whether the reply's final asserted answer equals the label; the review found tolerant passes where
  the label is only an intermediate value); a pass marked false is overruled. Item-clustered paired bootstrap, 50,000
  resamples. Per-item drops of 3 or more out of 8, and `gsm8k-3005` always, are reported. Items: the 68 source ids in
  `<scratch>/r5/numeric_ids.txt` (sha256 1be7bae6); the GameTerm beta build (as the stock arm). **Sizing:** stock's per-item variance gives SE 1.36 points for two arms of an unchanged
  model at k=8 (item-bootstrap SD 1.37 agrees); P(pass | unchanged) >= 0.9 needs M >= 4.4; **M = 5** (P = Phi(5/1.36 - 1.96) = 0.96). Caveat: 44 of 68 items are
  8/8 on stock, so the SE rests on 24 items; with Jeffreys-like shrinkage of the per-item rates P(pass | unchanged)
  falls to ~0.84, and a true -3 point drop fails ~68% of the time. Validation (newline split, before the audit): `g7.py`
  on run 5's diagnosis arms gives +1.1 [-1.5, +4.0]; the audit procedure is run on those arms before P3 (Log). The **calculator call rate** over the same episodes is
  reported with its CI (run 5: -5.5 [-9.4, -1.8]); not gated.
- **G3** still runs (132 items, T=0) for plain retention (gated) and numeric / selection (reported, rule-(c) judge for
  selection).
- G1c Countdown families: reported only. Side effects as run 5.
- Adoption: every gate passes. A failed gate ends the experiment; no rescue run. If G7 fails, the post-tool line stops
  and the maintainer gets the diagnosis (stop-spinning).

### 3. GPU time (estimate)

P2 ~10 min; P3 ~90 min (run 5: 86); G2 adapter ~35 min; G5 blocked adapter ~15; G5 long ~13; G6 adapter ~11; G1c ~6;
G3 adapter ~5; G7 adapter 8 x ~4 min ~32. Total ~3.5 h.

### Checklist

- [x] Adversarial review, blockers fixed (Log). - [x] PREDECLARED, committed, pushed. - [ ] Pinned worktree at that commit.
- [x] P2 bit-exact. - [x] P3 (stopped by the entropy trigger at u66). - [ ] Gates (information only). - [ ] Finding.

## Log

- **2026-09-28 review (fresh reviewer): NO-GO until fixed.** Blockers: (1) the sampler's `r6` arm was uncommitted
  (committed with this predeclaration); (2) the tolerant G7 rule passes answers where the label is only an intermediate
  value (fixed: newline split + a blind hand audit of every tolerant-only pass). Should-fix, done: The maintainer's approval of
  the gate swap recorded, calculator-rate prediction and reading, per-item drops reported, scorers import the pinned
  worktree, input hashes, mechanism and prior art, power caveat, diagnosis timing note. Nothing required a design change.
- **Audit dry run on run 5's diagnosis arms** (`<scratch>/r6/val.{blind,key,marks}.json`): 58 tolerant-only passes
  (both arms), blind LLM auditor marked 56 true, 2 false (both stock: "328 ounces" with 160 only a subtotal; "Dora gets 1
  pack"). With the audit applied: +1.5 [-1.1, +4.4] (would pass); call rate -5.5 [-9.4, -1.8].
- **2026-09-28 15:31-15:41 EDT P2** (`posttool-rloo-r6-check-001`): one update, reward 0.812, parity exact, KL 0; seed
  20261001 recorded; export + `verify` bit-exact (4,194,304 / 4,194,304). P3 update 1 reproduced it (loss -12.86996).
- **P3 STOPPED by the entropy trigger at update 66** (15:41-16:27 EDT): "entropy 0.5116 < 0.521 (mean of the last 10
  stepped updates)", composition-adjusted (`DEFAULTS`). Adapter exported and `verify` bit-exact. Verdict: **stopped, not
  adopted.** Gate arms run afterwards as information only.
- **Drift diagnosis from the saved records** (`<scratch>/r6/diag6.md`, before any new run): mixed, leaning benign.
  (1) Real within-category drop: repair-only batches 0.58-0.65 over u12-45 -> 0.437 in u57-66 (run 5 at the same
  updates 0.530); residual vs the baseline levels trends -0.019 per 10 updates (run 5 -0.007); steady from ~u51, not
  one outlier; length explains ~-0.01 of it. (2) No collapse: distinct final answers per group 1.58 -> 1.87,
  word-problem accuracy 0.90 -> 0.89, capped 0, other-tool ~0.02; mild homogenisation (within-group tail Jaccard 0.222
  -> 0.243, bare `<answer>N</answer>` finals 78% -> 90%); one repetition spike at u62. (3) The u61-66 states were not
  easy (all-correct groups 1.33 per update, run 5 2.17; every group mixed). (4) KL ~0.0085 (run 5 ~30% lower; limit
  0.025); length push negative in 14 of 16 updates of u51-66. (5) Baseline: u1-10 held a 0.12 blocked share (run 5
  0.00), lifting the fitted levels; with run 5's levels run 6 would not trip (worst ratio 0.894). Run 5 itself reached
  0.888 at u89 (limit 0.85): the trigger is borderline for this recipe on both seeds. Record note: run 5's
  `update-102/update.xml` does not parse (the pre-fix control-character bug).

## Finding
