# Post-tool RL: the record, in order

Can a small local model use a tool's result well? This series trains **Nemotron 3 Nano 4B** (thinking on) with
RLOO on the turn *after* a tool call inside [GameTerm](https://github.com/JulianAbeleda/gameterm), a terminal app
with a built-in assistant. The tool is a calculator; the questions are word problems and Countdown puzzles; later
runs add refused file/terminal calls ("blocked"). Sampling and training both run on one stack,
[tinygrad-arkey](https://github.com/JulianAbeleda/tinygrad-arkey) (`exp` branch). The experiments used a beta build
of GameTerm that is not public.

Every run follows the [RL run playbook](../docs/rl-run-playbook.md): hypothesis and gates written and committed
before the first counted update, a bit-exact save/reload check before the long run, predeclared stop triggers, a
verdict scored exactly as declared, and no rescue runs. Failures are kept as failures; they are most of the record.

## Scenarios (how results are reported)

- **relay**: the tool result is the answer; map it back to the question.
- **repair**: the calculator rejected the call; fix it or answer another way.
- **miss**: Countdown; the result is not the target yet; keep searching.
- **blank / empty**: the model gave no answer after a tool result (often a turn cut at the 4,096-token cap).
- **blocked**: GameTerm refused a file/terminal call by policy; the model should say so, not work around it or
  invent content.

## The runs

| Run | Hypothesis (short) | Outcome | Why |
|---|---|---|---|
| [0: thinking RFT](thinking-rft-posttool.md) | rejection-sampling fine-tuning that includes post-tool turns fixes empty answers after a calculator call | stopped before any update, by decision | replaced by on-policy RLOO |
| [1](rloo-posttool-calculator.md) | RLOO on the post-tool turn lifts held-out success without more blanks | **not adopted**: G1 +7.0 [+1.4, +12.8] pass, **G2 fail** (blanks 60 -> 95 of 712) | repair up (0.70 -> 0.86), miss down; capped turns doubled |
| [2](rloo-posttool-calculator-r2.md) | filter capped episodes + soft length penalty (DAPO-style) keeps run 1's gain without the truncation | **failed**: collapsed after ~u40, stopped off protocol at u54, no gate run | [audit](rloo-posttool-audit-20260927.md): filtering removed the only length brake; KL/entropy terms were ~1e-5 of the policy gradient (a bug) |
| [3](rloo-posttool-calculator-r3.md) | run 1 on repair + relay, with a length term and a working KL | **not adopted**: G1 +12.9 [+7.9, +18.2] pass, **G2 fail** (53 blanks, limit 50) | blank change -7 [-21, +7]: neither clearly better nor worse; threshold set at the prediction |
| [4](rloo-posttool-calculator-r4.md) | from stock, fresh word problems + blocked counter-examples, powered gates | **stopped by the entropy trigger at u98**, not adopted | information-only gates strong (repair +15.8, blanks 97 -> 20, blocked +16.6) but G3 retention -2 on selection items |
| [5](rloo-posttool-calculator-r5.md) | run 4's recipe, new seed, blocked rule (c), composition-adjusted trigger | **every gate passed except G3 numeric** (net real -1); **adopted by owner exception** | see "Current state" |
| [6](rloo-posttool-calculator-r6.md) | run 5 replicated on a new seed, with a powered numeric retention gate (G7) | **stopped by the entropy trigger at u66**, not adopted | drift diagnosis mixed, leaning benign; the trigger sits near its limit for this recipe on both seeds |

## Side investigations (inference only, no training)

- [Budget forcing](posttool-budget-forcing.md): force `</think>` at the cap? **H falsified**: 0 of 42 stock blanks
  and 5 of 82 adapter blanks become correct; it mostly turns a blank into a wrong answer.
- [A longer turn](posttool-longer-turn.md): raise the cap to 8,192? **H falsified**: about half of cut turns finish,
  but only 14-23% of those are right. Decision: keep 4,096. Lesson: a blank gate needs a companion
  "no new wrong answers" condition.
- [An honest out](posttool-honest-out.md): offer a `report_unsolved` tool at the budget? **H1 falsified** (the model
  takes the out in a visible form about 1 in 5 times); H2 and H3 hold.
- [Selection transfer](posttool-selection-transfer.md): run 4's G3 selection losses were a grader mismatch plus
  temperature-0 fragility, not a training failure; adds an out-of-distribution blocked probe and a
  composition-adjusted entropy trigger.
- [G3 numeric diagnosis](posttool-g3-numeric-diagnosis.md): run 5's numeric loss is fragile (two of three items are
  coin flips at T=1), with one real reading shift; at T=1 on all 68 numeric items run 5 is level with stock.
- [Probe entropy](probe-entropy-trigger.md): an entropy drift check on a fixed probe set, which batch composition
  cannot move; R < 0.96 separates run 2 from every healthy checkpoint (proposed, not yet used in a run).
- Loop speed: [iteration speed](posttool-loop-iteration-speed.md) (53.2 -> 36.7 s per update from prefix reuse and
  follow-up refill) and [slot refill](rloo-slot-refill.md) (1.32x faster sampling, parity exact).
- Literature: [controlling reasoning length in RL](reports/rl-length-control-literature-20260927.md) (DAPO,
  Dr. GRPO, SimpleTIR, L1 and others), written after run 1.

## Current state

One adapter is adopted: **run 5, by owner exception**, recorded as an exception and not a pass. Its held-out
results against stock:

| Scenario | Stock -> adapter | Difference, 95% CI |
|---|---|---|
| repair (calculator rejected the call) | 76.3% -> 91.8% | **+15.5 [+11.2, +20.0]** |
| relay | 94.4% -> 95.1% | +0.8 [-0.6, +2.1] |
| blanks, relay family (2,620 episodes) | 97 -> 16 | -3.1 pts [-4.1, -2.1] |
| blocked (refused call), held-out | 27.1% -> 39.4% | **+12.3 [+7.7, +16.9]** |
| blocked, out-of-distribution probe | 38.1% -> 48.7% | **+10.6 [+5.6, +15.4]** |

Why an exception: G3 numeric retention failed as declared (net real -1, one greedy sample per item). The
[diagnosis](posttool-g3-numeric-diagnosis.md) then found that result fragile: on all 68 numeric items at T=1 the
adapter is level with stock (+1.5 [-1.1, +4.4] under the audited rule). Known costs carried with the adapter: the
calculator call rate on numeric items fell 5.5 points [-9.4, -1.8], one real reading shift (`gsm8k-3005`), and a
blocked gain below run 4's. Run 6, the replication, was stopped by its entropy trigger, so run 5 has not been
reproduced on a second seed.

The training code for this loop (`rloo_tinygrad`, `rloo_posttool`, the stop triggers and gates) is not yet in this
repository; it will be published separately.

## Reading notes

These are working records, lightly edited for publication. Paths such as `<runs>/`, `<scratch>/` and `<worktree>`
stand for private run-output directories, unpublished analysis scripts and local checkouts. "DayCare (private
commit)" marks a commit in the private development repository. Module names such as `daycare/nursery/rloo_posttool.py`
refer to code that is not yet public. tinygrad-arkey revisions are public on its `exp` branch.
