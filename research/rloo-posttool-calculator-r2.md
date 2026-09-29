# RLOO on the post-tool calculator turn, run 2: truncation handled as the literature recommends

Status: **predeclared** 2026-09-27, before any run-2 update. The maintainer approved the design
(with citations required). This is a DayCare LoRA RLVR feature for Nemotron 3 Nano 4B
on the GameTerm harness. Run 1: [rloo-posttool-calculator.md](rloo-posttool-calculator.md).

## Hypothesis

Run 1 raised held-out success on the states where the calculator result does not settle the question (G1 +7.0
points, CI [+1.4, +12.8]; repair 0.697 -> 0.861). It failed G2: blank answers after a tool result rose from 60 to
95 of 712, because turns that ran into the 4,096-token cap doubled. In run 1 a truncated turn scored 0, the same
as a wrong answer, so the update had no pressure to finish inside the budget, and some capped episodes were long,
sound reasoning that the gradient punished anyway. DAPO names exactly this: scoring truncated samples like wrong
ones "may introduce noise ... as a sound reasoning process can be penalized solely due to its excessive length"
(Yu et al., arXiv:2503.14476, Sec. 3.4).

**H:** training on repair states (plus a small relay share), with truncated episodes removed from the update
(Overlong Filtering) and a soft penalty for finishing near the cap (Soft Overlong Punishment), keeps run 1's
repair gain without its truncation and blank-answer regression.

Predictions (all against the same stock arms as run 1):
- repair held-out score at least **+10 points** over stock (stock 0.697);
- blank answers after a tool result in G1 + G1b episodes **at most stock's 60 of 712**;
- relay held-out score flat (stock 0.953; G1b's interval includes zero or is above it).

The hypothesis is falsified by any failed gate below (the same gates as run 1).

## Process

### What changes from run 1, and why (each cited)

1. **Training mix: repair 90%, relay 10%; no miss or empty in training.** Run 1's training rollouts were 49.5%
   relay (at ceiling, ~0.95 reward, little signal), 30.6% repair, 16.7% miss, 3.2% empty. Miss and empty are
   Countdown search, a separate skill whose training reward fell in run 1 (0.691 -> 0.570) while turns got longer.
   The relay share keeps the easy relay behavior anchored. Implementation: `posttool_tasks.training_mix(shares)`,
   seeded: 125 repair + 14 relay train states (139). The held-out exam is unchanged and still contains miss and
   empty.
2. **Overlong Filtering: an episode whose last turn hit the 4,096-token cap is removed from the update** (from
   the policy loss and from its group's leave-one-out baseline); it still counts in every reported rate. A group
   left with fewer than two finished episodes is dropped. Sources: DAPO's Overlong Filtering (Yu et al.,
   arXiv:2503.14476, Sec. 3.4, which reports it "significantly stabilizes training"), and SimpleTIR
   (arXiv:2509.02479), which excludes a whole multi-turn trajectory containing a degenerate ("void") turn rather
   than scoring it. Implementation: `rloo_posttool.masked(policy='overlong_filter')`.
3. **Soft Overlong Punishment: training reward = outcome + penalty on the episode's longest turn,** zero until
   `cap - buffer`, then `((cap - buffer) - length) / buffer`, reaching -1 at the cap (DAPO Eq. 13). Buffer =
   **819 tokens**: DAPO used a 4,096-token buffer inside a 20,480-token maximum, a ratio of 0.2; 0.2 x 4,096 =
   819, so the penalty starts at 3,277 tokens. A turn that reaches the cap is filtered (item 2), so the penalty in
   effect grades finished turns in the last 20% of the budget, as in DAPO, where the two are used together
   (their ablation: AIME24 avg@32 36 -> 41 when the soft penalty is added to filtering). Implementation:
   `posttool.soft_overlong_penalty`, `posttool.shaped_reward(policy='soft_overlong')`. Evaluation always scores
   the plain outcome.

Checked, not changed:
- **Loss normalization.** Dr. GRPO (Liu et al., arXiv:2503.20783) shows GRPO/PPO implementations divide each
  sample's loss by its length |o|, which under-penalizes long wrong answers; Arora and Zanette (arXiv:2502.04463)
  report the same effect in an RLOO implementation. Ours does not: the policy-gradient term sums
  `logp * weight * advantage / trajectories` over tokens with no per-sample 1/|o| (`rloo_tinygrad.TailSteps._grad`).
  Only the KL and entropy terms are token means, over the whole batch.
- **The entropy bonus (1e-3) is a suspect to watch, not changed.** DAPO links entropy growth with length growth
  (arXiv:2503.14476, Sec. 3.3-3.4). Run 1's entropy fell (0.54 -> 0.50), so it did not blow up; run 2 logs it per
  update.

### Fixed settings (identical to run 1 unless listed above)

- **Starts from stock:** a fresh zero-initialized LoRA on the base model, not run 1's adapter, so the comparison
  isolates the three changes. Run 1 and run 2 are each compared against the same stock arms.
- Nemotron 3 Nano 4B BF16, thinking on; LoRA rank 32, alpha 64 on the final MLP block.
- Sequence RLOO, group 8, 4 states per update, **102 updates** (as run 1; about three passes over the 139
  training states), Adam lr 2e-4, KL 1e-3, entropy 1e-3, clip 1.0, TIS cap 2, 0.1-nat parity stop, temperature
  1.0, 4,096 tokens per turn, 8-turn cap, seed 20260924, `posttool-tasks-002` states.
- Stack: tinygrad-arkey exp 4322ae9e2 (shared-prefix reuse, follow-up refill), DayCare from a detached worktree
  at the commit recorded in the run.
- Command: `rloo_posttool train --categories repair relay --mix repair=0.9,relay=0.1 --mask overlong_filter
  --reward soft_overlong --buffer 819 --updates 102 --protocol rloo-posttool-calculator-r2.md`. The run record
  stores `mix`, `mask` and `reward`.
- Logged per update: outcome reward (all episodes), trained (shaped, kept) reward, trained episodes, mixed
  groups, parity, KL, entropy, rho, stage times, and the capped-turn, blank-after-tool and other-tool rates.

### Evaluation (unchanged from run 1, so run 1's stock arms are reused)

The same held-out exam, episodes, seeds (adapter 20260926) and outcome scoring as run 1. Stock arms on file:
`posttool-g1-stock-001`, `posttool-g1b-stock-001`, `posttool-g3-stock-001`.

### Checklist and gates

- [x] P1. This file committed before any run-2 update; design approved by the maintainer.
- [x] P2. Output-path check: unit tests for the mask and the penalty (`tests/test_posttool.py`,
  `tests/test_rloo_posttool.py`), then one real-model update with the run-2 policies, export and `verify`
  bit-exact.
- [x] P3. Train 102 updates; stop on a parity violation, a non-finite value or a FloatingPointError. (Stopped at
  54 by the maintainer's decision, off protocol; see Log.)
- [ ] G1. Held-out miss + repair + empty (93 states x 4), stock vs adapter, task-clustered bootstrap (50,000
  resamples, rng 20260923); pass if the lower bound of the 95% interval is above zero.
- [ ] G1b. Held-out relay (85 x 4); fail if the interval's upper bound is below zero.
- [ ] G2. Blank answers after a tool result in G1 + G1b episodes: adapter strictly fewer than stock (60).
- [ ] G3. Retention (only if G1, G1b and G2 pass): the 132-item suite through HT-020, adapter vs stock, thinking
  on, temperature 0; zero plain-answer losses, no net real loss after hand audit (clarification audited by hand).
- [ ] Stop rule: a failed gate ends the experiment; no rescue runs.

Once a recipe passes, the deployable adapter will be built in stages (a curriculum) from the proven recipes; that
is a later, separately predeclared step.

## Log

- P2 (2026-09-27): unit tests pass (penalty flat to 3,277 then linear to -1 at 4,096; run-1 policies reproduce the
  direct update bit for bit; truncated episodes contribute no loss token and still count in the capped rate).
  Real-model check `posttool-rloo-r2-check-001` (DayCare (private commit), not counted; 512-token turns, so the penalty is
  not meaningful there): the record stores mix/mask/reward; 139 training states (125 repair, 14 relay); of 32
  episodes 14 were truncated and filtered, 17 trained (one group dropped for having a single finished episode);
  parity exact on the 8,637 trained tokens; export + `verify` bit-exact (4,194,304/4,194,304).
- P3 started: `posttool-rloo-r2-001`, `gpu-run time`, from a zero-initialized adapter.
- **Stopped off protocol by the maintainer's decision** (2026-09-27, about 13:18 EDT). The operator sent
  SIGINT during the 55th update, after 54 completed updates (`update-054/` is partial). The reason was reward
  collapse and runaway length. This is not a predeclared stop trigger: parity stayed exact and no value was
  non-finite. The last saved weights are the raw checkpoint at update 50; nothing was exported. G1, G1b, G2 and G3
  were not evaluated. The adapter is not adopted: run 2 failed on training evidence.

## Finding

**Failed; not adopted.** Training collapsed after about update 40 and the maintainer stopped it at update 54 of 102. No
gate was evaluated. The hypothesis is not supported: the recipe did not keep run 1's repair gain without the
truncation regression. On the training states it produced far more truncation than run 1.

### What happened (training data, per update block)

| Updates | Outcome reward | Capped turns | Blank final answers | Other-tool calls | Trained episodes (of 32) | Entropy | KL at block end |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-20 | 0.778 | 2.4% | 9.2% | 4.4% | 30.8 | 0.59 | 6.2e-3 |
| 21-40 | 0.823 | 4.8% | 10.0% | 1.9% | 29.8 | 0.52 | 5.3e-2 |
| 41-46 | 0.724 | 13.3% | 21.4% | 0.5% | 26.2 | 0.42 | 9.5e-2 |
| 47-54 | 0.43 | 38.6% (27-49% per update) | 54.5% | 0% | 16.7 | 0.33 | 4.7e-2 |

In GameTerm terms: by the end, about half of the repair and relay episodes would have shown the user a blank reply
after a long think. Those are the same scenarios where run 1 and stock mostly answered. Finished episodes grew
steadily: 676 -> 780 -> 945 -> 1,076 tokens mean (updates 1-20, 21-40, 41-47, 48-54). Run 1's finished episodes
stayed at about 450-600 tokens over the same updates.

### Diagnosis: survivorship length bias (a design flaw, not a bug)

Checked on CPU from the update records and checkpoints; the scratch analysis is in
`<scratch>/{ext,an,an2,an3}.py` and `data.json`.

1. **Direction, not step size.** The adapter's parameter displacement from initialization was the same in both
   runs: RMS 6.7e-4, 1.2e-3 and 1.5-1.6e-3 at updates 10, 30 and 50. Run 2 moved as far as run 1, in a different
   direction.
2. **The overlong filter removed run 1's implicit length brake.** In run 1 a capped episode scored 0 and stayed
   in its group. With every token carrying a negative advantage, about 2,000 negative token-advantage units per
   update pushed against running long. Run 2 dropped capped episodes from both the loss and the leave-one-out
   baseline (`rloo_posttool.masked`), so nothing pushed back.
3. **Among the episodes that survived, length paid.** Episode length correlated positively with advantage
   (+0.17 over updates 1-10): long careful answers that finished were often correct, and the policy learned
   "longer". The ones that grew too long were then filtered out rather than penalized.
4. **The soft penalty barely touched anything.** Soft Overlong Punishment reaches only finished episodes longer
   than 3,277 tokens, which were 0.5-1% of finished episodes in updates 1-40. DAPO sized its buffer for
   ~16k-token reasoning (arXiv:2503.14476). On short-answer post-tool turns of a 4B model the buffer sits where
   almost nothing finishes, so it cannot steer the length distribution before it tips into truncation.
5. **KL tracked the trained length** (r = 0.64). It is measured on finished tokens only, so it understates the
   drift.

The code matches the predeclared design: filter, then penalize inside the buffer. The mechanism, not its
implementation, is what failed here. DAPO's paired filter and penalty relied on many long, near-cap samples to
make the penalty bite. That condition does not hold for these turns.

### Side finding: the KL term is inert in both runs

The policy term sums log-probability x advantage over every token (`rloo_tinygrad.TailSteps._grad`), while the
KL term is a token mean scaled by 1e-3. Its gradient is about 1e-5 of the policy gradient, so the "KL to
stock 1e-3" in both predeclarations was effectively off. This did not cause the collapse (run 1 had the same
term), but a next run needs a KL that actually constrains.

### Run 1's blank answers, split (G2 episodes, stock 60 vs adapter 95)

| | Truncated at the cap | Voluntarily empty | Reasoning never closed |
|---|---:|---:|---:|
| stock | 42 | 5 | 13 |
| run-1 adapter | 82 | 1 | 12 |

The truncated tails are mostly self-verification loops: repeating "422.21?", re-deriving 39 + 19, "13.2105?". The
model usually already had the answer and did not stop.

### Direction for a run 3 (not predeclared)

A candidate recipe:
- **An honest-give-up reward, ordered:** correct > an honest "not sure, best answer X" > wrong > silent
  truncation. The abstain reward stays small (about 0.25 on a -1/+1 scale, per arXiv:2601.20126) and shrinks on
  relay. The ordering follows TruthRL (arXiv:2509.25760).
- **A repetition penalty** for the verification loops (Demystifying Long Chain-of-Thought Reasoning,
  arXiv:2502.03373).
- **Keep capped episodes in the group and baseline, with a penalty,** instead of filtering them. Budget-aware
  alternatives are AnytimeReasoner (arXiv:2505.13438) and Elastic Reasoning (arXiv:2505.05315); SimpleTIR's
  exclusion (arXiv:2509.02479) and DAPO's filter (arXiv:2503.14476) fit long-reasoning regimes, not this one.
- **A KL term scaled to the policy gradient**, so it actually constrains.

## Sources

- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", arXiv:2503.14476 (Overlong
  Filtering and Soft Overlong Punishment, Sec. 3.4, Eq. 13; token-level loss and entropy, Sec. 3.3).
- Xue et al., "SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning",
  arXiv:2509.02479 (void-turn trajectory filtering).
- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO), arXiv:2503.20783.
- Arora and Zanette, "Training Language Models to Reason Efficiently", arXiv:2502.04463.
- Research summary: `<runs>/reports/RL length control reasoning models.md`.
- Finding only (not used in the design): "Rewarding Intellectual Humility: Learning When Not To Answer", arXiv:2601.20126 (abstention reward scale); TruthRL, arXiv:2509.25760;
  Demystifying Long Chain-of-Thought Reasoning, arXiv:2502.03373; AnytimeReasoner, arXiv:2505.13438; Elastic
  Reasoning, arXiv:2505.05315.
