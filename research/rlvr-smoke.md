# First DayCare RLVR engineering gate

> Published copy of a working record (2026-09-21), lightly edited for publication. `<runs>/` stands for the private
> run-output directory; the pilot scripts named here are not published. GameTerm is
> [gameterm.arkey.ai](https://gameterm.arkey.ai); the experiments used a beta build that is not public.

LoRA describes trainable parameters, not the optimization objective. The prior
Nemotron end-to-end experiment proved supervised completion loss through LoRA,
XML checkpoint, export and GameTerm. It did not prove policy-gradient learning.

This pilot adds real on-policy RLOO to DayCare's tinygrad training path. It is a
finite-action contextual bandit: for each question, four fixed native calculator
call sequences define the possible actions. The model scores each sequence with
summed conditional token log probabilities. The categorical policy is softmax
of those scores divided by temperature 2. This is a restricted policy, not
unrestricted language generation, tool discovery, or multi-turn GRPO. All 35
GameTerm schemas and captured system/ledger messages remain in the prompt.
Only calculator calls are permitted actions in this first engineering gate.

Three training questions and three held-out questions use independently checked
integer arithmetic; all share a task template. They establish a mechanical
check, not broad generalization. Sixteen actions are sampled with replacement
from the current policy, then GameTerm's actual calculate normalizer/executor
runs every sampled expression. Reward is one only when a successful receipt
matches the independently computed exact answer. Environment outputs and
rewards are detached. There are no supervised target updates or corrections.
Candidate actions are pre-authored; a correct candidate's index is never a
cross-entropy target. The verifier can score actions, but the optimizer only
receives sampled actions and scalar rewards.

For a group of K rewards, advantage i is reward i minus the mean of the other
K-1 rewards. Minimize negative mean advantage times sampled-action log
probability plus 0.01 times exact categorical KL(current || initial). Take one
Adam update per freshly sampled group, two rounds, three train tasks per round,
rank 4, alpha 8, last block only, learning rate 0.00005, seed 20260921. Clip
gradient norm to one. No critic, replay, PPO clipping, or best-of selection.
The sampler and differentiable policy use the same current tinygrad model;
there is no quantized rollout/trainer probability mismatch.

Gate: at least one mixed-reward group, finite nonzero gradient, changed adapter
weights, lossless XML reload, identical probabilities after loading through the
actual continuation loader, and separate before/after held-out action-policy
measurements. A zero-signal run is reported as such, not presented as learning.
No broad quality or general-intelligence claim follows from these six prompts.
All inputs/verifier hashes are frozen before updates. Runtime artifacts remain
outside Git under `<runs>/nemotron-rlvr-smoke-001`.

Authority: `daycare/nursery/rlvr.py` owns the RLOO baseline and policy objective;
`rlvr_smoke.py` (not published) composes the model, frozen task fixture and real verifier.
NativeToolChat owns tool syntax; existing feature/cache, LoRA and XML components
own model-specific representation and persistence. The XML method remains LoRA;
its provenance separately records the RL objective and verifier identity.

## Research grounding

- Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for
  Learning from Human Feedback in LLMs* (2024),
  <https://arxiv.org/abs/2402.14740>: basis for the leave-one-out policy-gradient
  estimator. Our finite action restriction is an engineering simplification,
  not a reproduction of their LLM training experiments.
- Shao et al., *DeepSeekMath* (2024), <https://arxiv.org/abs/2402.03300>:
  introduces GRPO. This pilot deliberately uses RLOO and must not be called GRPO.
- Xu and Wang, *Learning to Use Tools: Reinforcement Learning for Tool-Integrated
  Mathematical Reasoning* (2026), <https://arxiv.org/abs/2608.28447>: studies
  calculator interaction with verifiable final-answer rewards. Our single-action
  receipt reward is narrower than their complete-trajectory reward. This paper
  motivates a subsequent multi-turn experiment, not a claim that we implemented it.

Unit tests enumerate all two-sample groups and check that the expected RLOO
policy gradient equals the analytical reward gradient. Other tests check
zero-variance groups, action sampling, and the direction of reward-driven updates.

## Observed learning result

The learning/persistence gate passed on Nemotron 4B. The run took 348.1 seconds,
performed six optimizer updates and executed all 96 sampled calculator actions.
Ninety-five actions earned reward. One group contained both reward values and
produced a substantial nonzero reward gradient (norm 0.1380); the other groups
had no reward contrast and only tiny KL/numerical gradients. Adapter weights
changed. XML tensor roundtrip was exact, and resetting the adapter then invoking
the real continuation loader restored all six action distributions exactly.

| Restricted-policy measurement | Before | After |
| --- | ---: | ---: |
| Mean correct-action probability, 3 training questions | 98.1079% | 98.1263% |
| Mean correct-action probability, 3 held-out questions | 98.4669% | 98.4794% |
| Greedy correct choices, training | 3/3 | 3/3 |
| Greedy correct choices, held-out | 3/3 | 3/3 |

These are exact probabilities under the four-action restricted policy at
sampling temperature 2, not free-generation accuracy estimates. The tiny change
and near-ceiling baseline do not establish meaningful model improvement. The
next learning experiment needs harder predeclared training tasks with enough
reward variation. Unrestricted token-level rollouts and multi-turn calculator
receipt/final-answer credit assignment remain unproved.

The initial freeze attempt stopped before updates because its verifier tried
to parse the entire `expression = result` receipt as a number. The verifier now
parses the receipt value. Every action was executed before freezing, confirming
exactly one reward-positive action per task. The aborted freeze remains in a
separate runtime directory. No optimization occurred against the broken verifier.

The calculator fixture and all frozen source/input hashes still match after the
run. Runtime `train/run.json` contains every sampled action, policy probability,
reward, receipt, gradient norm and update. `analysis.json` contains all questions,
action menus and before/after distributions. `train/adapter.xml` stores the
LoRA weights; provenance identifies the objective as `finite-action-rloo` and
hashes the verifier. Its legacy adapter identifier field reads `native-tool-sft`;
that identifier is not the optimization objective. No supervised loss was used.

Validation: analytical policy-gradient tests plus full DayCare suite:
613 passed, 9 skipped. Repository size passes the hard budgets; durable product
code remains above the pre-existing 18,000-line soft review threshold.

## Next-gate constraint from the maintainer

The next RL experiment should target a behavior the small Nemotron demonstrably
struggles with, rather than this nearly solved action menu. Measure baseline
success and within-question reward variation on training-only diagnostic tasks
before deciding to optimize. Freeze separate held-out evaluation questions and
do not select training examples from their failures. A group that is always
correct or always wrong supplies no RLOO reward contrast. Distinguish inability
to generate an action from choosing poorly among supplied candidate actions;
this pilot only tested the latter. Do not silently relabel another finite-menu
experiment as free-form tool-use RL.

## Post-RL runtime compatibility

The RL checkpoint was merged into the verified original base package, exported
to F32 GGUF and quantized to Q4_K_M. GameTerm's actual shared headless harness
then ran two repetitions of the prior 40-question suite: 80 results, zero token
truncations, 74 final-answer events, and six expected unavailable-provider
approval outcomes. The server stopped afterwards. This is an export/runtime
compatibility result; these trials were not graded as evidence of improvement
or retention. The suite was separately hashed before RL updates.

The maintainer clarified that the next target is harder tasks Nemotron currently gets
wrong, not a new output syntax. A diagnostic script (not published) therefore
collects training-only baseline responses through the actual harness, four
samples per question with distinct seeds, before any next RL dataset is chosen.

## Harder-task diagnostic result

Sixteen unrestricted responses from the original Nemotron 4B were collected
through GameTerm: four task prompts, four server seeds, temperature 0.8, thinking
off, all 35 tools visible. These are training-only diagnostics, not held-out
confirmation, and no further training was performed on them.

| Task | Correct final answer | Actual calculator use |
| --- | ---: | ---: |
| Wage rate and fractional hours | 4/4 | 2/4 |
| Mixed coin denominations and spending | 3/4 | 2/4 |
| Moving speed excluding a stationary break | 3/4 | 1/4 |
| Dilution by adding pure water | 4/4 | 1/4 |

The speed failure subtracted a 45-minute break incorrectly: 4 hours became
3.75 instead of 3.25, followed by an incorrect speed. The coin failure printed
`calculate { ... }` as ordinary text and never invoked the tool or answered.
Both questions also had successful responses, demonstrating reward contrast
in this small sample. The wage question's seed-53 answer gave the correct final
210 dollars but an incorrect hourly rate of 29 rather than 28 in its explanation.
A final-answer-only reward would miss that inconsistency; final correctness and
faithfulness of intermediate claims are separate properties.

These are plausible next RL targets, not proof of an effective curriculum.
The finite-action RLOO backend cannot yet optimize these unrestricted multi-turn
responses. Next implementation must preserve sampled-token log probabilities,
mask environment receipt tokens from the policy loss, execute real tool calls,
and use a reviewed verifier before updating. Do not train on the new held-out
set or use observed evaluation failures to choose training questions.

An initial diagnostic request was rejected by GameTerm's parity check because
`seed` was added to the request body although its harness does not emit that
field. Seeds were moved to llama-server configuration, preserving native request
parity. The aborted run had zero result trials and was retained separately.
All successful diagnostic servers stopped after their runs. Full responses,
tool events and per-answer review are in runtime
`nemotron-rlvr-difficulty-001/review.json`.
