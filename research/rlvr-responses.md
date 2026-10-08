# Native-response RLVR pilot

> Published copy of a working record (2026-09-22), lightly edited for publication. `<runs>/` stands for the private
> run-output directory; the pilot scripts named here are not published. GameTerm is
> [gameterm.arkey.ai](https://gameterm.arkey.ai); the experiments used a beta build that is not public.

The maintainer authorized moving from the finite-action RLVR smoke to harder tasks
where original small Nemotron sometimes fails: moving-time conversion and mixed
coin values. This pilot samples actual tokens without an answer menu, through
the real GameTerm beta headless harness. DayCare is both the policy sampler and
the trainer. The baseline is the original BF16 4B with zero-initialized LoRA,
not the previous finite-action or calculator adapter.

## Frozen procedure

Two previously diagnosed **training-only** questions, four independently sampled
trajectories per question, one optimizer update per question group at most.
Freeze four new math holdouts and twenty previously evaluated ordinary controls
before updates. Never update from evaluation responses. Rank 4, alpha 8, last
MLP block only, Adam 1e-5, gradient norm capped at one, seed 20260922.

Training sampling temperature 0.8, top-p 1, repetition penalty 1, presence penalty
0; preserve captured token biases and thinking=false. These neutral sampling
controls differ from prior production sampling and make the sampling distribution
and differentiable policy explicit. Greedy before/after evaluation uses the same
harness and verifier. There are 128 generated tokens per assistant response,
256 model-generated tokens and three assistant requests maximum per trajectory.
Truncation, missing final completion and unrelated tool proposals receive zero.
All 35 tool schemas and GameTerm's actual prompt and ledger are present. OS and
external-provider execution remains blocked by the study fixtures.

The task asks for a brief explanation and a final `Answer:` line in the requested
units. Reward is one for a single final marker with the exact numerical answer,
otherwise zero. Optional requested units and Markdown emphasis are accepted.
The marker and verifier are declared before baseline inference. Reward does not
require a calculator call; genuine tool execution is counted independently.
Formatting failures must be separated from semantic failures in the report.
Final-answer rewards do not verify every intermediate claim.

RLOO advantage is each trajectory's reward minus the mean reward of its peers.
The objective is negative advantage times the **sum** of its generated-token log
probabilities, averaged over the group. Accumulate chunk gradients, then take one
optimizer step. No SFT corrections, restricted action menu, best-of selection,
replay, critic, importance-ratio clipping or KL term. Omitting KL is an explicit
small-pilot choice, not a recommended large-run configuration. Zero-variance
groups are skipped, not converted into imitation updates.

## Boundaries and validation

`NemotronHModel.advance` updates immutable frozen-prefix caches. Tests compare
incremental token stepping with full suffix evaluation and verify the shared
prefix is not mutated. `NativeRollout` (not published) applies the same temperature and logit
bias in sampling and learning. Sampled features, tokens and actual log
probabilities are persisted for every assistant request. Only sampled assistant
tokens enter the loss; user prompts and environment receipts are context only.
Before each update, replay all selected-token log probabilities and fail if the
maximum absolute difference exceeds 0.005 nats. This numeric tolerance accounts
for vector-versus-batched floating-point kernels; it does not justify a different
sampling policy or quantized rollout model.

`rlvr_native_harness.py` (not published) serves the current DayCare policy over a loopback
OpenAI-compatible streaming endpoint. The existing GameTerm Rust driver owns the
conversation, normalization, calculator execution and tool-result continuation.
The endpoint parses the selected Nemotron template's native tool tags into
provider tool calls; malformed calls remain prose. This parser has contract
coverage but is not asserted equivalent to all permissive llama.cpp parsing.
Backend exceptions abort the batch; they are not used as negative model rewards.

A synthetic provider contract already traversed the actual Rust harness:
calculator call → real receipt `52/(4-45/60) = 16` → final answer. This is a
bridge test, not a model-quality result. Mathematical tests cover the RLOO
baseline, analytical expected gradient, zero-variance behavior and equivalent
chunked sequence gradients. Verifier tests reject intermediate gold numbers,
conflicting final markers, wrong units and truncated answers.

Gate: real calculator execution during training, at least one mixed-reward
group, finite changed parameters, rollout/training logprob agreement, exact
lossless XML checkpoint reload, and recorded before/after held-out responses.
If reward contrast is absent, report no demonstrated learning. This small run
cannot establish broad generalization, statistical efficacy or general
intelligence retention. Keep every failed attempt and frozen artifact outside
Git under `nemotron-rlvr-responses-001`; do not retune against the holdout.

## Research grounding

- Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for
  Learning from Human Feedback in LLMs*, <https://arxiv.org/abs/2402.14740>:
  leave-one-out policy-gradient estimator; our two-group/no-KL configuration is
  an engineering pilot, not a reproduction of their training experiment.
- Xu and Wang, *Learning to Use Tools: Reinforcement Learning for Tool-Integrated
  Mathematical Reasoning*, <https://arxiv.org/abs/2608.28447>: motivates live tool
  interaction and verifiable outcomes with environment observations excluded
  from the policy loss. Their models/tasks/training scale differ from this pilot.

## Decoder setup checks

The first three setup runs stopped before any optimizer update. They exposed
an overly narrow unit alias check, repeated long-prefix tokenization, and eager
native decoding around three seconds per token. Their inputs and partial
responses remain in separately named aborted-run folders; none were used for
training or to select new holdout questions.

A fixed-capacity attention/JIT experiment passed the small FP32 hybrid test
but failed the real BF16 model's probability check twice (maximum errors
0.054636 and 0.059025 nats). Both attempts stopped before a completed baseline
response or any update. They remain in `aborted-decode-parity` and
`aborted-decode-parity-copy-only`. The fixed-capacity attention implementation
was removed; the 0.005-nat tolerance was not relaxed.

The decoder now compiles only the frozen recurrent and tokenwise blocks.
Attention calls the existing `NemotronHBlock.cached` implementation with its
original dynamic cache lengths and masks. Canonical cache copies alone use
`NOOPT=1` to avoid a pinned-backend misaligned-load failure. A hybrid-model test
compares repeated token steps against eager suffix evaluation, including a
second conversation with reset caches.

Before collecting any response, four greedy continuation steps compare full
vocabulary probabilities with the eager decoder. If this optional optimization
exceeds 0.005 nats, discard its preflight outputs and use the authoritative eager
decoder for the entire experiment. Preflight never consumes the training random
stream and never contributes tokens or gradients. The trainable final MLP stays
outside JIT. Sampling/training logprob replay remains a hard update gate.

An update-path review caught an optimizer call outside tinygrad's required
training-mode context, before any real update was attempted. `step_policy` now
owns finite-gradient validation, norm clipping and the optimizer's train-mode
scope. A real tinygrad LoRA/Adam test verifies increased probability for a
rewarded token, reduced probability for its failed peer, changed adapter weights
and byte-identical base weights, with inference mode restored afterward.

## Executed update

Run `nemotron-rlvr-responses-001`, training revision (a private commit), used the original
BF16 Nemotron 4B in DayCare's CUDA backend. The final block-JIT preflight had
zero full-vocabulary logprob difference on all four steps. The final source
passed 620 tests with 9 skips; a separate CUDA test covers decoder cache resets.

The speed group scored `[1, 1, 1, 1]`; its zero-contrast update was skipped.
The coin group scored `[0, 1, 0, 1]`; one real RLOO update changed the adapter.
The failed answers were 255 cents (omitted eight quarters) and 3295 cents
(omitted quarters and introduced further arithmetic/unit errors), versus the
correct 455 cents. These were mathematical failures, not formatting-only
failures. Pre-clipping gradient norm was 5.936617, capped at one before Adam.
Maximum sampled-logprob replay error was 0.0000195503 nats. Reloaded XML tensor arrays exactly matched the saved arrays.

None of the eight sampled training trajectories called a tool. Therefore the
predeclared **live-calculator training gate does not pass**, even though native
response sampling, reward contrast, gradient updating and checkpointing worked.
This is not evidence of learned multi-turn tool use. Review every sampled answer
in the run's `training-review.json`; retain its original policy requests, raw
responses, feature/token/logprob arrays and Rust harness event logs.

## Before/after result

| Measure | Before | After |
| --- | ---: | ---: |
| Semantically correct held-out math | 3/4 | 3/4 |
| Strict final-line reward | 2/4 | 2/4 |
| Correct ordinary-answer controls | 20/20 | 20/20 |
| Actual calculator executions in evaluation | 0 | 0 |

All 24 final responses and generated token sequences were identical. Frozen
features also matched exactly; selected-token log probabilities changed by at
most 0.000611365 nats. Thus the update changed the policy probabilities without
changing these greedy answers. The held-out failure omitted three quarters and
answered 275 instead of 350 cents, both before and after training.

The 310-cent answer was mathematically correct but put `Answer: 310` inline;
it failed the stricter formatting contract in both arms. This is reported
separately, rather than called a loss of mathematical ability. Evaluation used
an assistant review against frozen answers with arm labels withheld; it was not
an independent human review. Training failures were checked separately and were
both mathematically incorrect.

For both evaluation families, observed paired correctness-difference SD and SE
were zero; exact McNemar p=1. The paired bootstrap interval also collapses to
zero because every observed pair is unchanged. This small, one-seed sample
**cannot establish equivalence**, broad intelligence retention, or training-seed
variance. There is no measured improvement and no observed score regression.

The final uninterrupted run took 1,629 seconds, excluding setup investigations.
Stop at this pilot: the full predeclared gate failed on absent calculator use,
and accuracy did not improve. A next small study should diagnose harder
training-only tasks under this exact policy and elicit real tool trajectories
before investing in larger training. Do not turn these holdout failures into
training examples while continuing to call this set held out.

Local evidence (runtime data remain outside Git):

- Every evaluation question and response: `<runs>/nemotron-rlvr-responses-001/responses.md`
- Scores, paired statistics, and training record: `<runs>/nemotron-rlvr-responses-001/analysis.json`
- Every sampled training answer and error review: `<runs>/nemotron-rlvr-responses-001/training-review.json`
- Frozen protocol: `<runs>/nemotron-rlvr-responses-001/protocol.json`
- Hardware and runtime provenance: `<runs>/nemotron-rlvr-responses-001/environment.json`

The final report is recreated by the reporter script (not published) from the run directory. The saved blind
review must exist; the reporter refuses unreviewed answers.
