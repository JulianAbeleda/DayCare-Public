# Training Map

This note maps the major training families and when to use each.

Use it with Two-Track Strategy: the pure LLM route is
for understanding and baselines, while the model route is the goal.

## Summary Table

| Method | What changes | Main data | Best use | Main risk |
|---|---|---|---|---|
| Pretraining | all weights | massive unlabeled corpus | create base capability | expensive, data quality dominates |
| Continued pretraining | all weights or adapters | domain corpus | add domain distribution / terminology | catastrophic drift if data is narrow |
| SFT / instruction tuning | full model or adapters | prompt-response pairs | teach behavior format and task imitation | memorizes style; weak at preference tradeoffs |
| LoRA | low-rank adapters | same as SFT/DPO/RL data | cheap specialization | capacity bottleneck, target-module sensitivity |
| QLoRA | LoRA over frozen 4-bit base | same as LoRA | single/few GPU fine-tuning | quantization/training instability edge cases |
| DPO | full model or adapters | chosen/rejected pairs | align preferences without RL rollout | pair quality and reference model matter |
| RLHF/PPO | policy, reward model, maybe value model | demonstrations + preference labels | optimize subjective human preference | complex, unstable, reward hacking |
| RLVR/GRPO | policy | prompts + verifiers | math/code/schema/correctness tasks | sparse rewards, verifier overfitting |
| JEPA | representation/world model | unlabeled text/images/video/state | semantic latent prediction, world modeling | evaluation and collapse control are harder than SFT |
| Persistent agent | state/memory/policy loop | interaction traces + feedback + environment state | trainable companion or long-running assistant | can feel coherent while learning bad habits |

Data curation and evaluation sit underneath every row. They are not a separate
optimizer, but they decide whether each optimizer teaches useful behavior or
memorizes leakage. See [Data Curation And Evaluation](data-and-evaluation.md)
and the [Exhaustive Research Map](exhaustive-research-map.md).

Persistent-agent work is a separate system layer. See
model Lineage (artificial-life prior art) for the
architecture pattern: persistent state, drives, online learning, grounded
actions, and verifier-backed feedback.

## Core Distinction

Training methods differ along two axes:

```text
What is being learned?
  facts / distribution / behavior / preference / reasoning policy / world model

How is the feedback produced?
  next-token loss / demonstrations / comparisons / learned reward / verifier / latent prediction
```

For BoltBeam, the useful feedback is unusually verifiable:

- JSON schema validity
- candidate matches profile/search space
- route is reachable or blocked for a named reason
- predicted verdict matches ledger result
- proposed measurement command exists
- tinygrad candidate compiles and measures

That makes data splits unusually important. The same ledger can be training
data, validation data, and evaluation authority unless the split is explicit.

That makes BoltBeam a natural fit for SFT first, then RLVR later.

## Recommended Order For DayCare/BoltBeam Work

1. Build a trace dataset from BoltBeam decisions.
2. Train a LoRA SFT model to imitate:
   - bottleneck classification
   - next action selection
   - scope writing
   - refusal to promote unsupported claims
3. Add DPO pairs:
   - good scope vs speculative scope
   - measured conclusion vs guessed conclusion
   - scoped evidence vs row leakage
4. Add RLVR only for outputs with strict verifiers:
   - valid JSON
   - valid candidate id
   - all cited artifacts exist
   - no candidate outside search space
   - no promotion with missing rollback
5. Keep final promotion and roofline gates out of training data.

## Why Not Start With RLVR?

RLVR is powerful when the reward is correct and cheap. It is dangerous when the
reward is incomplete. For BoltBeam, some rewards are easy:

- schema pass/fail
- route reachability pass/fail
- exact ledger verdict match

Other rewards are expensive:

- actual compile success
- token correctness
- wall-clock speed

The model should learn the discipline through SFT first, then use RLVR for
verified refinement.
