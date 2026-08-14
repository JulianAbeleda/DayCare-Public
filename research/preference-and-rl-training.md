# Preference And RL Training

## RLHF / PPO

RLHF usually means:

```text
SFT model
  -> collect preference comparisons
  -> train reward model
  -> optimize policy with PPO or related RL method
```

Canonical paper:

- Ouyang et al., "Training language models to follow instructions with human
  feedback" (InstructGPT):
  <https://arxiv.org/abs/2203.02155>

### RLHF Pros

- Can optimize behavior humans prefer, not just imitate examples.
- Can improve helpfulness and reduce undesirable behavior.
- Handles subjective preference better than strict verifiers.

### RLHF Cons

- Complex: policy, reward model, reference model, rollout loop.
- Expensive.
- Reward models can be gamed.
- Human preference data is noisy and costly.
- PPO-style training is sensitive to hyperparameters.

## DPO

DPO directly trains on chosen/rejected pairs without training an explicit reward
model or running an RL rollout loop.

Primary paper:

- Rafailov et al., "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model":
  <https://arxiv.org/abs/2305.18290>

Official tooling:

- TRL `DPOTrainer`:
  <https://huggingface.co/docs/trl/en/dpo_trainer>

### DPO Pros

- Much simpler than PPO/RLHF.
- Stable and practical.
- Uses preference pairs directly.
- Good second step after SFT.

### DPO Cons

- Needs high-quality chosen/rejected pairs.
- Less interactive than RL rollout methods.
- Can overfit preference data style.
- Does not automatically discover behavior beyond the pair distribution.

## RLVR

RLVR means reinforcement learning with verifiable rewards. Instead of a learned
reward model or human preference at every step, the system uses programmatic
checks: math answer, code test, compiler result, schema validity, route
correctness, etc.

Important references:

- Cobbe et al., "Training Verifiers to Solve Math Word Problems":
  <https://arxiv.org/abs/2110.14168>
- DeepSeekMath introduced GRPO and used it for mathematical reasoning:
  <https://arxiv.org/abs/2402.03300>
- DeepSeek-R1 reports reasoning gains via RL:
  <https://arxiv.org/abs/2501.12948>
- RLVR survey/list:
  <https://github.com/opendilab/awesome-RLVR>

### RLVR Pros

- Rewards can be objective and cheap.
- Strong fit for code, math, schemas, compilers, and route-search systems.
- Less dependent on subjective human labels.
- Excellent match for BoltBeam-style gates.

### RLVR Cons

- Only works where verification is real.
- Sparse rewards can make learning inefficient.
- Verifiers can be incomplete or exploitable.
- The model may learn to satisfy the checker instead of the true goal.

## GRPO

GRPO is a policy optimization method popularized by DeepSeekMath. It estimates a
relative baseline from groups of sampled outputs, avoiding a separate critic in
the original formulation.

References:

- DeepSeekMath / GRPO:
  <https://arxiv.org/abs/2402.03300>
- TRL `GRPOTrainer`:
  <https://huggingface.co/docs/trl/en/grpo_trainer>

### GRPO Pros

- Practical for verifier-based tasks.
- Avoids some PPO infrastructure.
- Fits group rollouts where multiple answers can be scored.

### GRPO Cons

- Still needs rollout infrastructure.
- Reward design is the core difficulty.
- Can be noisy or inefficient without enough samples.

## Recommended Ladder

For DayCare/BoltBeam:

```text
SFT/LoRA first
  -> teach the audit format and basic decisions

DPO second
  -> prefer disciplined, cited, scoped answers over bad ones

RLVR/GRPO third
  -> use verifiers for exact properties
```

Do not let RLVR promote routes directly. It should train a proposal model. The
actual promotion still belongs to BoltBeam evidence and tinygrad measurement.

