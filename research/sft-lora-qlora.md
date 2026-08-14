# SFT, LoRA, And QLoRA

## Supervised Fine-Tuning

SFT trains the model to imitate desired outputs from prompt-response examples.
Instruction tuning is the common LLM version: the input is an instruction or
conversation, and the target is the desired assistant answer.

Useful references:

- Wei et al., "Finetuned Language Models Are Zero-Shot Learners" introduced
  FLAN-style instruction tuning and showed zero-shot gains:
  <https://arxiv.org/abs/2109.01652>
- Instruction tuning survey:
  <https://arxiv.org/html/2308.10792>
- Hugging Face TRL `SFTTrainer` official docs:
  <https://huggingface.co/docs/trl/en/sft_trainer>

### SFT Pros

- Simple.
- Stable.
- Best first step for teaching format and workflow.
- Works well with LoRA.
- Easy to evaluate with held-out examples.

### SFT Cons

- Imitates data, including bad habits.
- Does not directly optimize preferences or rewards.
- Can overfit style while failing hidden correctness.
- Needs high-quality examples.

## LoRA

LoRA freezes the base model and trains low-rank matrices inserted into selected
layers. It reduces trainable parameters and memory cost.

Primary paper:

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models":
  <https://arxiv.org/abs/2106.09685>

Official tooling:

- Hugging Face PEFT LoRA docs:
  <https://huggingface.co/docs/peft/package_reference/lora>
- PEFT overview:
  <https://huggingface.co/docs/peft/en/index>

### LoRA Pros

- Cheap relative to full fine-tuning.
- Easy to store multiple adapters.
- Good default for SFT/DPO/RLVR experiments.
- Lower risk of destroying the base model.
- No need to duplicate full model weights for every experiment.

### LoRA Cons

- Limited capacity.
- Sensitive to rank, alpha, target modules, and learning rate.
- May underperform full fine-tuning on deep domain shifts.
- Merging/unmerging adapters adds deployment complexity.

## QLoRA

QLoRA backpropagates through a frozen quantized base model into LoRA adapters.
It made large-model fine-tuning feasible on smaller hardware.

Primary paper:

- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs":
  <https://arxiv.org/abs/2305.14314>

### QLoRA Pros

- Much lower memory footprint.
- Practical for single-GPU or small-cluster experiments.
- Strong baseline for adapter training.
- Good match for local/open-source iteration.

### QLoRA Cons

- More moving parts: quantization, paged optimizers, adapter configs.
- Can be slower or less stable depending on kernels/hardware.
- Quantization artifacts may matter for some tasks.
- Not a substitute for high-quality data.

## Practical Recommendation

For a one-stop shop:

```text
TRL + PEFT
```

Use:

- `SFTTrainer` for imitation training.
- PEFT LoRA for adapters.
- DPO/GRPO later from the same ecosystem.

For DayCare/BoltBeam:

```text
Phase 1: LoRA SFT
  input: profile + evidence + current question
  target: correct audit conclusion / scope / next action

Phase 2: LoRA DPO
  chosen: disciplined evidence-backed answer
  rejected: speculative or over-promoting answer

Phase 3: LoRA GRPO/RLVR
  reward: schema/checker/ledger correctness
```

