# Open-Source Tooling

## One-Stop Shop Recommendation

Use:

```text
TRL + PEFT
```

Why:

- TRL supports SFT, DPO, GRPO, reward modeling, and more.
- PEFT supports LoRA and related parameter-efficient methods.
- Both integrate with Hugging Face Transformers and Datasets.
- This gives one stack for the first useful DayCare training loop.

Official references:

- TRL docs:
  <https://huggingface.co/docs/trl/en/index>
- TRL GitHub:
  <https://github.com/huggingface/trl>
- PEFT docs:
  <https://huggingface.co/docs/peft/en/index>
- PEFT LoRA docs:
  <https://huggingface.co/docs/peft/package_reference/lora>

## Other Tooling

| Tool | Best at | Why not first |
|---|---|---|
| TRL + PEFT | flexible one-stop SFT/DPO/GRPO/LoRA | needs more manual code than config tools |
| Axolotl | config-driven SFT/LoRA/DPO/GRPO | less direct if you want custom BoltBeam verifiers |
| LLaMA-Factory | easy UI/config fine-tuning | abstraction may get in the way of custom research loops |
| OpenRLHF | scalable RLHF/RLVR with Ray/vLLM | heavier operational stack |
| verl | serious distributed RL rollout/training | more infrastructure than day-one needs |

References:

- Axolotl GitHub:
  <https://github.com/axolotl-ai-cloud/axolotl>
- LLaMA-Factory docs:
  <https://llamafactory.readthedocs.io/en/latest/>
- OpenRLHF GitHub:
  <https://github.com/OpenRLHF/OpenRLHF>
- OpenRLHF docs:
  <https://openrlhf.readthedocs.io/>

## Practical Stack For DayCare

```text
Dataset:
  BoltBeam traces -> JSONL

Training:
  TRL SFTTrainer + PEFT LoRA

Preference:
  TRL DPOTrainer

Verifier RL:
  TRL GRPOTrainer + custom reward functions

Deployment:
  adapter checkpoint first
```

## When To Move Beyond TRL

Move to `verl` or `OpenRLHF` only when:

- rollouts dominate runtime
- you need distributed inference workers
- you need Ray/vLLM orchestration
- TRL becomes the bottleneck, not the model/data

Until then, one stack is simpler.

