# Exhaustive Research Map

This note is the coverage checklist for DayCare. The goal is not to memorize every paper. The goal is to know which
training method answers which kind of problem, what evidence supports it, and what failure mode to watch before using
it for BoltBeam-style models.

Read this through the two-track split in Two-Track Strategy: pure LLM work builds
understanding and baselines, while model work is the target architecture for persistent trainable agents.

## 1. Pretraining

Pretraining learns the base distribution with next-token prediction over a large corpus. It is the highest-leverage
stage because most factual knowledge, syntax, code patterns, and broad representations enter the model here.

Core references:

- Kaplan et al., "Scaling Laws for Neural Language Models": <https://arxiv.org/abs/2001.08361>
- Hoffmann et al., "Training Compute-Optimal Large Language Models": <https://arxiv.org/abs/2203.15556>
- The Pile: <https://arxiv.org/abs/2101.00027>
- Dolma: <https://arxiv.org/abs/2402.00159>
- DataComp-LM: <https://arxiv.org/abs/2406.11794>

Practical rule: use pretraining only when you need new base capability or a broad new data distribution. Do not use it
to teach output format, refusal style, or workflow discipline.

## 2. Continued Pretraining

Continued pretraining keeps the language-model objective but changes the data mix. It is the realistic version of
pretraining for domain adaptation.

Core references:

- "Reuse, Don't Retrain": <https://arxiv.org/abs/2407.07263>
- "Balancing Continuous Pre-Training and Instruction Fine-Tuning": <https://arxiv.org/html/2410.10739v1>
- "Domain-Adaptive Continued Pre-Training of Small Language Models": <https://arxiv.org/abs/2504.09687>
- "Continual Pre-Training is (not) What You Need in Domain Adaption": <https://arxiv.org/html/2504.13603v1>
- DeepSeekMath, which combines continued pretraining with GRPO: <https://arxiv.org/abs/2402.03300>

Practical rule: use continued pretraining when the model lacks durable domain representations: compiler traces,
roofline reports, kernel logs, route ledgers, or measurement language. Re-run instruction tuning afterward if the base
was already instruction-tuned, because continued pretraining can degrade instruction behavior.

Before continued pretraining, run model archaeology: inspect existing candidate models, probe their DayCare/BoltBeam
behavior, and check whether LoRA/SFT can close the gap more cheaply than changing base representations.

## 3. Supervised Fine-Tuning / Instruction Tuning

SFT teaches behavior through prompt-response examples. It is the right tool for style, structure, task imitation,
scope-writing, and response protocol.

Core references:

- FLAN / instruction tuning: <https://arxiv.org/abs/2109.01652>
- Instruction tuning survey: <https://arxiv.org/html/2308.10792>
- Self-Instruct: <https://arxiv.org/abs/2212.10560>
- Alpaca: <https://crfm.stanford.edu/2023/03/13/alpaca.html>
- LIMA: <https://arxiv.org/abs/2305.11206>

Practical rule: start BoltBeam training with SFT over high-quality audit traces. Use small, curated examples before
large noisy data. The model should learn to classify evidence, write scoped plans, and avoid unsupported promotion.

## 4. LoRA / QLoRA / PEFT

PEFT methods specialize a model cheaply by training a small number of parameters. LoRA is the default first adapter
method; QLoRA makes larger models trainable on smaller hardware by quantizing the frozen base.

Core references:

- LoRA: <https://arxiv.org/abs/2106.09685>
- QLoRA: <https://arxiv.org/abs/2305.14314>
- AdaLoRA: <https://arxiv.org/abs/2303.10512>
- DoRA: <https://arxiv.org/abs/2402.09353>
- Hugging Face PEFT: <https://huggingface.co/docs/peft/en/index>

Practical rule: use LoRA/QLoRA for the first DayCare/BoltBeam experiments. Full fine-tuning is not justified until the
data and evaluation loop prove value.

## 5. Preference Optimization

Preference methods train from comparisons or desirable/undesirable outputs. They are useful when "good" cannot be
fully captured by a single target answer but can be ranked.

Core references:

- InstructGPT / RLHF: <https://arxiv.org/abs/2203.02155>
- DPO: <https://arxiv.org/abs/2305.18290>
- KTO: <https://arxiv.org/abs/2402.01306>
- ORPO: <https://arxiv.org/abs/2403.07691>
- TRL DPOTrainer: <https://huggingface.co/docs/trl/en/dpo_trainer>

Practical rule: use DPO/KTO/ORPO after SFT when you can construct clean pairs: measured conclusion over guessed
conclusion, scoped plan over speculative plan, route-bound verdict over hand-wavy speed claim.

## 6. RLHF / PPO

RLHF uses demonstrations, preference data, a reward model, and policy optimization. It is powerful but operationally
complex and sensitive to reward model quality.

Core references:

- InstructGPT / RLHF: <https://arxiv.org/abs/2203.02155>
- Hugging Face TRL PPO tooling: <https://huggingface.co/docs/trl/main/en/trainer>
- OpenRLHF: <https://openrlhf.readthedocs.io/>

Practical rule: do not start DayCare with PPO-style RLHF. It adds machinery before the data, task, and reward contract
are mature.

## 7. RLVR / GRPO / Verifiers

RLVR trains with rewards that are mechanically checkable. This is a natural match for code, math, schemas, compilers,
unit tests, and route ledgers.

Core references:

- Training verifiers on GSM8K: <https://arxiv.org/abs/2110.14168>
- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- RLVR reasoning analysis: <https://arxiv.org/abs/2506.14245>
- Awesome RLVR: <https://github.com/opendilab/awesome-RLVR>
- TRL GRPOTrainer: <https://huggingface.co/docs/trl/en/grpo_trainer>

Practical rule: RLVR is promising for BoltBeam only where the reward is strict and cheap: schema validity, candidate
id exists, rollback exists, cited artifact exists, verdict matches ledger, no promotion without correctness and route
evidence. Use expensive GPU speed tests as final verification, not dense training reward.

## 8. Data Curation And Evaluation

Training quality is mostly data quality. This layer cuts across every method.

Core references:

- The Pile: <https://arxiv.org/abs/2101.00027>
- Dolma: <https://arxiv.org/abs/2402.00159>
- DataComp-LM: <https://arxiv.org/abs/2406.11794>
- Benchmark contamination survey: <https://arxiv.org/html/2406.04244v1>
- Soft contamination: <https://arxiv.org/html/2602.12413v1>

Practical rule: keep training, validation, and final evaluation artifacts separated by candidate, date, and verdict.
Never train on the final promotion gate. For BoltBeam, the ledger can become both a training source and a contamination
risk.

## 9. JEPA / World Models

JEPA-style training predicts latent representations rather than tokens. It is relevant to representation learning and
world-model research, but it is not the first tool for a BoltBeam assistant.

Core references:

- LeCun, "A Path Towards Autonomous Machine Intelligence": <https://openreview.net/pdf?id=BZ5a1r-kVsf>
- I-JEPA: <https://arxiv.org/abs/2301.08243>
- V-JEPA overview: <https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/>
- V-JEPA 2: <https://arxiv.org/abs/2506.09985>
- LLM-JEPA: <https://arxiv.org/abs/2509.14252>
- VL-JEPA: <https://arxiv.org/abs/2512.10942>

Practical rule: track JEPA as a representation idea. For a text-only BoltBeam experiment, predict embeddings of hidden
verdicts, bottlenecks, next actions, or route outcomes from the visible audit context. Treat it as a sidecar latent
state model until held-out retrieval/classification proves that it adds signal beyond SFT.

## 10. Alignment Removal / Uncensored Models

This topic is useful as a behavior case study: model behavior can be shaped by dataset filtering, post-training, or
representation edits. It is not the main DayCare objective.

Core references:

- Eric Hartford, "Uncensored Models": <https://erichartford.com/uncensored-models>
- Dolphin: <https://erichartford.com/dolphin>
- Abliteration: <https://huggingface.co/blog/mlabonne/abliteration>
- Refusal direction: <https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction>
- Defense paper: <https://arxiv.org/html/2505.19056v1>

Practical rule: study this to understand how data and representation edits move behavior. Do not make alignment
removal a training target for BoltBeam.

## 11. model Lineage (artificial-life prior art)

The *Creatures* series of artificial-life games is a useful historical reference for trainable companion agents. The
original commercial code is not the main reusable artifact, but `openc2e` provides an open-source engine
reimplementation, and old developer resources describe brain/genetics interfaces.

Core references:

- `openc2e`: <https://github.com/openc2e/openc2e>
- `openc2e` project site: <https://openc2e.github.io/>
- Creatures Developer Resource: <https://double.nz/creatures/>
- Grand and Cliff paper: <https://www.sci.brooklyn.cuny.edu/~sklar/teaching/s10/alife/papers/grand-cliff-jaamas97.pdf>
- `nornbrain`: <https://github.com/thechimpmatrix/nornbrain>

Practical rule: copy the architecture pattern, not the 1990s implementation: persistent state, drives, memory,
grounded actions, online learning, and feedback loops. For DayCare, combine a text-JEPA latent state model with an
LLM language layer and verifier-backed task gates.

## 12. Open-Source Training Tooling

Tooling should match the training stage:

- TRL for SFT, DPO, GRPO experiments: <https://huggingface.co/docs/trl/en/index>
- PEFT for LoRA/QLoRA adapters: <https://huggingface.co/docs/peft/en/index>
- Axolotl for config-driven fine-tuning: <https://github.com/axolotl-ai-cloud/axolotl>
- LLaMA-Factory for broad model/method support: <https://github.com/hiyouga/LLaMA-Factory>
- OpenRLHF for distributed RLHF/RLVR systems: <https://openrlhf.readthedocs.io/>

Practical rule: start with TRL + PEFT. Move to Axolotl/LLaMA-Factory when repeatability and config management matter.
Move to OpenRLHF only when RL rollout throughput is the bottleneck.

## 13. BoltBeam Training Dataset

The best DayCare-specific data is not generic chat data. It is structured evidence:

- prompt: current candidate state, artifacts, measured evidence, constraints
- target: verdict, next action, scoped implementation packet, or refusal to promote
- metadata: candidate id, route family, evidence refs, date, model, target, correctness status, speed status

Practical rule: the first useful dataset is SFT over high-quality audit decisions. Then build DPO pairs from good vs
bad reasoning traces. Then add RLVR only where a verifier can grade the output without judgment.

## Recommended Sequence

For DayCare/BoltBeam:

1. Build a clean trace dataset from existing BoltBeam decisions.
2. Train a LoRA/QLoRA SFT model to imitate disciplined audit output.
3. Add DPO/KTO/ORPO pairs for scoped vs speculative decisions.
4. Add RLVR for schema/ledger/verifier-backed subtasks.
5. Consider continued pretraining only after you have enough domain corpus to teach durable representations.
6. Avoid full pretraining until the corpus and compute budget justify a base model.
