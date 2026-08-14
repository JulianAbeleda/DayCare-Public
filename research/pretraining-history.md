# Pretraining History

This note tracks the history behind the pure LLM route. The goal is to
understand how modern chat models emerged from pretraining, not to reproduce
the full industrial stack.

## Short Lineage

```text
Transformer
  -> GPT-1: generative pretraining + supervised fine-tuning
  -> BERT/T5: pretraining as general transfer learning
  -> GPT-2: unsupervised multitask behavior from web-scale causal LM
  -> GPT-3: in-context/few-shot learning from scale
  -> InstructGPT: SFT + reward model + RLHF
  -> ChatGPT: conversational productization of post-trained LLMs
```

## 2017: Transformer

*Attention Is All You Need* introduced the Transformer architecture:
self-attention, feed-forward blocks, residual connections, layer normalization,
and parallelizable sequence training.

Reference:

- Vaswani et al., "Attention Is All You Need":
  <https://arxiv.org/abs/1706.03762>

Why it matters: the Transformer made large-scale sequence modeling practical.
GPT-style models are decoder-only Transformers trained with a next-token
objective.

## 2018: GPT-1

GPT-1 is the clean starting point for modern generative pretraining:

```text
large unlabeled text corpus
  -> causal language-model pretraining
  -> supervised fine-tuning on downstream tasks
```

Reference:

- Radford et al., "Improving Language Understanding by Generative
  Pre-Training":
  <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>

Why it matters: GPT-1 showed that a general language model could learn useful
representations from unlabeled text and transfer with relatively small
task-specific fine-tuning.

## 2018: BERT

BERT used masked language modeling and bidirectional context:

```text
unlabeled text
  -> mask tokens
  -> predict masked tokens using left and right context
  -> fine-tune for understanding tasks
```

Reference:

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding":
  <https://arxiv.org/abs/1810.04805>

Why it matters: BERT made pretraining the default recipe for NLP understanding
tasks. It is not the same architecture as GPT-style generation, but it proved
the transfer-learning thesis at scale.

## 2019: GPT-2

GPT-2 scaled causal language modeling on web text and showed that many tasks can
emerge from next-token prediction without task-specific supervised training.

Reference:

- Radford et al., "Language Models are Unsupervised Multitask Learners":
  <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>

Why it matters: GPT-2 moved the story from "pretrain then fine-tune every task"
to "a sufficiently large language model learns task behavior from natural text
patterns."

## 2019: T5

T5 framed many NLP tasks as text-to-text transfer learning:

```text
input text
  -> output text
```

Reference:

- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer":
  <https://arxiv.org/abs/1910.10683>

Why it matters: T5 made the pretraining/fine-tuning design space explicit:
objective, architecture, dataset, scale, transfer format, and task mixture.

## 2020: GPT-3

GPT-3 scaled decoder-only causal language modeling to 175B parameters and
showed strong zero-shot, one-shot, and few-shot behavior through prompting.

Reference:

- Brown et al., "Language Models are Few-Shot Learners":
  <https://arxiv.org/abs/2005.14165>

Why it matters: GPT-3 made in-context learning the central property of large
language models. But GPT-3 was still not a polished assistant. It was powerful,
but it did not reliably follow user intent.

## 2022: InstructGPT

InstructGPT is the key bridge from pretrained language model to useful
assistant behavior:

```text
GPT-3 base model
  -> supervised fine-tuning on labeler demonstrations
  -> reward model from ranked outputs
  -> RLHF policy optimization
```

Reference:

- Ouyang et al., "Training language models to follow instructions with human
  feedback":
  <https://arxiv.org/abs/2203.02155>

Why it matters: this showed that post-training can matter more for user
preference than raw model size. The paper reports that a 1.3B InstructGPT model
was preferred to the 175B GPT-3 model on their prompt distribution.

## 2022: ChatGPT

ChatGPT should be understood as:

```text
large-scale pretraining
  + instruction/demo fine-tuning
  + preference ranking
  + RLHF-style alignment
  + conversational interface
  + product feedback loop
```

The important lesson is that modern chat behavior did not come from pretraining
alone. Pretraining built broad capability; post-training shaped the interface,
helpfulness, refusal behavior, and conversational usability.

## DayCare Interpretation

For the pure LLM route, the practical lesson is:

```text
do not train a base model from scratch first
use model archaeology to pick an existing base or instruct model
curate DayCare/BoltBeam traces
SFT/LoRA for task behavior
DPO/ORPO/KTO for preference tradeoffs
RLVR/GRPO only where verifiers exist
```

For the model route, this history still matters, but the LLM is not the
whole agent. The language model supplies language competence and reasoning; the
model-style layer supplies persistent state, drives, consequences, and lifetime
adaptation.
