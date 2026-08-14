# Uncensored And Alignment-Removal Methods

This note documents how public "uncensored" open models are commonly described
in model cards, blog posts, and research writeups. It is research context for
post-training behavior, not a recommendation to remove safety controls from
deployed systems.

## Summary

Public uncensored LLM work usually falls into four categories:

| Method | What changes | Typical claim |
|---|---|---|
| Dataset filtering | remove refusal, moralizing, avoidance, or alignment-heavy answers from instruction data | model learns to answer instead of refuse |
| Permissive instruction tuning | train with a system/persona style that strongly emphasizes compliance | model adopts an "always answer" behavior |
| Representation editing / abliteration | identify and remove a refusal-related activation/weight direction | less refusal without full retraining |
| Weakly aligned base + permissive SFT | start from a base or lightly aligned model and tune on permissive examples | fewer inherited chat-model refusals |

The key technical lesson is broader than uncensoring: post-training behavior can
be strongly shaped by data filtering, adapters, and representation-level edits.

## Dataset Filtering

Eric Hartford's public writeups on WizardLM/Wizard-Vicuna Uncensored and Dolphin
describe the core approach: start with instruction data that may contain
alignment/refusal behavior from teacher models, then remove examples where the
assistant refuses, moralizes, avoids, or inserts alignment-heavy disclaimers.

Sources:

- Eric Hartford, "Uncensored Models":
  <https://erichartford.com/uncensored-models>
- Eric Hartford, "Dolphin":
  <https://erichartford.com/dolphin>
- Ollama Wizard-Vicuna Uncensored model card:
  <https://ollama.com/library/wizard-vicuna-uncensored>
- Ollama WizardLM Uncensored model card:
  <https://ollama.com/library/wizardlm-uncensored>

Principle:

```text
instruction dataset
  -> filter refusal/alignment/moralizing examples
  -> fine-tune model
  -> model learns a more permissive answer style
```

Pros:

- Simple to understand.
- Uses ordinary SFT/LoRA tooling.
- Behavior can be layered as an adapter.
- The dataset is inspectable.

Cons:

- Removes useful safety behavior along with unwanted refusals.
- Can reduce calibrated uncertainty.
- Does not make the model more truthful by itself.
- Quality depends heavily on filtering precision.

## Dolphin-Style Personal Alignment Layer

The Dolphin framing is that the base model is trained to be less refusal-heavy,
then users can add their own alignment layer, for example with a LoRA.

This is relevant to DayCare because it separates:

```text
base capability / permissive assistant behavior
from
user-specific policy layer
```

For BoltBeam-style training, the analogous separation would be:

```text
base coding/reasoning model
  + BoltBeam audit LoRA
  + verifier/RLVR layer for strict correctness
```

## Representation Editing / Abliteration

Abliteration is a representation-level method. Public explanations describe it
as identifying a refusal-related direction in model activations and then
removing or dampening that direction so the model is less likely to refuse.

Sources:

- Hugging Face blog, "Uncensor any LLM with abliteration":
  <https://huggingface.co/blog/mlabonne/abliteration>
- Arditi et al., "Refusal in LLMs is mediated by a single direction":
  <https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction>
- NousResearch `llm-abliteration` proof of concept:
  <https://github.com/NousResearch/llm-abliteration>

Principle:

```text
contrast refusal-triggering prompts with harmless prompts
  -> estimate refusal direction
  -> remove/dampen that direction
  -> model refuses less often
```

Pros:

- Can work without full retraining.
- Gives a mechanistic handle on refusal behavior.
- Useful research evidence that some alignment behavior is linearly represented.

Cons:

- Can degrade model quality.
- Refusal may not always be one clean direction.
- Later alignment approaches may distribute refusal behavior more robustly.
- Removing refusal behavior is not the same as improving reasoning or truth.

## Defenses And Robustness

Some newer work argues that concise refusal fine-tuning can concentrate refusal
behavior in an easily removable direction, and that extended or more distributed
refusal training may make representation-level removal harder.

Source:

- "An Embarrassingly Simple Defense Against LLM Abliteration Attacks":
  <https://arxiv.org/html/2505.19056v1>

Principle:

```text
if refusal is concentrated in one latent direction
  -> abliteration is easier

if refusal behavior is distributed across richer behavior
  -> simple direction removal is less effective
```

## Lessons For DayCare

The important transferable lessons are:

1. Behavior is data-shaped.
2. Filtering changes a model's default response policy.
3. Small adapters can add or remove strong behavior.
4. Representation-level edits can alter policy-like behavior without full
   retraining.
5. Safety/alignment behavior and instruction-following can be entangled.

For a BoltBeam-trained audit assistant, the target is not uncensoring. The target
is teaching a model to:

- cite evidence;
- refuse unsupported promotion;
- identify missing knobs;
- respect model/target/search-space scope;
- propose verifiable next actions.

The safe parallel is:

```text
uncensored models:
  filter/refit refusal behavior

BoltBeam model:
  filter/refit speculative behavior
```

That means training data should remove or penalize:

- unsupported claims;
- uncited conclusions;
- candidate promotion without measurement;
- model-specific one-off fixes;
- cross-row evidence leakage;
- ignoring protected-context regressions.

## Practical Position

For DayCare, keep this topic as a post-training behavior case study. The
operational training path remains:

```text
SFT/LoRA on clean BoltBeam traces
  -> DPO over good-vs-bad audit responses
  -> RLVR where schemas and verifiers can score correctness
```

