# JEPA And World Models

## Core Idea

JEPA means Joint-Embedding Predictive Architecture. Instead of predicting raw
pixels or raw tokens, a model predicts representations in latent space.

The intuition:

```text
context representation
  -> predict target representation
  -> learn semantic structure without reconstructing every surface detail
```

JEPA is not a standard LLM fine-tuning method. It is closer to a
self-supervised representation/world-model direction.

For DayCare, the interesting version is text-only JEPA: predict the latent
representation of a hidden or future text span from the representation of the
visible context, without training the model to reconstruct every token.

## Foundational References

- LeCun, "A Path Towards Autonomous Machine Intelligence":
  <https://openreview.net/pdf?id=BZ5a1r-kVsf>
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture" (I-JEPA):
  <https://arxiv.org/abs/2301.08243>
- Meta AI V-JEPA overview:
  <https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/>
- V-JEPA 2:
  <https://arxiv.org/abs/2506.09985>
- LLM-JEPA:
  <https://arxiv.org/abs/2509.14252>
- VL-JEPA predicts target text embeddings instead of autoregressive target
  tokens in a vision-language setting:
  <https://arxiv.org/abs/2512.10942>

## Text-Only JEPA Shape

A text-only JEPA experiment should keep the JEPA principle intact:

```text
context text
  -> context encoder
  -> predictor
  -> predicted target embedding

target text
  -> frozen or momentum target encoder
  -> target embedding

loss = distance(predicted target embedding, target embedding)
```

The target is not the raw token sequence. The target is the embedding of the
held-out span, future step, verdict, or outcome. That is the point: force the
model to learn semantic state instead of surface reconstruction.

Useful DayCare/BoltBeam targets:

- next audit verdict embedding;
- next scoped action embedding;
- candidate outcome embedding;
- route bottleneck embedding;
- roofline bucket embedding;
- "missing evidence" state embedding.

## Minimal Experiment

Start with a small encoder/predictor before touching a full LLM:

```text
input:  BoltBeam state summary + candidate facts + measured artifacts
mask:   hide verdict / next action / bottleneck summary
target: embedding of the hidden field from a frozen sentence encoder
train:  context encoder + predictor to match that target embedding
eval:   nearest-neighbor retrieval or linear probe over held-out outcomes
```

This keeps the first experiment cheap and falsifiable. If the learned latent
state cannot retrieve the right verdict/action on held-out candidates, it is
not ready to replace or augment SFT.

## Evaluation For Text-Only JEPA

Do not grade it by generative fluency. Grade it as a representation model:

- held-out verdict retrieval;
- held-out bottleneck classification;
- next-action classification;
- candidate-family holdout generalization;
- separation between "promote", "refute", "blocked", and "needs evidence";
- improvement when its embedding is added as a feature to a simple classifier.

Only after that should it feed a text decoder or proposal model.

## Pros

- Learns abstract representations rather than reconstructing low-level details.
- Strong fit for perception and world-model settings.
- Can be self-supervised.
- May be useful for planning and prediction in latent space.
- Text-only JEPA could learn compact compiler/search state without copying
  answer wording.

## Cons

- Not a drop-in replacement for SFT/LoRA/DPO on LLMs.
- Harder to connect directly to route decisions than supervised labels.
- Tooling is much less standardized than TRL/PEFT.
- Evaluation is less straightforward for codegen/search tasks.
- Text targets can collapse into weak semantic averages unless the target
  encoder, masks, negatives, and held-out evaluations are chosen carefully.

## Relevance To BoltBeam

JEPA is interesting if the target becomes:

```text
learn a latent model of hardware/compiler behavior
```

For example:

```text
candidate route + target facts + shape facts
  -> latent performance state
  -> predicted bottleneck / reachability / next action
```

But it is probably not the first training layer. The first layer should be:

```text
SFT/LoRA on BoltBeam traces
```

Then maybe:

```text
latent predictive model over candidate outcomes
```

Text-only JEPA is worth trying as a sidecar representation model, not as the
first replacement for SFT. The most practical path is:

```text
SFT assistant for readable decisions
  + text-JEPA state encoder for latent outcome prediction
  + verifier/BoltBeam gates for authority
```

The JEPA model proposes or ranks latent states. It does not promote candidates
without the normal correctness, route, speed, and roofline evidence.

See also model Lineage (artificial-life prior art)
for a related architecture where a latent state model sits inside a persistent
agent loop with drives, memory, user feedback, and environment transitions.
