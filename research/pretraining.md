# Pretraining And Continued Pretraining

For the historical lineage from GPT-1 through ChatGPT, see
[Pretraining History](pretraining-history.md).

For the XML-first file hierarchy and artifact lifecycle, see
[XML Pretraining Format](xml-pretraining-format.md).

## Principle

Pretraining teaches the base distribution and broad capabilities by predicting
tokens over a very large corpus. It is the most expensive and highest-leverage
stage because it changes the foundation of what the model can represent.

Continued pretraining keeps the same objective but shifts the data distribution
toward a domain: code, math, biomedical text, compiler traces, or BoltBeam-style
performance logs.

## Key Papers

- Kaplan et al., "Scaling Laws for Neural Language Models" found empirical
  power-law relationships between loss and model size, data size, and compute:
  <https://arxiv.org/abs/2001.08361>
- Hoffmann et al., "Training Compute-Optimal Large Language Models"
  (Chinchilla) argued that many large models were undertrained and that model
  size and training tokens should scale together under a fixed compute budget:
  <https://arxiv.org/abs/2203.15556>
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer" studied transfer learning objectives and unified
  tasks into text-to-text form:
  <https://arxiv.org/abs/1910.10683>
- Radford et al., "Improving Language Understanding by Generative
  Pre-Training" is the GPT-1 reference for causal generative pretraining plus
  downstream fine-tuning:
  <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>

## Pros

- Builds real capability into the weights.
- Improves generalization when the domain distribution is large and diverse.
- Can teach concepts that are repeatedly useful, not just response style.
- Continued pretraining can absorb domain language and representations better
  than SFT alone.

## Cons

- Expensive.
- Easy to damage instruction-following behavior if continued pretraining is not
  followed by alignment/post-training.
- Requires strong data curation; garbage data becomes base behavior.
- Harder to debug than SFT because failures are distributed across the model.

## When To Use

Use pretraining or continued pretraining when the model lacks domain knowledge
or base representations:

- new programming language
- new hardware/compiler domain
- rare technical vocabulary
- traces that appear often enough to deserve compression into weights

Do not use it just to teach a response format. That is SFT territory.

## DayCare/BoltBeam Interpretation

For BoltBeam, full pretraining is not the first move. A more realistic option is
continued pretraining on:

- compiler traces
- performance reports
- route ledgers
- roofline reports
- model profiles
- measurement plans

Then use SFT/DPO/RLVR to teach the model how to act on those facts.
