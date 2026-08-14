# Catastrophic Forgetting

Teaching a model something new can destroy what it already knew. This is not a
side-topic for DayCare: it is the reason the whole two-timescale design exists
(waking state vs sleeping consolidation, [model](subject.md) sec 8), and it is what
killed our first real training result.

> Opinion, separated from fact per repo principles: the measured DayCare result and
> the cited papers are facts; the practical rules at the end are our calls.

## The DayCare instance (measured, 2026-07-15)

We taught Ada her name and lobotomized her. Two *unrelated* learning mechanisms,
one curriculum, identical outcome:

```text
                      identity (target)   forgetting (guard)
batch LoRA SFT             4/4                  0/5          FAIL
online fast weights        4/4                  0/5          FAIL
```

She answers "Ada" to *"what is 2+2?"*. Details and artifacts:
[Scope: Learning the Name](name-learning-scope.md).

Three things this taught us that the papers below then explained:

1. **The method was not the variable -- the data was.** Two mechanisms, same
   failure. `curriculum.train_examples` was 12 prompts that all mean "say Ada",
   with zero general data and zero counter-examples.
2. **A narrow objective became her whole identity.** Not "forgot some trivia" --
   every prompt now returns the trained answer. Narrow training had broad effects.
3. **The metric could not fail.** "Does the reply contain 'ada'" is satisfied 4/4
   by a model that says Ada to anything. An on-task metric alone cannot detect
   this; only an *off-task* probe can.

## The bigger point: for facts, you should not have been training at all

Our failure is a local reproduction of a settled result. Ovadia et al. compared
knowledge injection head to head and found **RAG consistently outperforms
fine-tuning** -- and that **RAG alone beats RAG + fine-tuning**
(<https://arxiv.org/abs/2312.05934>). The correction loop we built is Reflexion,
which reinforces agents *without weight updates* via an episodic memory buffer
(<https://arxiv.org/abs/2303.11366>). Full note: [State vs Weights](state-vs-weights.md).

So the honest framing of this whole episode: we paid GPU time and a lobotomy to
rediscover that declarative content belongs in the file -- which sec 8 of subject.md
said, and the literature had already measured.

## What the literature says

**LoRA already mitigates it -- and we still blew through it.** Biderman et al. find
LoRA "mitigates forgetting more than common regularization techniques such as weight
decay and dropout" and "helps maintain more diverse generations", because full
finetuning learns perturbations of 10-100x greater rank.
<https://arxiv.org/abs/2405.09673>

We used LoRA, on the top-8 layers only, and still collapsed to a single output
(total loss of generation diversity -- the exact symptom that paper measures). So
the lesson is not "LoRA is unsafe"; it is that a single-intent curriculum defeats
the method that was protecting us.

**Replay/rehearsal is the standard mitigation, and it has known ratios.** Mixing
general-domain data into the task data so the model keeps rehearsing pretrained
knowledge is the workhorse fix; surveys sweep task:general ratios of 1:1, 1:2, 1:5,
1:7, 1:10 (GeoGalactica used 8:1:1). GeRe shows a *small fixed set* of general
replay samples is enough -- which suits a 12-example curriculum.

- Continual Learning of LLMs (survey): <https://dl.acm.org/doi/10.1145/3735633>
- GeRe, general-samples replay: <https://arxiv.org/html/2508.04676v1>
- Improved SFT to mitigate forgetting: <https://arxiv.org/html/2506.09428v2>
- EWC (regularization-based, the classic): <https://arxiv.org/abs/1612.00796>

**Narrow finetuning has broad, non-local effects.** Betley et al. finetuned models
narrowly (write insecure code) and got models that behaved differently across broad,
*unrelated* prompts -- "emergent misalignment". Two details matter to us: the effect
was **strongest in Qwen** (Qwen2.5-Coder-32B) and GPT-4o, and **adding a benign
motivating context to the dataset prevented it**. So the framing of the data, not
just its labels, decides whether narrow training generalizes catastrophically.

- Emergent Misalignment: <https://arxiv.org/abs/2502.17424>
- Persona-Model Collapse in Emergent Misalignment: <https://arxiv.org/pdf/2605.12850>

We are on Qwen3, and our failure is precisely a persona collapse: she did not merely
forget, she *became* the training target. For a project whose whole subject is
giving an agent a persistent self, that is worth staring at: the thing we were
trying to install is exactly the thing that ate her.

## Practical rules for DayCare

1. **Never train on a single-intent curriculum.** Replay general examples at ~1:3 to
   1:5 (task:general) to start; a small fixed general set suffices (GeRe).
2. **Include counter-examples** -- prompts where the trained answer is *wrong*. Ours
   had none, so `identity_reward`'s anti-name-spam penalty never fired once.
3. **Gate in the loop, and gate off-task.** An on-task metric cannot detect
   collapse. Run `daycare/nursery/evaluate.py` (identity **and** forgetting) during
   training, not after the celebration.
4. **Keep LoRA, low rank, low lr, few epochs.** LoRA was helping. 12 examples x 4
   epochs = 48 steps on one intent is the overfit engine.
5. **Assume narrow training has broad effects.** Always measure something the
   training never mentioned.
6. **Context in the data matters**, not just targets (the benign-motivation result).

## Why this connects to the architecture

The two timescales ([model](subject.md) sec 8) are a response to this: fast weights
hold the recent past and fade, and only sleep -- *eval-gated* -- writes to the slow
weights. That gate is the whole point, and we skipped it. The design was right; the
discipline was missing.

The [fast-weight lobe](fast-weight-lobe-scope.md) makes this sharper, not safer:
per-turn updates are more exposure, not less. Its bounds (decay, clip, min_signal,
top-K) are the mitigation, and they must be measured against an off-task probe --
not trusted because they are written down.
