# State vs Weights: what the field already settled

We derived the primitive route from our own principles, then measured it the hard
way. It turns out the field ran this race first, at scale, and got the same answer.
This note records the prior art so we stop re-deriving it -- and so we cite it
rather than claim we invented it.

Governing rule: Scope: What A model Should Learn. The measured
DayCare failure: [Catastrophic Forgetting](catastrophic-forgetting.md).

## The finding: for facts, the file beats the weights

**Ovadia et al., "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs"**
(<https://arxiv.org/abs/2312.05934>):

> *"While unsupervised fine-tuning offers some improvement, **RAG consistently
> outperforms it**, both for existing knowledge encountered during training and
> entirely new knowledge."*

And the part that should end the argument: **RAG alone outperformed RAG +
fine-tuning.** Adding the weight training made it *worse*.

So putting a fact in the context is not a shortcut we settled for. For knowledge
injection it is **the better-performing method**, measured on real models. Our
identity result (weights cost 2/5 of general capability to buy 1/3 of the name) is a
small, local reproduction of a known result.

## Our L4 is Reflexion

**Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"**
(NeurIPS 2023, <https://arxiv.org/abs/2303.11366>):

> *"reinforce language agents **not by updating weights**, but instead through
> linguistic feedback. Reflexion agents verbally reflect on task feedback signals,
> then maintain their own reflective text in an **episodic memory buffer** to induce
> better decision-making in subsequent trials."*

That is exactly `daycare/subject` L4: a correction is stored in `body.xml` and the
belief block carries it forward, so she avoids repeating it -- no weights touched.
Reflexion reached 91% pass@1 on HumanEval against GPT-4's 80% doing this.

We arrived at it independently from the minimization principles. That is a good sign
for the principles and a reason to cite the prior art, not to claim novelty. Their
framing is also a useful correction to ours: they call the stored text an *episodic
memory buffer* and treat feedback (scalar **or** free-form language) as the training
signal -- which is the same shape as our reinforcer + correction pair.

## The split the field uses

```text
facts, corrections, identity, preferences  ->  the file / context   (RAG, memory)
style, format, skill, behaviour            ->  fine-tuning
```

This is the **same declarative/procedural split** model sec 8 already
wrote -- *"declarative -> stays in the state file... never risked to SGD; procedural
-> migrates into weights."* The literature agrees with the rule we wrote and then
broke. Sec 8 was right; we carved the exception.

## What this closes, and what it leaves open

**Closed.** "Does weight-learning beat the file at facts/corrections?" -- no, per
2312.05934, on bigger models with a better setup than ours. Re-running that race on a
0.6B would spend GPU to re-derive a published result. Weights are not the route for
declarative content, and our own measurement agrees.

**Still open, and better founded.** Fine-tuning *is* the field's tool for
**procedural** targets -- style, format, a way of working. So the honest remaining
question is not "weights vs file for facts" but:

> Is there a **procedural** target -- something you cannot write down as a fact or a
> rule -- where weights genuinely beat the primitive route, and pay for the
> capability they cost?

That is the only version of the race worth GPU time, and it is the version sec 8
predicted all along.

## Practical rules

1. **Facts/corrections/identity -> state.** Cited, measured, and now twice-confirmed.
   Do not train them.
2. **Cite Reflexion when describing the correction loop.** It is prior art; our L4 is
   an implementation of it with drives attached.
3. **Reach for weights only for procedural targets**, and only with an off-target
   probe that can fail (the capability price is real: 2/5 in one epoch).
4. **RAG+fine-tune is not automatically better than RAG.** 2312.05934 found the
   combination *worse* than retrieval alone. Adding the weight route can subtract.
