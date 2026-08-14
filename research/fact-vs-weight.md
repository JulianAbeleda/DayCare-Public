# Fact vs Weight: what goes in the file, what goes in the brain

The operational form of the governing rule in [Scope: What A model Should
Learn](learning-scope.md). "Declarative vs procedural" ([model](subject.md) sec 8) is
the right idea but too fuzzy to decide with at 2am. This is the version you can
actually apply.

Written because we got it wrong: we classified a **name** as something to train, and
it cost 2/5 of her general capability ([Catastrophic
Forgetting](catastrophic-forgetting.md)). The classification was the whole error.

## The test

```text
Fact    you could write it on a sticky note.
Weight  you could only teach it by showing a thousand examples.
```

And the one line that decides it:

> **Weights are for producing what you never wrote down.**
> If she only needs to **repeat** it -> file. If she needs to **invent** correctly in
> it -> weights.

The file can only give back what you put in. Weights produce what you never put in.
That generalisation is the *only* thing weights buy -- and you pay for it in
capability, so do not buy it for something you could have typed.

## 5 facts (-> her file)

| | why |
|---|---|
| "Your name is Ada" | One line. Done. |
| "Use snake_case, not camelCase" | You just told her. Save it. |
| "I like short answers" | A preference. Writable. |
| "Never delete without a backup" | A rule. Five words. |
| "Last week we chose XML over JSON" | It happened. Write it down. |

**The tell:** you can say it *once*, in words, and it is complete.

## 5 weights (-> her brain, maybe)

| | why not a fact |
|---|---|
| Writing code that *feels* like your codebase | A thousand tiny choices; no list captures it |
| Knowing when an answer is too long | Depends on everything. You would just know. |
| Good taste in naming | You cannot write the rule, only show the taste |
| Fluency in a new language | Not a list of sentences -- a whole skill |
| Sensing when someone is frustrated | Judgment, not a checklist |

**The tell:** you *cannot* say it once. You would have to show examples forever.

**Facts you tell her. Skills you show her.** Telling is cheap, instant, reversible.
Showing is expensive, slow, and changes her in ways you did not intend.

## The procedure

```text
1. Can you write it down?                     yes -> FILE. done.
2. Too big for the context window?            yes -> FILE + retrieval (RAG). still the file.
3. Must it generalise to cases you
   cannot enumerate?                          no  -> FILE. done.
4. Is it unconditional (applies every turn)?  yes -> FILE. Training it can only
                                                     teach "always" -> collapse.
5. Only now: a WEIGHT candidate -- and it still has to beat the file on a probe
   that can fail, and pay for the capability it costs.
```

Steps 1-4 catch nearly everything. **Weights are rare by construction** -- which is
why the field retrieves facts and fine-tunes style ([State vs
Weights](state-vs-weights.md)).

## The worked example: teaching her a new coding language

This is a **legitimate** weight target -- the first one in this project. It passes
every test:

| test | a new language |
|---|---|
| writable? | **no** -- infinite valid programs, not a list |
| must generalise? | **yes** -- she must write code nobody showed her |
| conditional? | **yes** -- you write it in its contexts, not everywhere |
| told or shown? | **shown** -- nobody learns a language from a rule |

She must **invent** correct code in it, not repeat code you wrote. That is the line.

Two catches, both real:

1. **A language splits in half.** Do not train what the docs already say.
   ```text
   syntax reference / stdlib API   ->  facts. retrieve them.
   writing idiomatic code in it    ->  skill. weights.
   ```
2. **The scale is nothing like ours.** A language is continued pretraining on a large
   corpus -- millions of tokens. Our whole setup (LoRA, top-8 layers, a 60-example
   curriculum, 0.6B) is sized to nudge behaviour, not install a language. Right
   target, wrong order of magnitude. This matches
   [Exhaustive Research Map](exhaustive-research-map.md) sec 2: continued pretraining
   is for *"durable domain representations"*.

And still: **try the file first, even here.** Modern models pick up a new language
from docs-in-context surprisingly well, because they already know *programming* -- a
new language is mostly a remix. Reach for weights when it must be her **default
idiom**, fluent without being handed the docs every time.

## Two traps

- **"Too many sticky notes" is not a reason to train.** 10,000 facts do not become a
  skill; they become a filing cabinet you look things up in (that is RAG, and it is
  still the file). Weights are not for *lots* of facts -- they are for
  *no-such-thing-as-a-fact*. These get conflated constantly, and the conflation is
  what sells fine-tuning.
- **Watch for "always".** If it applies on every turn, showing it can only teach
  "always". Her name is unconditional -- which is exactly how we got a model that
  answers "Ada" to *"what is 2+2?"*.

## Why we got it wrong

The name was **writable** (step 1 -> file) *and* **unconditional** (step 4 -> file).
Two independent rules both said file. We trained it anyway, and step 4 executed
exactly as written: an unconditional target can only teach "always".

The collapse was not bad luck. It was a misclassification.
