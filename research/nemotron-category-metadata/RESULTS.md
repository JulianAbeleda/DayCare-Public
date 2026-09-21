# Nemotron category-metadata experiment

Date: 2026-09-20  
DayCare branch: `exp`  
GameTerm source revision: `9ed1b917`

## Result

A fresh rank-4 Nemotron 3 Nano 4B adapter trained with GameTerm's visible tool
categories passed every predeclared server gate. On a fresh 256-request tool
holdout in the complete GameTerm envelope it rose from **128/256 (50.0%)** to
**233/256 (91.0%)**, while keeping all **32/32** no-tool questions as plain
answers. On a separately frozen 180-question capability holdout, the base scored
**106/180** and the candidate **103/180**. The paired delta was **-1.67
percentage points**, SD **19.70**, SE **1.47**, and bootstrap 95% CI **[-4.44,
+1.11]**. The interval includes zero, as the adoption rule required.

This supports adopting the candidate for the final MacBook Air smoke. It does
not support claiming category-aware training itself improved tool accuracy over
the earlier adapter. The earlier, uncategorized higher-rate adapter scored
**239/256** on this same fresh categorized tool holdout, six items above the new
candidate; paired delta **-2.34 points**, 95% CI **[-5.47, +0.39]**. The new
candidate did materially better on fresh minimal capability: **103/180 versus
94/180**, +5.00 points, 95% CI **[+0.56, +9.44]**. Category exposure, lower
learning rate, and the plain-answer schedule changed together, so that recovery
cannot be assigned to one component.

## What was tested

The production-shaped envelope retained all **35 complete tool definitions**.
It sorted them by GameTerm's authoritative category order, prefixed every full
description with `Category: <category>.`, and appended one compact category to
tool index to the system message. `enable_thinking` remained false and decoding
used temperature zero.

The 304 training examples were the original disjoint set: 32 examples for each
of seven tool groups and 80 plain answers (26.3%). Labels came from the answer
key and schemas, never the model. The adapter used rank 4, alpha 8, learning
rate `5e-5`, and 304 updates. Its SHA-256 is
`ba1c17c9cba2b9170e53d726621f38c8092ae6b6b8895e567a5ac9f33e95da96`.

Both deciding suites were frozen and hashed before training:

- tool selection: 256 questions, 32 each for calculate, terminal, files, apps,
  media, windows, web, and no-tool; SHA-256
  `e5a4ddf8fdc1f370ae72c4fc42cae88b23a36a89a0eae4f121976233adf7210e`;
- capability: 180 questions, MMLU 60, ARC 40, HellaSwag 40, and GSM8K 40;
  SHA-256 `53c569333b3a7b12a2574def37111e946c211c26cc69b2f0d9707df6f82ef89c`.

The tool suite had zero exact overlap with training, GameTerm's original quiz,
or the earlier development suite. The capability suite had zero item-ID overlap
with the earlier capability suite. Neither deciding suite was inspected or
changed after training began.

## Tool-selection result

| Group | Base, full | Candidate, full | Prior adapter, full |
|---|---:|---:|---:|
| calculate | 23/32 | 30/32 | 32/32 |
| terminal | 17/32 | 32/32 | 30/32 |
| files | 18/32 | 32/32 | 32/32 |
| apps | 2/32 | 15/32 | 18/32 |
| media | 22/32 | 32/32 | 32/32 |
| windows | 3/32 | 31/32 | 31/32 |
| web | 11/32 | 29/32 | 32/32 |
| no tool | 32/32 | 32/32 | 32/32 |
| **Total** | **128/256** | **233/256** | **239/256** |

For candidate versus base in the full envelope, the paired gain was **+41.02
points**, paired SD **52.37**, SE **3.27**, bootstrap 95% CI **[+34.77,
+47.27]**, with 109 gains and four losses; exact McNemar p = `1.29e-27`.
All **223/223** candidate calls had schema-valid arguments.

Under the short prompt, base scored 200/256 and candidate 237/256. The gain was
+14.45 points, SD 41.37, SE 2.59, 95% CI [+9.38, +19.53]. Candidate no-tool
accuracy was 30/32 there; the production-shaped full prompt restored it to
32/32.

The 23 full-envelope candidate misses were concentrated in apps (17), web (3),
calculate (2), and windows (1). Common app errors selected `window_list`,
Spotify search, or terminal authority for launch requests. Two arithmetic
strings were interpreted as dates and sent to `bash`. One web request selected
Spotify search. These are routing errors rather than malformed calls.

## Capability result

| Task | Base, minimal | Candidate, minimal | Prior adapter, minimal |
|---|---:|---:|---:|
| MMLU | 40/60 | 40/60 | 33/60 |
| ARC | 35/40 | 35/40 | 35/40 |
| HellaSwag | 23/40 | 22/40 | 22/40 |
| GSM8K | 8/40 | 6/40 | 4/40 |
| **Total** | **106/180** | **103/180** | **94/180** |

The candidate changed seven paired items versus base: two gains and five
losses. Exact McNemar p was 0.453. This suite cannot prove equality; it bounds
the observed regression at this sample size. Its 95% interval excludes losses
larger than 4.44 points under this paired bootstrap.

With the full categorized GameTerm envelope on the 140 multiple-choice items,
base scored 84/140 and candidate 89/140: +3.57 points, SD 18.62, SE 1.57,
95% CI [+0.71, +7.14]. The adapter did not turn these plain questions into tool
calls.

## What categories contributed

Before training, adding the metadata to the untouched base changed the existing
320-question development suite as follows:

| Prompt | Without categories | With categories | Paired delta |
|---|---:|---:|---:|
| Short | 263/320 | 247/320 | -5.00 points, CI [-9.38, -0.62] |
| Full GameTerm | 144/320 | 164/320 | +6.25 points, CI [+1.88, +10.94] |

The category representation therefore helped this base model navigate the long
production prompt, while hurting the short prompt. That is evidence for the
user's “easier to keep in your head” theory in the high-context use case. The
third adapter arm prevents a stronger claim: the new adapter did not beat the
old adapter's tool score. A clean causal test of category-aware training would
hold learning rate, curriculum order, and examples fixed and change only the
metadata.

## Fend-assisted math

Every proposed `calculate` call was passed through GameTerm's real normalizer
and fend runner. In the full envelope, base solved 25/32 when correct direct
answers were included; the candidate solved **29/32**. Valid fend expressions
executed correctly. The three candidate failures were two date-like
misclassifications to `bash` and one wrong expression, not fend engine errors.
This supports the operational claim that the model solves more of these math
requests by choosing and supplying the calculator; it says nothing about an
increase in internal arithmetic ability.

## Continuity and adoption

On GameTerm's original 24-question quiz under the categorized envelope, base
scored 13/24 and candidate 23/24. The sole miss chose
`request_terminal_authority` for a terminal request. Plain retention improved
from 22/24 to 23/24.

The candidate bundle is at
`~/gameterm-tool-calling/nemotron-3-nano-4b-gt/`. Its Q4_K_M SHA-256 is
`f2f17618b9c16d3ce17afbe47a84b68195539d4f78f86e7f871a1c214bbe225b`.
Server gates passed. The deciding native-app check remains **unmeasured** until
it runs on the Air:

```bash
python3.12 scripts/smoke-calibrate.py --model nemotron-3-nano-4b-gt
```

Raw private artifacts remain outside Git under
`~/storage/daycare-runs/nemotron-category-001/`. `analysis.json` contains the
machine-readable paired statistics. Temporary llama-server processes were
stopped; the original 27B server remains stopped.
