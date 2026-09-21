# Nemotron retrieved-tool failure training

## Decision

**Reject the failure-trained candidate.** It scored 246/256 on the frozen
retrieved-tool holdout, down from 251/256 for the starting categorized adapter.
The candidate is retained as an experiment artifact and was not deployed or
transferred to GameTerm.

## Question and method

This run tested whether a small continuation LoRA, trained on new examples from
the starting model's failure categories, improves tool selection when GameTerm
first retrieves ten of its 35 tool schemas. It is the first real use of
`daycare.nursery.native_tool_sft`.

The method follows prior work rather than inventing an untracked recipe:

- ToolLLM separates retrieval from instruction-tuned tool use, matching the
  retrieval-then-native-SFT boundary used here: <https://arxiv.org/abs/2307.16789>
- ToolACE emphasizes diverse, verified function-call data. This run used
  deterministic labels, JSON Schema checks, and GameTerm normalizer checks:
  <https://arxiv.org/abs/2409.00920>
- Tool-REX reports gains from standardized descriptive tool metadata, which is
  what the hybrid retriever indexed: <https://arxiv.org/abs/2510.22670>
- STAR supplies the narrower hard-negative motivation for mixing confused and
  random tools. DayCare treats this as a data heuristic because its retrieval
  objective is not the same as token SFT: <https://arxiv.org/abs/2104.08051>
- The update itself is a frozen-base, rank-4 LoRA:
  <https://arxiv.org/abs/2106.09685>

The 416 training examples contained 112 plain answers (26.9%) and 304 tool
calls over 14 variable tool menus. The run continued from the categorized
adapter for one epoch at learning rate 2e-5, alpha 8, with gradient clipping at
1. Only the final MLP block was adapted. All 416 optimization steps were finite.

The 256-item deciding suite was frozen before training and has no exact request
overlap with training. It contains 32 cases for each of calculate, terminal,
files, apps, media, windows, web, and no-tool behavior. Both arms used the same
hybrid BM25/MiniLM rankings, top ten schemas, canonical GameTerm schema order,
temperature zero, and disabled thinking. Hybrid retrieval included an accepted
tool for all 224 tool-needed cases at rank 10.

## Results

| Group | Before | After | Change |
| --- | ---: | ---: | ---: |
| Apps | 32/32 | 32/32 | 0 |
| Calculate | 32/32 | 32/32 | 0 |
| Files | 32/32 | 32/32 | 0 |
| Media | 31/32 | 30/32 | -1 |
| No tool | 32/32 | 32/32 | 0 |
| Terminal | 32/32 | 28/32 | -4 |
| Web | 32/32 | 32/32 | 0 |
| Windows | 28/32 | 28/32 | 0 |
| **Total** | **251/256** | **246/256** | **-5** |

The paired difference is -1.95 percentage points. Across item differences its
sample standard deviation is 0.1387 and its standard error is 0.00867. A seeded
paired bootstrap gives a 95% percentile interval of [-3.91, -0.39] percentage
points. All five changed decisions were losses and none were gains; an exact
two-sided McNemar test gives p=0.0625. The primary adoption gate is practical,
not a significance loophole: the candidate had to avoid regressions and did
not.

Argument-schema failures remained 10 in each arm. The score change came from
selection, not newly malformed arguments.

## What failed

All five lost cases switched to `window_list`. Four asked for a shell action
while their retrieved menus contained `terminal_list` but omitted the training
label `bash`. The fifth asked to play a song while its menu contained
`media_status` but omitted the training labels used for that intent. The frozen
answer key accepts those fallback tools, and the starting adapter selected
them. The continuation data used one canonical target for each training
category and did not teach the accepted alternatives under those shortlist
conditions.

This is strong diagnostic evidence that the new examples narrowed the model's
fallback behavior. It does not establish that `window_list` frequency alone
caused the regression. The next dataset must cover accepted fallback labels
under realistic retrieved menus; lowering the learning rate alone would not
repair missing supervision.

## General capability and retention

| Probe | Before | After | Paired change | 95% paired bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| Minimal capability | 103/180 | 102/180 | -0.56 pp | [-2.78, +1.67] pp |
| Full GameTerm prompt | 89/140 | 85/140 | -2.86 pp | [-6.43, 0.00] pp |
| Combined diagnostic | 192/320 | 187/320 | -1.56 pp | [-3.75, +0.31] pp |

The minimal probe changed by one answer and its interval spans zero, so this run
does not show a general-intelligence regression. ARC was unchanged at 69/80;
GSM8K was unchanged at 6/40; HellaSwag changed 40/80 to 39/80; and MMLU changed
77/120 to 73/120 when the two prompt settings are combined. The combined suite
contains the same multiple-choice questions under two prompt settings, so its
320 rows are a diagnostic rather than 320 independent questions. The plain
no-tool slice stayed 32/32.

## Reproducibility

The immutable experiment directory is
`<server>/storage/daycare-runs/nemotron-retrieval-train-001`. Its manifest
records these pre-training hashes:

- dataset: `283b7698f50a4ea1a60b4c869c6c18012856c497894d1f16b760e983bbaf5560`
- deciding suite: `bf4276a01a5d485bf70b7937beb33174b3470732948f4ef74dfd0c89c6e65a88`
- production envelope: `1af1cf8496be505ec13fd3b65476ba80e134dc0d1adfff8ceba61760a18412e7`

Training used clean DayCare revision
`4cc927efc57b057ec91137d7be16c3122f433181`. The final adapter SHA-256 is
`76d78c9b1f7695c51989363bf4cc6fcd7840ef2785446c94817ea3579a6f0c72`;
the evaluated Q4_K_M SHA-256 is
`cdaaa9fdd89af7835bc54409afab1e7166ae0016d9f19d8588ff9dfa34ec12ad`.
`analysis.json` in the experiment directory records paired statistics, every
changed item, and hashes of evaluation artifacts.
