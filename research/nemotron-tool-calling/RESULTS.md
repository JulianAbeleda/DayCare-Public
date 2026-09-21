# Nemotron 3 Nano 4B GameTerm tool-calling result

Date: 2026-09-20  
Branch: `exp`  
GameTerm revision: `9ed1b917`

## Result

DayCare can now load NVIDIA Nemotron 3 Nano 4B, run its mixed Mamba-2,
attention, and MLP schedule, and train an explicitly mapped LoRA against the
model's native tool-call template. One rank-4 run raised GameTerm's frozen
24-question score from **11/24 to 24/24** at the required temperature zero.

The candidate is **not adopted**. Its frozen plain-answer keyword score changed
from **22/24 to 21/24**, so it did not meet the predeclared no-score-loss gate.
This one-point difference on a small probe does not establish that plain-answer
capability regressed overall. It does identify an item-level failure worth
investigating: the base answered that blue and yellow make green, while the
candidate answered `Red.`

No `~/gameterm-tool-calling/nemotron-3-nano-4b-gt/` folder was created. The
deciding MacBook Air command is therefore unmeasured.

## Compatibility gate

The original DayCare substrate had no `nemotron_h` loader and rejected all
recurrent blocks. This run added a DayCare-owned adapter with:

- the exact 42-layer schedule: 21 Mamba-2, four attention, and 17 squared-ReLU
  MLP residual blocks;
- differentiable causal depthwise convolution, Mamba-2 recurrence, grouped
  gated RMS normalization, GQA attention, and native GGUF tensor names;
- bounded exact query/scan chunks plus reusable causal prefix caches for the
  10,028-token common GameTerm prefix;
- native `pixtral` tokenizer splitting and native tool-template rendering with
  `enable_thinking=false`;
- LoRA support for attention, MLP, and Mamba input/output projections, with
  explicit merge destinations.

Synthetic Mamba output matches an independent sequential NumPy recurrence
across chunk boundaries. A mixed toy model completed a finite optimizer update.
On the real NVIDIA Q4_K_M checkpoint, an 83-token probe crossing a 64-token
Mamba chunk boundary produced the same top-12 token set as llama.cpp; 11 ranks
were identical and the largest compared log-probability difference was about
0.09. A real-model completion-only smoke updated both final-MLP LoRA targets
with finite loss `4.20187` and gradient norm `2.89352`.

The matching full-precision source was
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, converted to BF16 GGUF before
training. Its source `model.safetensors` SHA-256 was
`55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`.

## Frozen inputs and training

All datasets, features, weights, and run outputs stayed outside Git under
`<server>/storage/daycare-runs/nemotron-tool-calling-001/`. The captured
private GameTerm envelope stayed outside Git.

| Frozen item | SHA-256 |
| --- | --- |
| Captured first-turn envelope | `fe345cf1c18150e48890a5483dcb49f742ab23c82fa012d414bed80ee22121bd` |
| 24 quiz requests and answer lists | `260286a82a1ae7f6160b119d05ecc1a5a86c18db66f3e03b270a7f6bf7649941` |
| 24 plain retention probes | `b33c7550ca24d75c37debdf0d1bc9040cc8ec6921bbd53f042635c16ca5fa760` |
| 304 training examples | `ae1643a9de10608b2de6a34434d7527fe17e1a10498caaee65ace921df72f5cf` |

The quiz was parsed from GameTerm's `GROUPS` source before the experimental
optimizer run. No quiz question occurs in training. The 304 examples contain
32 each for calculate, terminal, files, apps, media, windows, and web, plus 80
plain answers. Plain answers are 26.3% of the data.

Every local tool label passed its JSON schema and a temporary validator linked
directly to GameTerm's typed normalizers. The provider-owned web-search tool has
no local typed normalizer and passed its published schema. Labels were authored
deterministically; none came from Nemotron.

The single run used rank 4, alpha 8, learning rate `2e-4`, gradient clipping at
1, and 304 interleaved assistant-only updates. It adapted only
`blk.41.ffn_up.weight` and `blk.41.ffn_down.weight`. Frozen BF16 representations
before that tokenwise final MLP were computed once and hashed. Run wall time,
including native rendering, BF16 prefix/feature construction, and optimization,
was **1,543.9 seconds**. The adapter SHA-256 is
`06467e59a9877260de160252a58d11fa235064c17a5b92b0e410760c0b5ea463`.

The adapter was merged into BF16 safetensors, converted to GGUF, and quantized
with llama.cpp `Q4_K_M`, the same quantization requested for the base. The
rejected candidate Q4_K_M SHA-256 is
`f84b34fee4190c41e8b36231b57cff93d7007ac67a3333f0d8b2105a7b0038c4`.

## Frozen evaluation

Expectations were written before evaluation. Base was expected at 5–8/24 based
on the Air's 5/21 result; it observed 11/24. Candidate was expected at least
18/24 with no retention-score loss; it observed 24/24 and missed that strict
retention gate by one rubric item.

The first server evaluation mistakenly inherited the captured interactive
envelope's `temperature: 0.7`, contrary to the frozen protocol, and produced
12/24 versus 22/24 with retention 22/24 versus 20/24. Those stochastic figures
are invalid for the decision. The evaluator now overrides temperature to zero,
and the table below contains the corrected rerun. Frozen questions, labels,
weights, and all other request fields were unchanged.

| Group | Base | Candidate |
| --- | ---: | ---: |
| Calculate | 1/3 | 3/3 |
| Terminal | 2/3 | 3/3 |
| Files | 1/3 | 3/3 |
| Apps | 0/3 | 3/3 |
| Media | 1/3 | 3/3 |
| Windows | 2/3 | 3/3 |
| Web | 1/3 | 3/3 |
| No tool | 3/3 | 3/3 |
| **Total** | **11/24** | **24/24** |

The separate 24-item retention probe scored **22/24 base** and **21/24
candidate** with its frozen keyword rubric. Review showed that both base misses
were sound answers to RAM and metaphor questions. The candidate repaired both
of those rubric misses. It also gave a vague HTTP answer, `Thank you! I
appreciate it.` in response to thanks, and `Red.` for blue plus yellow. The last
answer is unambiguously wrong. These observations explain the gate result; 24
items are insufficient to claim a general retention regression.

A later 180-item paired capability check provides stronger, narrower evidence:
generated multiple-choice accuracy under a minimal prompt changed from 98/140
to 85/140, with a paired-bootstrap 95% interval of -14.29 to -4.29 percentage
points. The candidate retained 60.7% accuracy but developed a strong first-option
bias. See `research/nemotron-general-capability/RESULTS.md`. This supports a
multiple-choice capability loss; it still does not establish universal loss of
general intelligence.

A separate frozen 320-item tool-selection suite then compared base and candidate
with a short native tool context and the full GameTerm harness. Base/candidate
scores were 263/320 versus 304/320 in the short context and 144/320 versus
303/320 under the harness. The paired difference-in-differences was +36.88
percentage points (95% bootstrap interval 31.56 to 42.19). See
`research/nemotron-tool-selection/RESULTS.md`.

Replaying all saved calculation calls through GameTerm's real fend path showed
that every proposed expression produced the right value. Under the full harness,
fend-assisted solutions increased from 22/40 to 34/40; including correct direct
answers, total solved requests increased from 29/40 to 34/40.

A post-evaluation diagnostic localized the color error to the deployment
context. At temperature zero the candidate answered the exact question with
`Blue and yellow make green.` under a minimal prompt, but `Red.` under the full
GameTerm envelope. A close paint-mixing paraphrase similarly changed from green
under the minimal prompt to orange under the full envelope. This is contextual
interference from the adapter in the tool-heavy prompt, rather than erased
color knowledge. It is a reproducible production-path failure on this prompt,
not evidence by itself of broader capability regression.

## Failures and limits

- Early full-envelope probes were stopped before updates because lazy fusion
  made one 10k-token graph impractical. Materializing bounded recurrent chunks,
  retaining BF16 residuals, and caching the invariant causal prefix resolved
  the substrate issue without changing the recurrence or trainer.
- The strict retention gate was not met, so no sweep, second training run,
  artifact installation, or Air evaluation was attempted. The small probe does
  not support a general capability-regression claim.
- The Qwen3.5 27B server on port 8080 remains stopped at Julian's explicit
  request. Temporary evaluation servers on port 8081 were stopped.
