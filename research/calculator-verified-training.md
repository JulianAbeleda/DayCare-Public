# Verified calculator training: research and small-test protocol

## Research basis and limits

Toolformer (Schick et al., 2023) samples API calls, executes them, and filters
them by whether their results reduce loss on subsequent text. Its lesson for
DayCare is to verify useful downstream behavior, not just valid call syntax.
We do not reproduce its self-supervised sampling or loss filter: our labels
come from externally authored GSM8K training answers and manually constructed
expressions, independently checked by GameTerm's calculator.
<https://arxiv.org/abs/2302.04761>

ToRA (Gou et al., 2023) learns from interactive mathematical trajectories with
reasoning, programs, tool observations, and answers. We borrow complete
question-to-execution-to-answer supervision. Our no-thinking native function
calls are a different setting from its rationale/program format; its reported
accuracy is not a prediction for this experiment.
<https://arxiv.org/abs/2309.17452>

QLoRA (Dettmers et al., 2023), Appendix A.1, varies rank and layer placement;
rank was not correlated with final performance when adapting all layers in
that study. This does not establish sufficient capacity for our final-MLP-only
rank-4 adapter. It does mean that a rank increase is not an evidence-based cure
without a controlled comparison. We leave placement and rank fixed here.
<https://arxiv.org/html/2305.14314#A1>

Dror et al. (2018) describe choosing statistical tests for NLP evaluations.
For paired binary outcomes we report discordant gains/losses, exact two-sided
McNemar, item-difference SD/SE, and a paired bootstrap interval. One training
seed cannot estimate training-seed variance.
<https://aclanthology.org/P18-1128/>

## Local evidence and implementation

The [independent prompting confirmation](calculator-label-confirmation.md)
scored 8/128 both without demonstrations and with complete-expression
demonstrations. That is a prompting failure, not a test of corrected SFT.

Inspection of `nemotron-math-routing-004` found 112 ordinary no-history word
targets among 688 rows (16.3%). The old validator counted 256 post-tool answers
toward its ordinary-answer quota. Its last 100 updates contained only math
calls and math results. The row-mean loss gives each row one optimizer update;
row proportions therefore matter independently of completion lengths.

`daycare.nursery.native_tool_sft` now excludes historical tool conversations
from the ordinary quota. Its scheduler distributes each group's original
quota across the epoch, shuffling within groups without adding repeats. These
are DayCare engineering choices, not prescriptions or thresholds from a paper.
The 25% ordinary quota is also a local policy, not a research-derived optimum.

Old labels such as `48+24` for 48 clips plus half as many in May conceal an
intermediate calculation. New labels use `48+48/2`. Label acceptance requires
schema validity, real normalizer/execution success, and exact equality to the
external answer. Full expressions are manually reviewed against the question;
matching the answer alone cannot establish semantic correctness.

## Frozen pilot

Runner: `scripts/calculator_verified_pilot.py`. Artifacts stay outside Git in
`storage/daycare-runs/calculator-verified-sft-001`.

- 32 GSM8K training questions (indices 0–31), each with a complete-expression
  call target and a separate answer target conditioned on the actual output.
  Constants such as 52 weeks/year and 60 minutes/hour are explicitly allowed.
- 32 ordinary replay targets and 16 other-tool replay targets: 112 total rows,
  28.6% ordinary. Replay comes from the accepted adapter's training data.
- Start from the accepted corrected adapter. DayCare only; rank 4, alpha 8,
  final block only, learning rate 5e-5, three epochs, one fixed seed. No search
  or checkpoint selection against evaluation answers.
- Freeze and hash 64 fresh GSM8K questions (indices 2200–2263), 20 ordinary
  probes, and 16 other-tool selection probes before training. Check math text
  against located prior datasets/suites. Base-pretraining exposure is unknown.
- Preserve all 35 production tools, native template, thinking disabled,
  temperature zero, 160 output tokens per turn, and four assistant turns.
  No tool-forcing instructions or demonstrations in either evaluation arm.
- Record raw completions and finish reasons. Treat truncated completions as
  failures. Math uses the existing exact last-number parser for comparability;
  inspect traces and separately report strict number-only accuracy.
- Evaluate the same Q4 export format before and after. Report training-item
  accuracy only as a fit diagnostic, never as generalization evidence.

Investment gate: at least 8 additional correct held-out math answers out of 64,
positive paired-bootstrap lower bound, exact McNemar p < .05, no ordinary or
other-tool item losses, no new ordinary tool use, and zero candidate loop-limit
or unrelated-tool failures on math. This is a conservative spending gate,
not proof of broad capability retention. If it passes, run an independent
128-question confirmation before expanding data, rank, or compute. If it fails,
stop and record the result; do not expand the sample to find significance.

This comparison tests the combined repair package, not each repair's causal
contribution. A successful result would justify later ablations and multiple
training seeds, not immediate deployment. Final adoption still requires the
real GameTerm check on the Air; this experiment does not transfer a model.

## Result: spending gate failed

Implementation and frozen runner revision: `5c5a6ac`. Training completed all
336 updates in 1,107.6 seconds. Adapter SHA-256:
`ed8ee92b7c9405f4838c353fce4accc553abbd226459acbe5dc105d1a3b07194`.
Model, input, binary, output and execution provenance are saved with the run.

| Paired probe | Accepted starting Q4 | Repaired-training Q4 |
| --- | --- | --- |
| Fresh math correct | 12/64 | 12/64 |
| Fresh math calculator use | 0/64 | 0/64 |
| Ordinary answers correct | 19/20 | 20/20 |
| Other-tool selection correct | 15/16 | 16/16 |
| Training math correct (diagnostic) | 5/32 | 5/32 |
| Training math calculator use (diagnostic) | 0/32 | 1/32 |

The same 12 held-out math items were correct in both conditions: zero paired
gains and zero losses. Item-difference SD and SE are both zero; exact McNemar
p = 1.0. The empirical paired bootstrap degenerates to [0, 0] because every
observed paired difference is zero. **This is not a meaningful claim that the
population effect is exactly zero or that the models are equivalent.** Twenty
math response strings changed, but none of those changes altered correctness.
Strict number-only math accuracy was 11/64 in both arms.

Ordinary answers had one gain and no losses: SD 0.22361, SE 0.05, bootstrap
interval [0, 15] percentage points, exact p = 1.0. The gained item was formatting:
the starting model supplied the correct word `modern` plus an explanation;
the candidate complied with the word-only instruction. This is not evidence
of increased general intelligence. Other-tool selection had one gain and no
losses: SD 0.25, SE 0.0625, interval [0, 18.75] points, exact p = 1.0.
These intervals describe item sampling, not variability across training seeds.

No held-out response was truncated; no math item looped or called an unrelated
tool. Raw response inspection confirmed no tool calls on ordinary probes in
either arm. Thinking was disabled and returned reasoning content was empty.
Selection probes only inspected calls; no non-calculator actions were executed.

The scheduler repair was realized in training: the final 28 updates contained
eight ordinary examples, eight math calls, eight post-result answers, and four
other-tool examples. All 32 training expressions executed and matched the
external key before training. These establish implementation correctness, not
the causal efficacy of either repair.

After the primary gate failed, a **post-hoc diagnostic**, using only the 32
training questions, evaluated the unquantized BF16 export with identical
inference settings. It scored 5/32 and made zero calculator calls. Thus Q4
conversion cannot be the sole explanation for weak activation. This diagnostic
was not used to select another checkpoint or tune a second run.

Even the Q4 candidate's one training-item call was wrong: on train item 24 it
used `-(100+15+23)` rather than `100-15-23`, received -138, and answered -6.
Neither expression construction nor result consumption was demonstrated by
that call. Lower training loss did not establish successful task learning.

Do not expand the data, increase rank, or deploy this candidate on this evidence.
No independent 128-question confirmation was triggered, since the prerequisite
gain failed. All temporary inference servers were stopped. This result applies
to this small, one-seed, three-epoch final-MLP run; it does not refute Toolformer,
ToRA, complete-expression supervision, or LoRA generally. Before another
training campaign, inspect the first tool-decision token's loss and native
training/inference format parity: averaged completion loss can conceal a failed
decision token. That is a proposed diagnostic, not an established cause here.

Validation: `python3 -m pytest -q` — 579 passed, 9 skipped. `python3 sz.py` —
within repository budgets. The history-quota test rejects counting post-tool
answers as ordinary retention; the unequal-group scheduling test checks that
ordinary examples remain distributed throughout the epoch without duplication.
