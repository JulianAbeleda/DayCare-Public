# Countdown end to end through GameTerm: thinking off vs on

> Published copy of a working record (inference only, stock model), lightly edited for publication. `<runs>/`
> stands for the private run-output directory; the analysis scripts named here are not published. GameTerm is
> [gameterm.arkey.ai](https://gameterm.arkey.ai); the experiments used a beta build that is not public.

Date: 2026-09-24. Stock Nemotron 3 Nano 4B BF16, no adapter, no training.

## Question

The [post-result probe](post-result-probe.md) localized the Countdown failure
to recovery after a calculator miss, and showed thinking fixes most of it for a
single turn. Does that hold for complete multi-turn episodes in the production
harness?

## Protocol

A diagnostic script (not published) drives GameTerm's headless
`calculator_study` (HT-020, which takes thinking from the
envelope and checks the first request against it). All 16 held-out Countdown
tasks never used for SFT (`rl-tasks.json` and `probe-tasks.json` in
`countdown-recipe-prep-003`: eight unseen, eight held-out; three- and
four-number), four server seeds (71–74), temperature 1.0, full 35-tool
production envelope, 4,096 tokens per turn, GameTerm's own turn loop and
limits. The two arms differ only in `enable_thinking`. Runs:
`<runs>/countdown-e2e-thinking-{off,on}-001/`.

## Result

| Metric | Thinking off | Thinking on |
|---|---:|---:|
| Correct | 15/64 (23%) | **46/64 (72%)** |
| Correct through a calculator result | 2 | 4 |
| Episodes that selected the calculator | 31 | 4 |
| Mean calculator calls per episode | 5.5 | 0.1 |
| Incomplete or truncated | 14 | 16 |
| Tasks with mixed rewards across seeds | 11/16 | 7/16 |
| Generated tokens (all requests) | 48,442 | 115,556 |
| Wall time, 64 episodes | 11 min | 11 min |

Paired by episode: 33 wins, 2 losses. Task-clustered bootstrap 95% interval
of the accuracy gain: +35.9 to +60.9 points.

## Interpretation

1. **Thinking is the lever, and it needs no training.** It more than triples
   end-to-end accuracy. Stock with thinking on is now the baseline any
   trained adapter must beat.
2. **With thinking on, Countdown is not a tool-use task.** The model solves
   it by reasoning and almost never calls the calculator. Countdown measures
   arithmetic search; it cannot show whether training improves tool use.
3. **The remaining failures stop at GameTerm's output cap.** Of the 18
   thinking-on failures, 16 were incomplete or truncated at about 4,096
   tokens; only 2 used the wrong numbers. *Superseded below:* the cap is
   GameTerm's 4,096-event provider limit, and more budget does not fix these
   episodes; they are hard four-number searches.
4. **Cost.** Thinking generated 2.4 times the tokens, but it replaced many
   short looping calls with few long turns, so wall time was unchanged here.
   Per-turn latency is higher and must be judged against GameTerm's
   interactive use; that is not measured here.

## Adoption gate: retention with thinking on

The retention-gate script (not published) runs the frozen 132-item GameTerm suite
from the large-model study (`large-model-comparison-001/suite.json`) on the 4B
model, thinking off vs on, temperature 0, 4,096 tokens, same envelope
otherwise. Runs: `thinking-retention-{off,on}-002` (identical to the `-001` runs made with a JSON-writing draft of the script). Automatic scores follow
`large_model_review.judge` (last-number rule for numeric answers).

| Family | Off | On (automatic) | Wins/losses |
|---|---:|---:|---:|
| plain (retention control) | 40/40 | 40/40 | 0/0 |
| math | 50/56 | 53/56 | 5/2 |
| oracle | 7/8 | 8/8 | 1/0 |
| smoke | 4/4 | 3/4 | 0/1 |
| selection | 11/16 | 15/16 | 5/1 |
| clarification | unmeasured | unmeasured | — |

Every disagreement was audited by hand. All 11 wins are real (thinking-off
answered wrongly or selected no tool). Three of the four losses are scorer
artifacts, not wrong answers: the last-number rule read "$14.00 ... for 3
trays" as 3, "fifteen dollars" as no number, and "rounded to 2 decimals" as 2.
The scorer was not changed after seeing results; the audit is reported
instead. The one real loss is a selection item (`terminal-fresh-07`) where
thinking-on added `request_terminal_authority`. Audited: numeric families
61/68 off vs 67/68 on; selection 11/16 vs 15/16.

Clarification, audited by hand against the suite's expected question: off
asked 0/8 times (it answered with assumed values), on asked 1/8 (`clarify-03`)
and on `clarify-06` called web search instead of asking. Neither arm is
acceptable; thinking does not regress it. This is a known weakness, not a
thinking effect.

**Verdict:** thinking-on passes retention with zero plain-answer losses and
net gains elsewhere. It is the DayCare baseline. Making it GameTerm's
deployment default is a GameTerm design decision and is proposed, not made,
here. Budget exhaustion still needs an explicit stop rule.

## Untrained levers after thinking (first turn, direct to llama-server)

The residual failures are not a budget problem. GameTerm's provider caps every
response at 4,096 stream events and 300 seconds (`crates/provider/src/lib.rs`,
`Limits::default`), independent of `max_tokens`; that cap produced every
thinking-on "truncation" above, and an 8,192-token GameTerm run was identical.
Measured without GameTerm, with the post-result probe's `initial` condition
(the same first request; runs `initial-turn-*`):

| Variant (thinking on) | Correct | Truncated | vs 4,096 baseline |
|---|---:|---:|---|
| 4,096 tokens, temperature 1.0 (baseline) | 42/64 | 16 | — |
| 8,192 tokens | 45/64 | 8 | 3 wins, 0 losses |
| temperature 0.6, top_p 0.95 (family guidance) | 37/64 | 17 | 9/14; CI −21.9 to +6.2 points |
| same + `--reasoning-budget 3072` | 37/64 | 13 | 9/14; no empty turns |

The 4,096 baseline matches the GameTerm run (42 vs 46 correct, 16 vs 16
truncated). Doubling the budget solved 3 of the 16 truncated episodes; 8 still
ran out. All residual failures are four-number tasks (at 8,192, which GameTerm
cannot deliver under its cap: three-number 27/32 with no truncation,
four-number 18/32 with all 8). Whether the truncated reasoning repeats itself
or is still searching has not been classified. The reasoning runs
away on the harder search rather than needing a little more room. The
family's reasoning-mode sampling did not help. The reasoning budget works as a
stop rule (every episode now answers) but the forced answers were wrong; it is
a user-experience control, not an accuracy lever.

**Conclusion:** stock with thinking on, default sampling, is the untrained
ceiling: about 72% end to end through GameTerm. The four-number holdout is
only 8 tasks × 4 seeds, too small to detect a realistic trained gain. A trained route must
beat it there without losing retention.

## Reopen gate for a trained route (2026-09-24)

An adversarial review recommended closing Countdown training (thinking-on
stock as the deliverable) unless a headroom gate passed: pass@k well above
pass@1 under GameTerm's cap on the *training* partition, with correct traces
fitting under the cap. Run `sft-partition-4num-thinking-001`: the 24
four-number `sft-tasks.json` tasks, first turn, thinking on, 4,096 tokens,
temperature 1.0, eight seeds (pass@8 instead of the suggested pass@16, to halve
cost).

| Measure | Result |
|---|---|
| pass@1 / @2 / @4 / @8 | 0.42 / 0.62 / 0.78 / **0.88** |
| Tasks with at least one success | 21/24 (none all-correct, 3 never) |
| Correct reasoning length | median 1,290 chars, p90 5,086, max 6,367 |
| Truncated samples | 81/192 |

The gap from pass@1 to pass@8 is 46 points, and every correct trace already
fits under the cap by construction. Of the 81 truncated traces, 59 repeat no
line in their last 3,000 characters and only 3 repeat more than three: the
reasoning is still searching, not looping. **The gate passes:** a
rejection-sampling fine-tune on self-generated correct thinking traces (no
solver-written traces) is justified as the next experiment. It must beat stock
thinking-on on a fresh, larger four-number holdout and pass the retention gate.
Result: +8.9 points but the interval crossed zero; not adopted
(that record is not published; its successor is
[thinking RFT with post-tool turns](thinking-rft-posttool.md)).

## Limits

16 tasks × 4 seeds; one model; one budget (4,096); temperature 1.0 as in the
earlier diagnostics rather than a deployment setting.
