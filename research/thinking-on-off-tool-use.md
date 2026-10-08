# Thinking on vs off for tool use: what the records show

> Published copy of a working record (a documentation review; no new measurement), lightly edited for
> publication. `<scratch>/` stands for unpublished analysis outputs; "private commit" marks a commit in the private
> development repository.

Status: **DONE 2026-09-29, documentation review only.** A new thinking-off run of the public benchmarks was planned
(stock Nemotron 3 Nano 4B, `--chat-template-kwargs '{"enable_thinking":false}'`, which makes the template prefill
`<think></think>`) and **cancelled by the maintainer before any H+P was committed or any request sent**. Nothing below is a new
measurement except a CPU-only look at existing BFCL score files.

## Hypothesis (the maintainer's claim, as tested against the records)

"For tool CALLING, the fix was just turning thinking on (vs off); the remaining issue was interpreting the tool
result; our benchmarks should confirm this." Two parts:

- **C1.** Thinking on (vs off) fixed first-call tool use (whether to call, which tool, arguments).
- **C2.** What remains weak, even with thinking on, is using the tool result (post-tool interpretation).

## Process

Read-only: every thinking-on vs thinking-off comparison in `research/` and its git history, the published
`research/nemotron-tool-calling` and `research/nemotron-tool-selection` records, and
the thinking-on stock arm of [public-benchmarks-r5](public-benchmarks-r5.md) (`<scratch>/bench-r5/out/stock`).

## Finding

**C1 is not shown by the records; C2 is only half right.** Documented thinking-off -> on effects are larger on
post-tool turns than on first calls, and no public benchmark was ever run with thinking off.

### Existing measurements, by what they measure

| Source (commit) | What is measured | Thinking off | Thinking on |
|---|---|---:|---:|
| [countdown-thinking-e2e](countdown-thinking-e2e.md), retention gate (private commit) | **First call**, `selection` family: right tool in GameTerm's 35-tool envelope, T=0, 16 items | 11/16 | 15/16 (5 wins / 1 loss) |
| same, numeric families (audited) | end-to-end answer, mostly first call + relay | 61/68 | 67/68 |
| same, clarification | ask instead of assume | 0/8 | 1/8 |
| same, Countdown end to end (private commit) | whole episode, 64 = 16 tasks x 4 seeds | 15/64 | **46/64** |
| same, episodes that called the calculator | tool **use** | 31/64 | 4/64 |
| [post-result-probe](post-result-probe.md) (private commits) | **Post-tool**: next turn after a prefilled calculator result, bare / wire | | |
| - hit (result is the answer; relay) | | 64 / 63 | 64 / 63 |
| - miss (result is not the target) | | 12 / **2** | 42 / **33** |
| - rejected (GameTerm refused the call) | | 0 / 5 | 15 / 16 |
| [rloo-posttool-calculator](rloo-posttool-calculator.md) headroom (thinking on only) | post-tool relay, held-out pass@1 | - | 0.99 |
| same | post-tool miss / repair / empty, held-out pass@1 | - | 0.58 / 0.83 / 0.34 |
| Public `nemotron-tool-selection` (2026-09-20, thinking **off**, cce7296) | first-call selection, 320 items, full GameTerm harness | base 45.0%, rank-4 adapter **94.7%** | not measured |

Thinking-on stock on public benchmarks ([public-benchmarks-r5](public-benchmarks-r5.md), no thinking-off arm): BFCL v4
simple_python 95.0, multiple 95.0, simple_java 57.0, simple_javascript 56.0, parallel 53.5, parallel_multiple 54.0
(non-live AST pooled 75.7), irrelevance 74.6, **multi_turn_base 26.5**; When2Call tool_call 94, request_for_info 65,
cannot_answer 45, macro F1 68.8, tool hallucination 17/100 at T=0 and 20.4% at T=1 x10.

### What this supports

1. **Thinking helped first-call selection a little, on 16 items** (11 -> 15/16, one real loss). The large first-call
   fix on record is not thinking: under the full GameTerm harness base selection was 45% with thinking off, and a
   rank-4 final-MLP adapter (thinking off) took it to 94.7% (+49.7 [+45.3, +54.1]). Thinking-on selection in that
   320-item suite was never measured, so "thinking alone fixes harness selection" is untested, not refuted.
2. **In Countdown, thinking replaced tool use rather than fixing it**: calculator episodes fell 31 -> 4 of 64 while
   accuracy tripled. That is a reasoning effect, not a tool-calling fix.
3. **Thinking's largest documented effect is post-tool**: after a miss 12 -> 42/64 (bare) and 2 -> 33/64 (GameTerm
   wire format), after a rejection 0 -> 15/64; task-clustered CIs exclude 0. So the claim's split (thinking fixed
   calling, results remain) runs against the records: thinking fixed much of result use too.
4. **The remaining weakness is narrower than "interpreting the result"**: relaying a result that answers the
   question is already solved (64/64; held-out pass@1 0.99, thinking on). What stays weak with thinking on is the
   turn after a result that does **not** settle the question (miss 33/64 in wire format, rejection 16/64, empty
   turns that burn the 4,096-token cap). That is recovery/search after a result, which is what run 5 trained.
5. **Public benchmarks, thinking on:** strong where one call answers the request (simple_python / multiple 95),
   weak on abstention and asking (cannot_answer 45, hallucination 17-20%, request_for_info 65), on multi-call
   (parallel 53.5) and on multi-turn (26.5). The 26.5 vs 75.7 gap is consistent with C2 but does not isolate it:
   multi_turn_base scores final environment state over 3-5 user turns with long tool lists, so a failure can be a
   wrong first call, a wrong argument, a missing step or a misread result. A quick CPU heuristic over the 147 failed
   items (first failing turn: model's first call not among ground-truth functions ~40, right function then fails
   later ~31, stops after one step ~14, the rest unassignable) could not separate these reliably (exploratory calls
   such as `ls` count as "wrong"); not reported as a result. Example `multi_turn_base_0`: `find` returned no match
   and the model ended the turn instead of creating the folder, a post-result failure.

### What it does not support

- That thinking **is** the first-call fix: no thinking-off BFCL/When2Call run exists (cancelled), and the only
  on/off first-call comparison is 16 items.
- That our public benchmarks confirm the split: they have no off arm, and BFCL multi-turn mixes call errors with
  result errors.

**Verdict:** partly right. Thinking-on is the right baseline and does help first-call selection modestly (11 -> 15
of 16), but the records show its big wins on post-tool turns (miss, rejection) and a thinking-off adapter as the
big first-call selection fix. The remaining gap is recovery after an unhelpful or refused result (and abstention /
asking on the first turn), not relaying a result. Confirming C1 on public benchmarks needs the cancelled
thinking-off arm.
