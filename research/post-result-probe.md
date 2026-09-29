# Post-result probe: what Nemotron does after a calculator result

> Published copy of a working record (inference only, stock model), lightly edited for publication. `<runs>/`
> stands for the private run-output directory; the analysis scripts named here are not published. GameTerm is
> [gameterm.arkey.ai](https://gameterm.arkey.ai); the experiments used a beta build that is not public.

Date: 2026-09-23. Inference only; no weights changed.

## Question

Full-harness Countdown runs show Nemotron 3 Nano 4B calling the calculator and
then failing: the r32 readiness run
(`countdown-diagnostic-r32-readiness-008`) had 24 trajectories with a
successful calculator call, of which 1 was correct, 16 kept calling until the
budget ran out, and 7 answered wrongly. Is the failure *reading the result*, or
*deciding what to do when the result is not the target*?

## Method

The probe script (not published) prefills one calculator exchange and samples
only the next assistant turn. This separates result use from call selection.
Items are the 16 Countdown tasks never used for SFT (`rl-tasks.json` and
`probe-tasks.json` in `countdown-recipe-prep-003`), each in three conditions:

- **hit**: an exact-solver expression whose real calculator receipt equals the
  target;
- **miss**: a valid all-numbers expression whose receipt is not the target;
- **rejected**: a schema-invalid call answered with GameTerm's TC-060 text for
  a malformed non-file call (checked against `result.rs` at freeze).

The production 35-tool envelope and native template are used through
llama-server directly, not through GameTerm's turn loop; four seeds;
temperature 1.0; 1,024 tokens; thinking off. Calculator receipts come from the
GameTerm calculate runner. Runs:
`<runs>/post-result-probe-002/`.

## Result (thinking off, 64 samples per cell)

| Condition | Arm | Correct final | Calls again | Other |
|---|---|---:|---:|---|
| hit | stock | **64** | 0 | — |
| hit | r32 SFT | 59 | 0 | 5 truncated |
| miss | stock | 8 | 36 (2 repeat the same call) | 20 wrong finals, 9 truncated among them |
| miss | r32 SFT | 0 | **64** | 3 truncated |
| rejected | stock | 0 | 64 | — |
| rejected | r32 SFT | 0 | 64 | — |

## Interpretation

1. **Reading a correct result is not the problem.** Stock answers with the
   called expression and stops on every hit sample. Training "result → final
   answer" would target a behavior that already works.
2. **The failure is after a miss.** Stock either tries a new expression
   (diverse, rarely repeated) or commits to a guess that is usually wrong.
   Finding the next expression is Countdown search, not tool use.
3. **The r32 adapter learned "after a miss, call again" without the search.**
   Its training rows paired every wrong attempt with the solver's correct next
   call. On unseen tasks it always calls again, copies the solver's nested
   parenthesis style, and sometimes degenerates (arguments up to 2,037
   characters). Iterated, this is the full-harness loop.
4. After a rejected call both arms retry the calculator; neither switches to an
   unrelated tool.

## Consequences for training

- Do not train more "retry after a miss" targets unless the model can produce
  the retry. A retry budget with an honest stop is safer than an unconditional
  retry.
- Solver-written traces teach format, not search. Prefer self-generated
  successful trajectories, or test thinking before training.
- A single-turn probe cannot score recovery after a miss; that needs a
  multi-turn run.

## Thinking on (stock, 4,096 tokens)

Same frozen items and seeds, `enable_thinking=true`
(`post-result-probe-think-001`). The control is thinking off at the same
4,096-token budget (`post-result-probe-off4096-001`), so the budget is not
the difference.

| Condition | Thinking off | Thinking on | Paired wins/losses | Task-clustered 95% CI of the gain | Tasks improved |
|---|---:|---:|---:|---|---:|
| hit | 64 | 64 | — | — | — |
| miss | 12 | **42** | 34/4 | +31.2 to +62.5 points | 13/16 |
| rejected | 0 | 15 | 15/0 | +12.5 to +35.9 points | 9/16 |

With thinking on, every miss sample that produced a final answer was correct
(42/42); the other 22 were 12 empty turns that exhausted 4,096 tokens while
reasoning, 9 new calculator calls, and 1 other-tool call. Median reasoning
length was about 2,000 characters after a miss and 330 after a hit. After a
rejection, thinking often solved the task directly instead of retrying.

**Conclusion:** the post-miss failure is a search failure that thinking
largely fixes at inference time. That is the stronger lever than further
thinking-off SFT. The cost is latency and token budget; the empty
budget-exhausted turns need a stop rule.

## Production tool-result format (GameTerm wire envelope)

GameTerm does not show the model `expr = value`; it sends its
`WireToolResult` JSON, e.g.
`{"transport":"succeeded","stdout":"(70 * 70) / (70 - 1) = 71.0144927536","stderr":"","exit_code":null,"truncated":false}`,
with `content: null` on the calling assistant turn. The results above, and
every Countdown SFT row, used the bare receipt: a train/serve mismatch.
`daycare/harness/gameterm_wire.py` now owns that format; the probe rebuilds
16/16 captured GameTerm post-tool requests with identical messages. Reruns at
4,096 tokens (`post-result-probe-wire-{off,on}-001`; the rejected text is the
earlier GameTerm wording in both):

| Condition | Off, bare | Off, wire | On, bare | On, wire |
|---|---:|---:|---:|---:|
| hit | 64 | 63 | 64 | 63 |
| miss | 12 | **2** | 42 | **33** |
| rejected | 0 | 5 | 15 | 16 |

With thinking off, the wire format makes the miss much worse: 21 of 64 miss
samples answered with the expression that had just missed the target (3 with
the bare receipt). The envelope appears to hide that the result is not the
target. Thinking on still recovers most of it (33/64). Two consequences:
future SFT/RL data must use the wire format, and the envelope's effect on
result reading is itself worth testing on the GameTerm side.

## Limits

16 tasks × 4 seeds; the hit condition is easy because the answer is the called
expression; single next-turn only (no multi-turn recovery); llama-server rendering was not yet compared byte-for-byte with
GameTerm's later-turn requests. Four thinking-off stock outputs contained a
stray `</think>`.
