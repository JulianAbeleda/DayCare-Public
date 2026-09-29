# Public tool-use benchmarks: stock Nemotron 3 Nano 4B vs the adopted run-5 adapter

Status: **DONE 2026-09-29** (Finding below). Was: PREDECLARED 2026-09-28 (H + P committed before any benchmark request was sent; the serving parity check
below ran first because it decides whether the benchmarks measure the trained adapter at all).

**Summary (read with Addendum A):** no measurable change on either benchmark. BFCL non-live AST pooled +1.0
[-1.0, +3.0], multi_turn_base +4.5 [-1.5, +10.5]; When2Call macro F1 -2.5 [-6.4, +1.2]. The When2Call
tool-hallucination increase in the Finding (+7.0 at one greedy sample) did not survive resampling: at T=1 x 10 it is
+0.0 [-2.2, +2.2] (Addendum A).

This is a DayCare LoRA RLVR feature (post-tool calculator repair/relay + blocked-tool behaviour) for Nemotron 3 Nano
4B. Adapter: [run 5](rloo-posttool-calculator-r5.md), adopted by owner exception, frozen in
`<runs>/posttool-adopted-001/` (`adapter.gguf` sha256 49859a3c..., raw 4ba84282...). Base: the BF16 GGUF of
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, sha256 2126a5e8... (= run 5's `model_sha256`). Scripts and raw outputs
outside Git: `<scratch>/bench-r5/`.

## Hypothesis

The adapter is LoRA r32 on the final MLP block only, KL-anchored (beta 0.03), trained on **later-turn** GameTerm
states: the calculator rejected a call (repair), the tool result is the answer (relay), or GameTerm refused a
terminal/file call (blocked: ask for authority at most once, then say it is blocked; no circumvention). Both public
benchmarks below test mostly the **first-turn decision** (call a tool? which one? ask? refuse?), which run 5 never
trained. So the main prediction is **small effects**, with these directions:

- **When2Call `cannot_answer`** (no suitable tool): may rise; the blocked training rewards "say you cannot, do not
  reach for another tool". Tool-hallucination rate (a tool call where none fits) may fall. Expected size: a few
  points; the CI likely includes 0 at n = 100.
- **When2Call `tool_call`**: could fall slightly (run 5's known cost: calculator call rate -5.5 [-9.4, -1.8] points on
  numeric items, i.e. a small drift away from calling tools). **`request_for_info`**: no trained signal; no change.
- **BFCL non-live AST** (simple / multiple / parallel / parallel_multiple): tool selection breadth and argument
  filling; untrained. Prediction: no worse than stock by more than ~3 points in any category (the regression risk
  named in the brief); `irrelevance` (no function should be called) may rise, as `cannot_answer` above.
- **BFCL multi_turn_base**: the closest to training (the model sees tool results and continues), so the only place a
  gain from relay behaviour could show; predicted no worse, possibly better.

Falsified (for the "small effect, no regression" claim) by any category whose paired 95% CI lies entirely below
-3 points (a regression) or entirely above +3 (a real transfer). Reported honestly either way, per category.

## Process

### 0. Serving parity (done before predeclaring; `<scratch>/bench-r5/parity/`)

The benchmarks run on llama-server b9592 (llama.cpp, CUDA build, `--lora adapter.gguf`,
scale 1 x alpha/rank from the GGUF); run 5 was trained and gated on DayCare's tinygrad sampler (tinygrad-arkey exp
bbf307f83, pinned worktree `<worktree>`). Check: 9 held-out tasks-003 states (3 repair, 3 relay, 3 blocked,
10.5k-token GameTerm prompts, envelope-002 incl. its logit bias), the exact same token ids to both stacks, greedy
(tinygrad: Gumbel-max at T = 1e-4; llama-server `/completion`, T = 0), 384 tokens; tinygrad loaded `adapter.xml`
and `check_raw` confirmed it equals the raw npz bit for bit.

| Arm | Identical continuations | Divergences |
|---|---|---|
| stock | 9 / 9 (up to 384 tokens) | none |
| adapter | 6 / 9 | 3, at tokens 8, 43, 156; each a near-tie: tinygrad's token is llama-server's #2, 0.007 / 0.013 / 0.033 nats behind #1 |

So llama-server + `adapter.gguf` serves the trained policy; the only gap is bf16 kernel numerics at near-ties (two
stacks, same policy), and the adapter visibly acts in llama-server (adapter vs stock continuations differ on all 9).

### 1. Serving (both arms identical except `--lora`)

`llama-server -m <base BF16> [--lora posttool-adopted-001/adapter.gguf] --jinja -ngl 99 -c 65536 -np 4
--temp 0 -n 4096 --port 8090`. Thinking on (the template's default, as in GameTerm; reasoning returned in
`reasoning_content`, not scored). `-n 4096` = GameTerm's per-turn max_tokens. Server default temperature 0 applies to
requests that send none (When2Call's script sends none; its default would otherwise be llama.cpp's 0.8 sampler);
BFCL sends its default 0.001. One sample per item; 4 parallel slots (batching can flip near-ties; same for both
arms). GameTerm's em-dash logit bias is **not** applied (a GameTerm style setting; the benchmark clients do not send
it). Arms run back to back, stock first.

### 2. When2Call (github.com/NVIDIA/When2Call ecc8d42, Apache-2.0)

- **Mode: the official LLM-as-judge generation eval** (`evaluation/llm_as_a_judge/`), 300 items
  (`when2call_test_llm_judge.jsonl`: 100 each tool_call / request_for_info / cannot_answer; 17 of the cannot_answer
  items have no tools). The MCQ log-probability mode cannot run on llama-server b9592 (no prompt/echo logprobs on
  `/v1/completions`, llama.cpp issue 27174), so it is not run.
- Inference: `run_openai_inference.py` verbatim except `content or ""` (a crash guard for an empty content with no
  tool call). Its classification of a native tool call is deterministic (`message.tool_calls` present -> the call).
- **Judge: substitute.** The paper used GPT-4o(-mini); none is configured here. Judge = the official `JUDGE_PROMPT`
  (4 classes) applied by a fresh Claude subagent, items from both arms shuffled together, arm hidden, one label per
  item. Responses that are native tool calls are labelled `tool_call` without the judge (exactly the class the
  official judge prompt defines). Secondary, judge-free: the regex classifier
  `When2Call/local_eval/classify_response.py` (92.7-100% on the dataset's own reference answers); reported for
  robustness only.
- Metrics: per-class accuracy (the official `aggregate_llm_as_a_judge_results.py`), per-class F1 over the 4 predicted
  labels, macro F1 over the 3 gold classes, **tool-hallucination rate** = cannot_answer items answered with a tool
  call (all 100; and the 17 no-tool items). Differences adapter - stock: paired bootstrap over items, stratified by
  gold class, 10,000 resamples, 95% percentile CI.

### 3. BFCL (bfcl-eval 2026.3.23 = BFCL v4 data; gorilla 6ea5797 for reference)

- Generic OpenAI handler, native function calling: `OpenAICompletionsHandler` via the registered FC entry
  `openbmb/MiniCPM-SALA-FC` (handler + `underscore_to_dot`, nothing model-specific; the model name in the request is
  ignored by llama-server), `OPENAI_BASE_URL=http://127.0.0.1:8090/v1`, `--num-threads 4`, default temperature 0.001.
  The handler strips `reasoning_content` from history (thinking on). One `BFCL_PROJECT_ROOT` per arm.
- Categories: simple_python (400), simple_java (100), simple_javascript (50), multiple (200), parallel (200),
  parallel_multiple (200), irrelevance (240), multi_turn_base (200). `bfcl evaluate` scores (AST checker; multi-turn
  state/response checker). Per-item pass/fail from the score files; paired bootstrap per category as above; summary =
  pooled non-live AST items (simple + multiple + parallel + parallel_multiple).
- Items that error (context overflow, server error) count as failures in both arms and are reported.

### Stop rule

No reruns with changed settings after seeing results. A crash is fixed and the affected arm re-run in full.

## Log

- **2026-09-28 22:12 EDT stock arm** (`run_arm.sh stock`): When2Call 300/300 answers collected by 22:24 (not yet
  judged); BFCL generate: all 7 non-live categories complete (1,390 items), multi_turn_base running.
- **BFCL "Failed to decode" audit** (raised in review: 316 of ~1,475 multi-turn steps in the log;
  `bfcl_decode_audit.py`, raw `inference_log`, stock, 151 entries so far): 337 of 1,520 steps; **337 / 337 are
  plain-text assistant replies** (final answers, summaries, clarifying questions); **0** contain tool-call-like text
  (`<tool_call>`, `{"name": ...}`, `func(arg=...)`); llama-server log: 0 parse errors. In BFCL's FC handler a reply
  without `message.tool_calls` raises in `convert_to_function_call`, and that is its designed end-of-turn path
  ("Proceed to next turn"): 341 of 534 turns end that way. 20 random ones hand-checked: all genuine end-of-turn
  answers. Non-live: 0 tool-call-like text in 1,390 items; 7 empty responses (irrelevance 3, parallel 2,
  parallel_multiple 1, simple_python 1) all hit the declared 4,096-token cap inside thinking (a model failure under
  the declared settings, scored as a fail). **Verdict: no parser mismatch; the generic handler is correct for
  Nemotron via llama-server `--jinja`; no Nemotron handler is needed and the stock run stands.**
- **Adapter arm** 23:42 EDT: When2Call 300/300 by 23:51; BFCL generate running. Stock BFCL scored (`bfcl evaluate`).
  Parallel categories checked for a server default (llama-server `parallel_tool_calls` defaults to the template's
  capability, not false): stock returns 2-5 calls on 57-60% of parallel items, so multi-call parsing works; the
  1-call answers are the model's.
- **When2Call judged** (4 fresh blind subagents, 302 free-text replies of both arms shuffled, arm hidden; 298 native
  tool calls labelled `tool_call` without the judge). 2 stock replies were empty (content "" and no call: the
  4,096-token cap inside thinking); one judge labelled one of them `tool_call`, the other `cannot_answer`; both
  **overridden to `empty`** (wrong in every class) by the scorer, since neither is a call or a refusal.
  Scored by `w2c_score.py` (numbers in the Finding).
- **Adapter BFCL** generated 23:51-00:59 EDT, scored. Decode audit, full multi_turn_base: stock 506 of 2,049
  steps, adapter 564 of 2,219, **0 tool-call-like text in either arm** (all plain-text end-of-turn replies). Non-live
  empty replies (4,096 cap in thinking): stock 7, adapter 2; tool-call-like text in a text reply: stock 1, adapter 2
  (checked: prose mentioning a field name). Every generate/evaluate run exited 0; no server errors.

## Finding

**Hypothesis (small effects, no regression) holds on BFCL and on 3 of 4 When2Call measures, but fails on When2Call
tool hallucination: the adapter calls an unsuitable tool more often (+7.0 points [+3.0, +12.0]), the opposite of the
predicted direction.** No BFCL category moves beyond the declared +-3-point band with its whole CI.

**Process.** As predeclared (DayCare (private commit)): llama-server b9592, thinking on, `-n 4096`, stock and adapter back to back;
When2Call's official generation eval (300 items), judged blind with its judge prompt by Claude subagents (substitute
for GPT-4o); BFCL v4 via its generic OpenAI FC handler; paired item bootstrap, 10,000 resamples (stratified by class
/ category). Serving parity (Process 0): the served adapter is the trained policy up to near-tie numerics.

| Benchmark / category | n | Stock | Adapter | Diff (95% CI) |
|---|---:|---:|---:|---|
| When2Call `tool_call` acc (call when a tool fits) | 100 | 94.0 | 93.0 | -1.0 [-5.0, +3.0] |
| When2Call `request_for_info` acc (ask for missing args) | 100 | 65.0 | 65.0 | +0.0 [-6.0, +5.0] |
| When2Call `cannot_answer` acc (no suitable tool) | 100 | 45.0 | 41.0 | -4.0 [-12.0, +4.0] |
| When2Call macro F1 (3 gold classes) | 300 | 68.8 | 66.3 | -2.5 [-6.4, +1.2] |
| **When2Call tool hallucination** (cannot_answer items answered with a call; lower is better) | 100 | 17.0 | 24.0 | **+7.0 [+3.0, +12.0]** |
| BFCL simple_python | 400 | 95.0 | 95.2 | +0.3 [-1.5, +2.0] |
| BFCL simple_java | 100 | 57.0 | 55.0 | -2.0 [-8.0, +3.0] |
| BFCL simple_javascript | 50 | 56.0 | 68.0 | +12.0 [+2.0, +24.0] |
| BFCL multiple | 200 | 95.0 | 95.0 | +0.0 [-3.0, +2.5] |
| BFCL parallel | 200 | 53.5 | 53.0 | -0.5 [-8.0, +7.0] |
| BFCL parallel_multiple | 200 | 54.0 | 58.0 | +4.0 [-2.5, +10.5] |
| **BFCL non-live AST pooled** (the 6 rows above) | 1,150 | 75.7 | 76.7 | +1.0 [-1.0, +3.0] |
| BFCL irrelevance (no call is correct) | 240 | 74.6 | 76.2 | +1.7 [-1.7, +5.0] |
| BFCL multi_turn_base | 200 | 26.5 | 31.0 | +4.5 [-1.5, +10.5] |

Accuracy in %; CIs are per row, not corrected for the 14 comparisons (simple_javascript's lower bound +2.0 at n = 50,
with 7 wins / 1 loss, is the kind of result one of 14 rows produces by chance; it does not pass the declared +3 rule).

- **Against the predictions.** `cannot_answer` was predicted to rise; it is flat to down, and the hallucination rate
  rises. The 7 cannot_answer items that flipped to a call show one pattern: the adapter **acts instead of asking or
  declining**, filling what it does not know (Toto tickets: buys 1 ticket for a date it picked, where stock asked how
  many; "set a reminder in a minute": `set_alarm` at an invented timestamp; a $200 payment to "Diego" with the amount
  as 20000; "play baby shark": `set_volume 100`; "get the dashboard": calls `add_custom_dashboard`). The no-tools
  subset (17 items) stays at 0 hallucinations in both arms. Consistent with the known run-5 costs (drift toward
  finishing fast; shorter turns from the length term) rather than with the blocked-tool training, which only covered
  refusals GameTerm itself returns, not "no tool fits" on the first turn.
- **Selection breadth** (the named regression risk): no loss. Pooled non-live AST +1.0 [-1.0, +3.0]; `multiple`
  identical (95.0); `request_for_info` identical.
- **Multi-turn** (closest to training): +4.5 [-1.5, +10.5], in the predicted direction, not significant.
- **Irrelevance vs hallucination**: BFCL irrelevance (+1.7, n.s.) and When2Call hallucination (+7.0, worse) disagree
  in direction; BFCL's irrelevance items offer an unrelated function for a knowledge question, When2Call's offer a
  near-miss tool for an action request, which is where the adapter over-acts.
- **Robustness.** Judge-free regex classifier (secondary): cannot_answer 48 -> 45, request_for_info 54 -> 55, tool_call
  94 -> 93, same direction. BFCL decode "failures" are end-of-turn text in both arms (Log), not parser mismatch.
- **Limits.** One greedy sample per item; When2Call's judge is a substitute (the paper's numbers are not reproduced
  and are not comparable); the MCQ log-prob mode was not runnable on llama-server; 4-slot batching can flip near-ties
  in either arm; no GameTerm em-dash bias.

**For the public write-up:** report BFCL as "no measurable change in tool selection (pooled +1.0 [-1.0, +3.0]); multi-
turn +4.5 [-1.5, +10.5]" and When2Call's hallucination increase as a known cost next to G3, not as a footnote.

## Addendum A: When2Call tool hallucination at temperature 1 (robustness of the +7)

**Why.** The +7.0 [+3.0, +12.0] rests on one temperature-0 sample per item; single-sample T=0 flips on G3 proved
fragile when resampled.

**Hypothesis (predeclared, committed before measuring).** The gap holds at T=1: adapter hallucination rate minus stock
hallucination rate > 0, with a paired, item-clustered bootstrap 95% CI excluding 0. Falsified if the CI includes 0
(then: the +7 was a T=0 near-tie artefact, not a stable over-acting tendency).

**Process.**
- Items: the 100 `cannot_answer` items of `when2call_test_llm_judge.jsonl` (same file, same tool preprocessing as
  `w2c_infer.py`). 10 samples per item per arm = 1,000 generations per arm.
- Serving: identical to Process 1 (same llama-server b9592, base BF16 GGUF, `--lora posttool-adopted-001/adapter.gguf`
  for the adapter, `--jinja`, thinking on, `-n 4096`), except `-np 8 -c 131072` (same 16,384 tokens per slot, more
  slots for throughput) and the request sends `temperature: 1.0` plus a per-sample `seed` (item, k), identical seeds
  across arms. Other samplers stay at llama-server defaults (top_k 40, top_p 0.95, min_p 0.05) in both arms. Stock
  first, then adapter.
- Scoring, judge-free: a sample is a hallucination iff the response carries a native tool call
  (`message.tool_calls`), the same deterministic rule that produced every `tool_call` label in the Finding; the regex
  classifier `classify_response.py` splits the rest (reported only as a distribution). Agreement check: regex labels on
  the original T=0 samples vs the judged labels.
- Statistics: per-arm rate = mean over 1,000 samples; paired difference = mean over items of (adapter item rate -
  stock item rate); 95% CI by bootstrap resampling items (clusters of 10 samples, both arms together), 10,000
  resamples, percentile. Also reported: distribution of per-item rates in each arm and of per-item differences (is the
  gap a few items or broad), and both arms' rates on the 7 items that flipped at T=0 (indices 1, 12, 13, 14, 50, 55,
  56 in the 300-item file).
- Script: `<scratch>/bench-r5/w2c_t1.py` (sampler) + `w2c_t1_score.py`; outputs `out/t1/{stock,adapter}.jsonl`.

### Finding A

**Hypothesis falsified: at T=1 the gap is zero.** Hallucination rate 20.4% (204/1,000) in both arms; paired
difference **+0.0 points [-2.2, +2.2]** (item-clustered bootstrap, 100 items x 10 samples). The T=0 +7.0 was
near-tie flips, not a stable over-acting tendency.

**Process.** As predeclared (DayCare (private commit)); stock 08:35-09:01, adapter 09:01-09:24 EDT, 1,000/1,000 each, 3 adapter
samples hit `-n 4096` (0 stock). Adapter applied: same seeds give different text on 941/1,000 samples, mean
completion 477 vs 559 tokens (run 5's known shortening), and `/lora-adapters` on the same command line lists
`posttool-adopted-001/adapter.gguf` at scale 1.0.

- **Judge-free = judged on hallucination.** On the original T=0 samples the regex labels give 17 / 24 hallucinations,
  identical to the judged labels (a call is a native `tool_calls`, deterministic in both); overall label agreement
  272/300 stock, 269/300 adapter (disagreements are in the free-text split, which the hallucination rate never uses).
  Regex split of the T=1 non-calls is also flat: cannot_answer 378 vs 377, request_for_info 214 vs 207, direct 204 vs 212.
- **Per-item rates are bimodal and the same in both arms.** Items at 0/10: stock 65, adapter 63; at 10/10: 10 and 10;
  the rest spread between. Per-item differences: 74 items equal, 14 adapter higher, 12 lower; largest +5/10 (item 83),
  +3 (53, 78); largest drop -6 (item 12). No broad shift and no concentrated one.
- **The 7 items that flipped at T=0** (stock no call -> adapter call), stock / adapter calls out of 10: item 1 4/3,
  12 10/4, 13 4/5, 14 3/1, 50 7/5, 55 2/3, 56 1/0. Stock sums 31/70, adapter 21/70: every one is a near-tie item
  where both arms call sometimes, and stock calls at least as often on 5 of 7. The T=0 examples (Toto tickets, invented
  alarm time, $200 as 20000) are real behaviours of the policy family, not adapter-specific ones.
- **Limits.** T=1 with llama-server's default top_k/top_p/min_p is a different operating point from GameTerm's; one
  run per arm; 8 concurrent slots.

**Correction to the Finding above:** When2Call tool hallucination is **no measurable change** (T=1 x10: +0.0
[-2.2, +2.2]), not a known cost. Public write-up: report BFCL and When2Call alike as no measurable change; mention the
T=0 +7 only as an example of why single greedy samples are not evidence. Scripts `<scratch>/bench-r5/w2c_t1.py`,
`run_t1.sh`, `w2c_t1_score.py`; outputs `out/t1/`.
