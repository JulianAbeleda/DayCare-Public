# RLOO on the post-tool calculator turn (one tinygrad stack)

Status: **predeclared** 2026-09-27. The maintainer approved the draft as written, with the decisions recorded
under Process, before any counted update.
This is a DayCare LoRA RLVR feature for Nemotron 3 Nano 4B on the GameTerm harness.

## Hypothesis

Calling the calculator is not where Nemotron 3 Nano 4B fails on GameTerm; what it does with the result is. The
stock headroom measurement below sharpens that: with thinking on, stock relays a result that answers the
question almost perfectly (word problems needing rounding, remainders, unit mapping: held-out pass@1 0.99), but
when the result does not settle the question (a Countdown miss, a call GameTerm rejected, the historical
empty-answer states) it often reasons until the 4,096-token turn cap and answers nothing, or answers wrongly.
Pass@8 is 17-41 points above pass@1 on those states, so the policy can already produce the right behavior; RLOO
should make it more likely.

**H:** sequence RLOO on post-tool episodes (the model's own call or the probe's call, the real calculator
result in GameTerm's wire envelope, thinking on, reward 1 only for a correct non-empty final answer) raises
held-out success on the states where the result does not settle the question (miss, repair, empty).

Falsified if the task-clustered bootstrap 95% interval of the held-out success difference (adapter - stock) on
those states has a lower bound at or below zero (G1). Adopted only if G1, G1b (no relay regression), G2 (fewer
empty answers after a tool result) and G3 (retention: zero plain-answer losses) all pass.

Why this might work where Countdown RLOO did not (v1 -2.7, v2 -0.7 points): Countdown RLOO trained the first
turn, where the reward scored any tool call as 0; here the trained turn is the one after the tool result, tool
calls are neither rewarded nor penalized, and a failure mode (running out of budget after a result) is
directly rewarded against. Why it might not: miss and empty states are Countdown search, which v1/v2 did not
improve at this adapter capacity; 102 updates on a last-block LoRA may not shorten reasoning; the repair and
empty held-out cells are small (61 and 4 states).

## Process

### Episodes (built and tested; `daycare/nursery/rloo_posttool.py`, `daycare/harness/posttool.py`)

- A state is a GameTerm later-turn request: the 35-tool envelope (`rft-posttool-001/envelope.xml`, thinking on),
  the user's request, the model's own `calculate` call and GameTerm's real result in the wire envelope.
- The episode samples the next assistant turn (4,096 tokens, temperature 1.0, full vocabulary). If it calls
  `calculate` again, the call runs through GameTerm's calculator, the next request is rendered as GameTerm
  would send it, and sampling continues, up to 8 sampled turns (GameTerm allows 32 tool iterations).
- Loss: every sampled token of every turn; prompts, history and tool results are prefilled, never sampled, so
  they are masked by construction (tiny-config test: loss tokens == sampled tokens, parity exact).
- Reward: 1 only for a completed turn, reasoning closed, no pending call, and a final answer the task's rule
  accepts (Countdown verifier; numeric `<answer>` exact match). No bonus or penalty for tool calls; a call still
  pending at the 8-turn cap scores 0 because there is no answer.

### GameTerm parity (evidence: `<runs>/posttool-parity-001/`)

| Check | Result |
|---|---|
| Messages: request N == request N-1 + one wire call (`content: null`) + one wire result | 2,329/2,329 captured 4B requests |
| Calculator: GameTerm `calculate` (runner from the GameTerm beta build) reproduces the captured result bytes | 2,251/2,251 calculate results |
| Parser: llama-server's argument string for calculate round-trips | all calculate calls (6 non-calculate array-argument calls differ in spacing; not used) |
| Token count (DayCare render + BOS) == llama-server `usage.prompt_tokens` | 2,329/2,329 |
| Exact text (/apply-template) and ids (/tokenize) vs llama-server b9592 (GameTerm's build), thinking on, post-tool states incl. rejections and 2-exchange histories | 65/65 text, 65/65 ids |

Not proven: the model's raw tool-call text is parsed by DayCare's parser, not llama-server's (proven only on
template-rendered calls; whitespace variants the model might emit are unchecked); GameTerm on the macOS client has an
osascript fallback for arithmetic fend refuses (Linux and this runner have none); the one-stack sampler's
decode numerics differ from llama.cpp's (the same policy, a different kernel stack); a call to any tool other
than `calculate` ends the episode with reward 0 (GameTerm would run it; 1-6% of episodes in the headroom run).

Found on the way: GameTerm's requests start with BOS (`tokenizer.ggml.add_bos_token`, confirmed by every
captured prompt count). The Countdown one-stack runs (v1, v2) sampled prompts without it; post-tool prompts
include it.

### Tasks and splits (`posttool-tasks-002`; manifest in `manifest.xml`)

Task-level split (seed 20260926, 30% held out, stratified by category and kind); a state inherits its task's
split, so no held-out task appears in training in any category. The 132-item retention suite: zero overlap.
`-002` differs from `-001` only in the answer instruction of the 288 word problems: `-001` said "Finish with
<answer>the number</answer>" and stock copied the placeholder literally (28 relay turns in
`posttool-headroom-001`); `-002` says "End your reply with the final number alone inside answer tags, like
<answer>42</answer>". The own calls were re-harvested on `-002` (`posttool-harvest-002`).

| Category | What the turn must do | Source of the call | Train states | Held-out states |
|---|---|---|---:|---:|
| relay | map a large/decimal result back to the question (12 templates: round up, floor, remainder, cents, percent, gap to goal, cheaper unit price) | model's own first-turn call (209), else the obvious expression (79) | 203 | 85 |
| miss | Countdown: result misses the target, compare and keep searching | probe construction (the model's own first-turn calls were all hits, 16) | 68 | 28 |
| repair | GameTerm rejected the call (schema) or the calculator refused it | GameTerm's rejection of the model's own call (110), constructed (68), own natural (8) | 125 | 61 |
| empty | historical empty-answer-after-tool states (post-result probe, thinking on) | probe construction | 13 | 4 |
| **all** | | | **409** | **178** |

### Stock headroom (thinking on; `posttool-headroom-002`, 12 states per cell (4 for empty/held-out), 8 episodes each)

| Category | Train pass@1 | Train pass@8 | Held-out pass@1 | Held-out pass@8 | Empty answers after a tool result (train / held-out, of 96 / 96 or 32) |
|---|---:|---:|---:|---:|---|
| relay | 0.91 | 1.00 | 0.99 | 1.00 | 0 / 0 |
| miss | 0.71 | 0.92 | 0.58 | 0.75 | 22 / 32 |
| repair | 0.65 | 1.00 | 0.83 | 1.00 | 19 / 4 |
| empty | 0.59 | 0.92 | 0.34 | 0.75 | 21 / 12 |

Reading: with an unambiguous answer format, stock thinking-on already relays word-problem results almost
perfectly (relay held-out 0.99): the interpretation of a result that answers the question is not the failure.
The headroom is where the result does not settle the question: after a miss, a rejection, and in the historical
empty states, pass@8 exceeds pass@1 by 17-41 points. The dominant failure is a turn that reasons until the 4,096
cap and answers nothing (101 of 962 sampled turns hit the cap; `incomplete` is 54 of 192 miss episodes, 31 of
128 empty episodes). Wrong numbers are rarer (relay 9, repair 9). 12 states per cell is a small sample: these
are headroom estimates, not gates.

### Fixed settings (v2's unless noted)

- Nemotron 3 Nano 4B BF16; LoRA rank 32, alpha 64 on the final MLP block (`ffn_up`, `ffn_down`); last block only.
- Sequence RLOO, group 8, 4 states per update, Adam lr 2e-4 (v2's sweep), KL 1e-3, entropy 1e-3, clip 1.0,
  TIS cap 2, the 0.1-nat parity stop, temperature 1.0, 4,096 tokens per turn, cap 8 turns.
- Training states: the train split of all four categories (409 states), shuffled once (seed 20260924). Relay
  stays in training although it is near ceiling: it keeps the easy relay behavior anchored while the step
  pushes on the hard states (an open choice for review: training on miss/repair/empty only, 206 states).
- Updates: **102** (one pass over the training states). Change from v2 (20): v2 showed 20 updates of 32
  rollouts did not move holdout accuracy; one pass is the smallest schedule that sees every training state.
- Stack: tinygrad-arkey `exp` (revision recorded), DayCare runner from a detached worktree at a fixed commit.

### Measured iteration time (non-counted timing run, research/posttool-loop-iteration-speed.md)

Non-counted timing runs (`posttool-timing-001`/`-002`, then after the shared-prefix reuse and follow-up refill
levers `-003`/`-004`, tinygrad-arkey 4322ae9e2): 53.2 s, then **36.7 s per warm update** (prime 9.4 s, decode
24.4 s, parity + grad + Adam 2.9 s), first update 272 s, 52 episodes per minute; those updates drew short turns.
The headroom runs (11% of turns at the 4,096 cap) imply ~105 s for a typical update that contains one capped
chain. **Projected: 102 updates in ~2-3 h**, plus G1's evaluation (~1 h per arm). Details and the quality
checks (8/8 pass): [posttool-loop-iteration-speed.md](posttool-loop-iteration-speed.md).

### Decisions at approval (2026-09-27, the maintainer)

- The procedure is the draft's, unchanged.
- Other-tool calls: option (c). An episode that ends in a call to a tool other than `calculate` keeps reward 0
  (unchanged), and its rate is reported separately per training update and per evaluation arm (stock vs
  adapter), next to the capped-turn and empty-after-tool rates. Runner: DayCare `rates()` in
  `rloo_posttool.py`.
- Evaluation seeds: 20260925 for the stock arm, 20260926 for the adapter arm.
- Engineering added after approval (no setting changes): raw adapter checkpoints every 10 updates.
- Training run: `posttool-rloo-r1-001`, `--protocol rloo-posttool-calculator.md` (recorded as counted).

### Checklist and gates

- [x] P1. This file reviewed by the maintainer and marked predeclared before any counted update.
- [x] P2. Output-path check on the real model: one update, export, `verify` in a fresh process (bit-exact).
- [x] P3. Train 102 updates; stop on a parity violation, a non-finite value or a FloatingPointError.
- [x] G1. Efficacy: the held-out miss, repair and empty states (93), stock (zero adapter) vs adapter, 4 episodes
  per state and arm, same settings as training. Score = mean reward per state. Pass only if the task-clustered
  bootstrap 95% interval (50,000 resamples, clusters = source tasks) of the mean paired difference has a lower
  bound above zero. Relay is gated separately because it is at ceiling and would dilute the pooled estimate:
- [x] G1b. No relay regression: all 85 held-out relay states, same procedure; fail if the interval's upper bound
  is below zero.
- [x] G2. Empty answers after a tool result in G1's and G1b's episodes (a completed or truncated final turn with no content
  and no call): adapter strictly fewer than stock.
- [ ] G3. Retention (only if G1 passes): the 132-item GameTerm suite through HT-020, adapter vs stock, thinking
  on, temperature 0: zero plain-answer losses and no net real loss after hand audit of every disagreement;
  count empty answers after a calculator call.
- [ ] Stop rule: a failed gate ends the experiment; no rescue runs.

## Log

- **Stock arm, run before approval; valid for the counted comparison only if the predeclared procedure is
  unchanged** (2026-09-27). Zero adapter, DayCare (private commit) (`rloo_posttool sample --split heldout`),
  tinygrad-arkey 4322ae9e2, `posttool-tasks-002/states.xml`, 4 episodes per state, 4,096 tokens per turn, cap 8,
  temperature 1.0, sampling seed 20260925 (the stock seed of the Countdown R5 procedure; the adapter arm would use
  20260926). Output path checked first (`posttool-g1-stock-check-001`: 3 states, written and parsed).
  - G1 stock (`posttool-g1-stock-001`, 93 held-out miss/repair/empty states, 372 episodes, 592 s): mean reward
    **0.669** (repair 0.697 over 61 states, miss 0.670 over 28, empty 0.25 over 4). pass@4: repair 0.95, miss 0.82,
    empty 0.50.
  - G1b stock (`posttool-g1b-stock-001`, 85 held-out relay states, 340 episodes, 276 s): mean reward **0.953**.
  - G2 stock counts (final turn with no content and no call, truncated included): **56 of 372** G1 episodes
    (repair 21, miss 27, empty 8) and **4 of 340** G1b episodes.
  - Failure mix, G1: 42 incomplete (4,096-token cap), 24 other-tool calls (reward 0; GameTerm would run them),
    14 wrong numbers, 9 reasoning not closed, remaining Countdown verifier errors. If the maintainer changes the eval
    procedure (states, episodes, seed, budget, scoring), these are rerun.

- **G3 retention stock arm, run before approval; valid for the counted comparison only if the predeclared
  procedure is unchanged** (2026-09-27). `scripts/thinking_retention_gate.py` (freeze + run), the frozen 132-item
  suite through HT-020 (`calculator_study`), stock model (no adapter), thinking on, temperature 0, 4,096 tokens,
  server seed 42: `posttool-g3-stock-001`. Output path checked first: the protocol froze with the same envelope,
  suite, model, server and GameTerm hashes as `thinking-retention-on-002`, and the report path parsed that run.
  Automatic scores: plain 40/40, math 53/56, oracle 8/8, selection 15/16, smoke 3/4, clarification unmeasured
  (needs the audited review). All 132 replies are byte-identical to `thinking-retention-on-002` (temperature 0,
  same stack), so that run's hand audit applies unchanged. Empty answers after a calculator call: 0 of the 68 items that called it.

- P2 output-path check (`posttool-rloo-check-001`, DayCare (private commit), tinygrad-arkey 4322ae9e2): one update at 512
  tokens, raw checkpoint written, export + `verify` in a fresh process bit-exact (4,194,304/4,194,304); the
  checkpoint equals the final raw save. Not counted.
- P3 in progress (`posttool-rloo-r1-001`, counted, `gpu-run time`): updates 1-30 mean reward 0.838 / 0.816 /
  0.831 per ten, parity exact on every token, KL to the initial adapter 1.25e-2 at update 30, entropy 0.52-0.56,
  other-tool rate 0.9-2.8%, capped-turn rate 3-6%, ~90 s per update; raw checkpoint at 30.

- **P3 complete** (`posttool-rloo-r1-001`, counted; DayCare (private commit), tinygrad-arkey 4322ae9e2): 102 updates, no
  stop. Parity exact on all 2,644,937 sampled tokens (max error 0, rho 1.0 throughout). Mean training reward per
  20 updates 0.827 / 0.858 / 0.827 / 0.831 / 0.872 (updates 101-102: 0.812); KL to the initial adapter 5.1e-2 at
  the end; entropy 0.54 -> 0.50; other-tool rate 0.9-1.9%, capped-turn rate 3-9%, empty-after-tool rate 7-12%
  per 20 updates; 2.24 h (~79 s per update). Adapter exported (`adapter.xml` 1009ca700ebd..., raw
  8c378af7db4d...); `verify` in a fresh process bit-exact (4,194,304/4,194,304). G1/G1b adapter arms next.

## Finding

**Verdict: not adopted.** G1 passes, G1b passes, and **G2 fails**, which by the stop rule ends the experiment.
G3 was not run.

| Gate | Result | Verdict |
|---|---|---|
| G1: held-out miss + repair + empty (93 states x 4 episodes) | stock 0.669, adapter 0.739; paired difference **+7.0 points**, task-clustered 95% interval **[+1.4, +12.8]** (75 clusters, 50,000 resamples); 37 states gained, 21 lost | pass (lower bound > 0) |
| G1b: held-out relay (85 x 4) | stock 0.953, adapter 0.924; **-2.9 points**, interval [-6.2, +0.3]; 8 gained, 17 lost | pass (the upper bound is not below zero), but the point estimate is a loss |
| G2: empty answers after a tool result, G1 + G1b episodes | stock **60** (56 + 4), adapter **95** (81 + 14) of 712 | **fail** (the adapter must have strictly fewer) |
| G3: retention | not run: the stop rule ends the experiment at G2 (stock arm on file, `posttool-g3-stock-001`) | not run |

Runs: `posttool-g1-{stock,adapter}-001`, `posttool-g1b-{stock,adapter}-001`, adapter `posttool-rloo-r1-001`
(seeds 20260925 stock, 20260926 adapter). The analysis script is `posttool-rloo-r1-001/gates.py` (outside Git, with the run): bootstrap over source
tasks, with states grouped by their task.

### What changed, as GameTerm scenarios

- **Relay**: the calculator result is (or leads directly to) the answer; map it back to the question.
- **Repair**: GameTerm rejected the call's arguments, or the calculator refused the expression; fix and retry,
  or answer another way.
- **Miss**: Countdown; the result is not the target yet; keep searching.
- **Empty**: the post-result probe's states where thinking-on stock had answered nothing after a tool result.

| Scenario | Questions | Share of exam | Episodes | Correct, stock | Correct, adapter | Score, stock -> adapter |
|---|---:|---:|---:|---:|---:|---|
| relay | 85 | 48% | 340 | 324 | 314 | 0.953 -> 0.924 |
| repair | 61 | 34% | 244 | 170 | 210 | 0.697 -> **0.861** |
| miss | 28 | 16% | 112 | 75 | 59 | 0.670 -> **0.527** |
| empty | 4 | 2% | 16 | 4 | 6 | 0.250 -> 0.375 |

What the GameTerm user sees on a failure (episode counts, stock -> adapter):

| Scenario | Blank reply after a 4,096-token think (cut off) | Blank reply after the tool result | Another tool called instead of answering | Wrong or unreadable answer |
|---|---|---|---|---|
| relay | 0 -> **11** | 4 -> 3 | 2 -> 1 | 10 -> 11 |
| repair | 8 -> 19 | 13 -> 8 | **19 -> 0** | **34 -> 7** |
| miss | 27 -> **45** | 0 -> 2 | 4 -> 2 | 6 -> 4 |
| empty | 7 -> 7 | 1 -> 0 | 1 -> 0 | 3 -> 3 |

"Wrong or unreadable" includes a missing answer tag and the Countdown verifier's rejections (wrong numbers
used, syntax). No episode was still calling the calculator at the 8-turn cap. The adapter takes fewer tool
turns: 1.40 vs 1.76 per repair episode, and 1.06 vs 1.18 per relay episode.

The adapter learned to recover from a calculator refusal. It stopped reaching for other tools (19 -> 0) and
stopped guessing (34 -> 7 wrong answers). It also learned to think longer. On Countdown misses, and even on
some easy relay questions, the longer thinking now runs into the 4,096-token cap and the user sees nothing:
capped turns rose from 7.3% to 15.1% of turns in G1 and from 0% to 3.1% in G1b. That is exactly the failure
G2 guards against, and it got worse.

Worked examples, taken from the evaluation rollouts (four episodes per arm each):

- **Relay, a regression** (`relay-download-10`). "A 111,872 MB file downloads at 89.9 MB per second. How many
  minutes, rounded up?" (21). The model's own call was `111872 / 89.9 + 60`, and GameTerm returned 1304.4.
  - Stock recomputed or re-reasoned and answered 21 in 4 of 4 episodes.
  - Adapter: 2 of 4. One episode relayed the wrong result as `<answer>1305</answer>`. One thought for 4,096
    tokens and the user got a blank reply.
- **Repair, a win** (`relay-loaves-17`, calculator refusal). "388.3 kg of flour, 426 g per loaf; how many
  whole loaves?" (911). The call `int(388300/426)` was refused: "unknown identifier 'int' ... Write plain
  arithmetic".
  - Stock: 0 of 4. It answered 9, answered nothing, or called another tool.
  - Adapter: 4 of 4. It retried `388300/426` (2 episodes) or reasoned directly (2), then answered `<answer>911</answer>`.
- **Miss, a regression** (`miss-3-420057`). "Make 56 from [7, 13, 21]." The call `7+13+21 = 41` missed.
  - Stock found `7*(21-13)` in 4 of 4 episodes, one of them after a checking call.
  - Adapter: 1 of 4. Two episodes thought to the 4,096 cap (blank reply) and one called another tool.
- **Empty, a win** (`empty-train-59-miss`). "Make 94 from [78, 26, 74, 10]"; the call `78+26+74+10 = 188` missed.
  - Stock: 0 of 4. Two episodes were cut off, one reused 74 twice, one wrote `... + 74 * 0`.
  - Adapter: 1 of 4, with `((78-74)*26)-10`; the other three were cut off.

### Training-mix diagnosis (from the 3,264 training episodes of `posttool-rloo-r1-001`)

- **Mix of training episodes:** 49.5% relay, 30.6% repair, 16.7% miss, 3.2% empty. The held-out G1 set is 66%
  repair, 30% miss and 4% empty. (Counting by task id instead of state category would report 0% repair,
  because every repair state is built on a relay or miss task. The category, which is the first field of the
  state id, is the right key.)
- **Relay** was at ceiling throughout: training reward 0.947 in the first half, 0.953 in the second. Half the
  rollouts carried almost no learning signal, and the relay regression above appeared anyway.
- **Repair** training reward rose from 0.775 to 0.842, which matches the held-out gain.
- **Miss** training reward fell from 0.691 to 0.570, with cut-off endings rising from 52 to 93 of 272. The
  policy got longer on Countdown search while training, not only at evaluation.
- **Empty** training reward fell from 0.641 to 0.525.
- **Reading:** the reward pays only for a correct answer, and a cut-off turn scores 0, the same as a wrong
  answer. The update therefore gave no specific pressure to stop and answer. The longer thinking that helped
  repair spilled over into the Countdown search, where it runs out the budget. This is an interpretation of
  the logs, not a tested cause.

Per the stop rule, this experiment ends here, with no rescue runs. A next attempt needs its own
predeclaration. Its obvious design questions are:

1. Whether the turn budget should stay at the 4,096 cap GameTerm serves, or the reward should distinguish a
   cut-off turn from a wrong answer.
2. Whether relay should be dropped from training.
3. Whether miss and empty states belong in a post-tool experiment at all, since they are Countdown search.
