# Post-tool RLOO: does the one-stack loop run as intended, and as fast as intended?

Date: 2026-09-26. Predeclared: the predictions below are committed before any measurement.

## Hypotheses

Two things are measured: quality (the experiment ran as intended) and speed (it ran as fast as intended).

**H1, quality: the loop runs as intended.** Every one of these holds on the timing run; any failure falsifies H1:
1. Episode rendering: the post-tool prompt (model's own call + real calculator result in the wire envelope) is
   token-identical to what GameTerm HT-020 sends on the later turn (parity test passes, or the unproven part is named).
2. The calculator actually ran, and the result in the context is its real output.
3. Tool-result tokens are masked out of the loss; only sampled tokens get gradient.
4. Rewards are right: a hand check of at least 16 scored rollouts agrees with the verifier on every one
   (correct non-empty answer -> 1, else 0; tool calls neither rewarded nor penalized).
5. Sampler/trainer parity is bit-exact on every scored token (max error 0).
6. No NaN / non-finite value / FloatingPointError; no empty rollouts from a sampler fault.
7. The adapter changes in place, and save -> reload reproduces log probabilities bit-exactly.
8. A rerun with the same seed reproduces rewards and parity exactly (determinism).

**H2, speed: it runs as fast as intended.** One update (4 prompts x 8, thinking on, real 4B) takes 20-35 s wall
after the first, so a 20-update run takes ~10 min, because each stage is bounded by numbers already measured on
tinygrad-arkey `exp` (table below). Falsified if wall per update exceeds 52 s (1.5x the upper prediction).

## Why it matters

The next RL experiment trains the post-tool turn: Nemotron 3 Nano 4B reading and relaying a real calculator
result on the GameTerm harness (calling the tool is not the failure; interpreting the result is). Iteration speed
decides how many experiments we can run, so the loop's speed is itself a claim to test. If the one-stack loop
(tinygrad-arkey `exp` sampler + trainer, bit-exact capture) is as fast as today's measurements imply, one update
takes well under a minute and a full run takes minutes, not an hour.

## Process (fixed before measuring)

- Policy: Nemotron 3 Nano 4B BF16, thinking on, LoRA on the final block (as the Countdown runs).
- Episodes: post-tool states (prompt + the model's own calculator call + the real result in GameTerm's wire
  envelope), 4 prompts x group 8 = 32 lanes, continuation capped at 4096 tokens per turn.
- Loop: `daycare/nursery/rloo_tinygrad.py` on tinygrad-arkey `exp` (prefix sharing 73bf13a28, prefill shared pool
  2694bdd20, coalescing heuristic 257491111, decode routes 2f029d81d), timed with `gpu-run time` (exclusive GPU).
- Run: 2 updates, not counted as an experiment; exercises save -> reload -> exact check.

## Predictions (per update, after the first)

| Stage | Basis (measured 2026-09-25/26) | Predicted |
|---|---|---|
| Prime | shared ~10k envelope once (~1.2 s), per-prompt suffix ~1-2k tokens at ~46 ms / 256 tokens | 2-3 s |
| Sample | continuations ~300-1500 tokens; ~2k steps x 10.6 ms at B=32; continuous batching | 10-25 s |
| Parity + grad + Adam | ~26k tokens -> ~800 slices x ~9.4 ms at 32 lanes | 7-8 s |
| Reward, save, bookkeeping | exact verifier, adapter export | ~1 s |
| **Wall per update** | | **20-35 s** |

- First update: +1-2 min one-time compile and warm-up.
- 20-update run: ~10 min (Countdown R4: 56 min at ~165 s per update).
- Reference: Countdown one-stack update 165 s; vendored llama.cpp loop ~5 h per update at this scale.

## Pass rule

H1 passes only if all eight checks hold. H2: the prediction holds if measured wall per update is within 1.5x of 20-35 s (i.e. <= 52 s). Any stage above 1.5x its
predicted range gets an explanation before the loop is called proven.

## Risks named in advance

1. Continuation length: if thinking-on post-tool turns run toward the 4096 cap, sampling dominates (50-60 s/update).
2. The shared envelope not reused across updates (+~1 s per update).
3. Drain: one long rollout holds the batch.
4. Multiple tool calls per rollout: each extra call is a re-prefill.

## Finding

Runs (not counted): `posttool-timing-001` and its same-seed rerun `posttool-timing-002`, DayCare (private commit),
tinygrad-arkey exp 327bc54db (contains all four named commits), `gpu-run time`, 2 updates of 4 states x 8
episodes from `posttool-tasks-001/states.xml` (train split), thinking on, 4,096 tokens per turn, cap 8 turns,
lr 2e-4. Parity evidence: `posttool-parity-001/`.

### H1 (quality): holds (8/8, one named gap in check 1)

| # | Check | Evidence | Result |
|---|---|---|---|
| 1 | Rendering is what GameTerm HT-020 sends | 2,329 captured 4B later-turn requests: messages 2,329/2,329, token counts (with llama.cpp's BOS) 2,329/2,329; llama-server b9592 /apply-template + /tokenize on 65 post-tool states (thinking on, rejections, 2-exchange histories): 65/65 text, 65/65 ids. Unproven: DayCare's parser (not llama-server's) reads the model's raw call; checked only on template-rendered calls | pass (named gap) |
| 2 | The calculator ran; the context holds its real output | GameTerm `calculate` (runner built from the GameTerm beta build) reproduced 2,251/2,251 captured results; the timing run executed 28 follow-up calls (e.g. `ceil(164980/49) = 3367`, a schema rejection for `{"expr": ...}`), rendered into the next turn | pass |
| 3 | Tool-result tokens masked | loss tokens == sampled tokens in both updates (42,430 and 10,162); prompts/history/results are prefill only (50,240 and 34,827 prefill tokens); tiny-config test asserts it | pass |
| 4 | Rewards right on >= 16 rollouts | all 32 episodes of update 2 checked by hand against the rule: 21 correct answers scored 1; the 11 zeros were 4 wrong numbers (3367 and 2799 minutes for a 57-minute download: seconds not converted; 287 jugs for 286), 4 literal `<answer>the number</answer>` (the v1 prompt's placeholder, since reworded) and 3 turns that called a non-calculate tool with no answer. No disagreement | pass |
| 5 | Sampler/trainer parity bit-exact | 42,430/42,430 and 10,162/10,162 log probabilities exact, max error 0, rho 1.0 | pass |
| 6 | No NaN / non-finite / sampler-fault empties | both runs complete; no FloatingPointError; no zero-length turn | pass |
| 7 | Adapter changes in place; save -> reload exact | KL to the initial adapter 1.1e-4 after update 1, step RMS 1.4e-4 (= lr); `verify` in a fresh process reloaded the exported adapter equal to the raw save and reproduced 4,194,304/4,194,304 probe log probabilities bit for bit | pass |
| 8 | Same-seed rerun reproduces | rewards, loss (-38.97997, 11.86571), all 32+32 episode summaries and the exported adapter (raw sha256 fca02b10...) identical between the two runs | pass |

### H2 (speed): falsified, narrowly (53.2 s per update > 52 s)

One post-warmup update per run (update 2), identical within 0.3 s across the two runs:

| Stage | Predicted | Measured (update 2) | Ratio | Why, where > 1.5x |
|---|---|---|---|---|
| Prime (prefill) | 2-3 s | 12.6 s: 3 rounds; 34,827 prefill tokens (the 10,240-token shared envelope primed once per round = 30,720, plus 4,107 own tokens) | 4-6x | The envelope is shared within a sampling call, not across calls: each follow-up round (a model's second or third calculator call) re-primes it (risk 4). |
| Sample (decode) | 10-25 s | 37.0 s: 1,728 decode steps x 21.4 ms, 10,162 tokens generated (275 tok/s), lane utilization 18% | 1.48x | Under 1.5x. The step is 21 ms, not 10.6 ms (attention reads the ~10k shared prefix every step), and rounds serialize: a round lasts as long as its longest turn (1,000 tokens here) while the other lanes idle (risk 3). |
| Parity + grad + Adam | 7-8 s | 3.2 s (1.4 + 1.6 + 0.2; 318 slices x 4.9 ms) | 0.4x | fewer tokens than predicted |
| Reward, render, tool, bookkeeping | ~1 s | ~0.8 s (render 0.4, calculator 1 ms, verifier < 10 ms) | ok | |
| **Wall per update** | **20-35 s** | **53.2 s** (rerun 53.5 s) | | |

- First update: 346 s (sampling 329 s, mostly compiling the decode graphs per attention bucket), vs +1-2 min predicted.
- Episodes per minute: 36 (32 episodes, 40 sampled turns, 8 follow-up calls per update).
- Prefill vs decode tokens per update: 34,827 prefilled (tool results and history included) vs 10,162 sampled.
- Sample size: one warm update per run. These two updates drew short relay/repair turns (longest 1,000 tokens).
  The stock headroom run over all four categories (`posttool-headroom-001`, 940 turns, same sampler at 32
  lanes) had median 340 tokens per turn but 11% of turns at the 4,096 cap (Countdown miss and empty states)
  and 26 ms per decode step. With ~40 turns per update most updates will contain one capped turn, so a
  typical update is ~4,096 x 22-26 ms + ~15 s = 105-120 s (risk 1).
- Projected full run (102 updates, one pass over the 408 training states): 1.5 h at the measured 53 s,
  about 3-3.5 h at the headroom-derived 105-120 s; plus ~6 min first-update compile.

Levers, not implemented (each a code change for a later decision): keep the shared envelope primed across
rounds (about -8 s per update here); feed follow-up turns into the running batch instead of a new round
(removes the round barrier; needs a sampler API for requests added mid-generation); a lower per-turn cap on
Countdown-style states (changes the task, so it is a protocol decision).

## Finding 2: after the two speed levers (2026-09-27)

Change (engineering only; every predeclared setting unchanged): tinygrad-arkey exp 4322ae9e2 keeps a primed
shared prefix across `generate` calls (skipped only when the same tokens are primed, saved and loaded and the
prefill's rows still hold them) and adds `generate(follow=...)`, so a follow-up turn joins the running batch as
soon as a lane frees instead of after the slowest turn of a round; DayCare (private commit) runs each update's episodes
in one sampling call. Tests: reuse is bit-identical to re-priming, follow-up rollouts match the model's
reference log probabilities, same seed gives the same schedule and samples (tinygrad-arkey unit tests; DayCare
tiny multi-turn update with a shared prefix).

Runs: `posttool-timing-003` and its same-seed rerun `-004` (the name `-002` was already the first finding's
rerun), same states, seed and settings as `-001`, `gpu-run time`. The schedule changed, so the sampled episodes
differ from `-001`'s; update 2 drew 42 turns (longest 690 tokens) against `-001`'s 40 (longest 1,000).

**H1: holds again (8/8).** Parity exact (37,812/37,812 and 8,190/8,190, max error 0); loss tokens == sampled
tokens; 30 follow-up calls ran through GameTerm's calculator; no non-finite value or empty sampler turn;
`verify` reloaded the adapter bit for bit (4,194,304/4,194,304); `-004` reproduced `-003` exactly (episodes,
losses, adapter raw sha256 681022a84dba...). Rewards of update 2 spot-checked against the rule: 18 correct,
11 `no_number` (the v1 placeholder prompt, reworded in `posttool-tasks-002`), 2 reasoning not closed, 1 wrong.

**H2: now holds (36.7 s per warm update <= 52 s; the predicted 20-35 s range is missed by 1.7 s).**

| Stage (update 2) | Predicted | Finding 1 (`-001`) | Finding 2 (`-003`) | Note |
|---|---|---|---|---|
| Prime | 2-3 s | 12.6 s (shared primed 3x: 30,720 tokens) | 9.4 s (shared primed 0x; 14 own prompts, 5,097 tokens) | Envelope reuse works (0 shared tokens re-primed), but priming 14 short own prompts at position 10,240 costs ~0.67 s each: per-prompt overhead, not the envelope, is now the prime cost (still > 1.5x). |
| Decode | 10-25 s | 37.0 s: 1,728 steps x 21.4 ms, 18% lanes busy | 24.4 s: 1,120 steps x 21.8 ms, 23% lanes busy | The barrier is gone: the step count follows the longest episode chain, not the sum of round maxima. Utilization stays low because 32 episodes mostly finish in a few hundred tokens and one long chain drains alone. |
| Parity + grad + Adam | 7-8 s | 3.2 s | 2.9 s | |
| Other | ~1 s | ~0.8 s | ~0.8 s | |
| **Wall** | **20-35 s** | **53.2 s** | **36.7 s** (rerun 36.4 s) | |

- First update: 272 s (was 346 s), mostly decode-graph compilation.
- Episodes per minute: 52 (was 36).
- Remaining levers: per-prompt prime overhead (batch or bucket short own prompts); cross-update overlap is the
  only way past the single-chain drain and would make sampling off-policy, so it is a protocol question.
- Projection for the predeclared 102-update run: the headroom runs' 11% capped turns still imply one ~4,096-token
  chain in most updates, i.e. ~4,096 x 22 ms + ~15 s = ~105 s for a typical update (the drain is the chain
  length, which the levers do not shorten), so ~2-3 h rather than 1.5-3.5 h; the 36.7 s here is the
  short-turn case.
