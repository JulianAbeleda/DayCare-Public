# A longer post-tool turn: do cut-off blanks finish, and are they right?

Status: **measured** 2026-09-27 (hypothesis and process predeclared in a private commit, before any measurement). Inference only, no
training. Approved by the maintainer 2026-09-27. Input to a product decision: should GameTerm raise its per-turn cap
(the GameTerm beta repo's `crates/provider/src/lib.rs:56`, `Limits::max_events = 4096`) instead of training blanks away?

## Why

On the held-out post-tool exam (G1 + G1b of [rloo-posttool-calculator.md](rloo-posttool-calculator.md)) most blank
replies are turns that thought until the 4,096-token cap and were cut: stock 42 of 60 blanks, run-1 adapter 82 of 95,
run-3 adapter 43 of 53 ([rloo-posttool-calculator-r3.md](rloo-posttool-calculator-r3.md)). Forcing `</think>` on
those turns rescued almost none ([posttool-budget-forcing.md](posttool-budget-forcing.md): stock 0 of 42), because the
stock cut turns are unfinished searches, not finished work. The open question is the opposite lever: let the search
run on. Test-time scaling work (s1, arXiv:2501.19393, "Wait" extension) finds that more thinking helps up to a point
and then plateaus; the cut turns here are the hardest tail (4-number Countdown), so the plateau may already be near.

## Hypothesis

**H:** resuming each turn cut at the 4,096 cap (no `</think>` added) with a total budget of 8,192 tokens for that
turn, **at least half** of the cut-off blanks finish (end on their own) by 8,192, and **at least 80%** of the
finished ones are correct by the exam's rule, in both the stock and the run-3 adapter arm.

Secondary predictions (stated, not gated): by 6,144 total tokens, fewer than two thirds of the 8,192 finishes have
happened; the run-1 adapter's loop turns (16 of 82) mostly do not finish (a loop does not end by itself).

Not a gate: a measurement for a product decision. Decision line: raising the cap is worth building if at 8,192 at
least 40% of the cut-offs become correct answers in both stock and run-3 (that is what H implies: half x 80%).

## Process

- Arms and blanks (all `posttool-tasks-002/states.xml`, envelope `rft-posttool-001/envelope.xml`, 4 episodes per
  state, 4,096 tokens per turn, cap 8 turns, temperature 1.0, 32 lanes):
  - stock: `posttool-force-{g1,g1b}-stock-001/blanks.json` (seed 20260925; reproduced the published exam exactly).
  - run-1 adapter: `posttool-force-{g1,g1b}-adapter-001/blanks.json` (seed 20260926; `posttool-rloo-r1-001`).
  - run-3 adapter (`posttool-rloo-r3-001`, seed 20260926): its exam `posttool-r3-{g1,g1b}-adapter-001` did not keep
    the cut turns' tokens, so each exam is **regenerated** with the same command, seed and settings plus
    `--keep-blanks` into `posttool-longer-{g1,g1b}-r3-001/`. Reproduction check first: every episode's reason and
    turn lengths vs the published `sample.xml`; on any mismatch it is reported and the regenerated run is the
    "before".
- Resume (`rloo_posttool resume`, DayCare commit with this file): every blank that stopped at the cap (`stop ==
  limit`) is resumed from its exact request (same rendering, BOS, history) + its 4,096 sampled tokens, with no token
  added, for 4,096 more tokens (turn total 8,192), same sampler, stop tokens, logit bias, temperature 1.0, seed = the
  arm's exam seed. One continuation per blank. If the resumed turn ends in a `calculate` call, the call runs through
  GameTerm's calculator and the episode continues as at the exam (turn cap 8; each follow-up turn gets 4,096 tokens,
  the loop's single limit; a follow-up cut there counts as still cut - conservative).
- 6,144 is read off the same run (sampling is causal): a resumed turn that ended by total token 6,144 is the same
  under a 6,144 cap; one that ended later counts as still cut at 6,144.
- Scoring: the resumed turn (original tokens + continuation) is parsed with `posttool.parse_turn` and the episode's
  last turn scored by `posttool.episode_reward` (the exam's rule). Outcomes: **correct / wrong** (incl. unreadable,
  verifier errors, other tool, reasoning not closed) / **still cut** (at 6,144 or 8,192) / **empty** (finished with
  no answer and no call).
- Main population: cut inside thinking (`incomplete`, reasoning not closed at the cap): stock 42, run-1 82, run-3 43.
  Turns cut after `</think>` (none expected) reported separately.
- Reported per arm and category (miss = Countdown search / repair of Countdown / empty / relay): cut-offs; finished
  by 6,144 / 8,192; correct / wrong / still cut; exam score (G1, G1b: mean reward per state) and blank count (G2:
  replies with no answer and no call, G1 + G1b) as if the cap were 6,144 / 8,192; the distribution of extra
  thinking tokens (to `</think>`) of finished turns; wall-time cost per extra-long turn at single-stream decode (user
  latency); GameTerm scenario framing; what exactly changes in `the GameTerm beta repo` and whether events ~ tokens.

## Finding

**H falsified.** Letting a cut turn think on to 8,192 tokens finishes about half of the cut-offs (stock 22 of 42,
run-3 22 of 43: the first half of H holds), but the finished turns are mostly **wrong**: correct 5 of 22 (23%) stock,
3 of 22 (14%) run-3, far under the predeclared 80%. As a share of all cut-offs, 5 of 42 (12%) and 3 of 43 (7%) become
correct answers, under the 40% decision line. **Raising the cap is not an accuracy lever.** It mostly trades a blank
for a rule-breaking Countdown answer, after roughly twice the wait.

**Reproduction.** The regenerated run-3 exam (`posttool-longer-{g1,g1b}-r3-001`, DayCare (private commit) + a logging fix,
tinygrad-arkey 4322ae9e2) is **identical** to the published one: 712 of 712 episodes have the same reason and turn
lengths. The stock and run-1 blanks come from `posttool-force-*` (already shown identical to their exams). Every
Countdown puzzle among the cut-offs is solvable (brute force over + - * /; all 47 held-out Countdown states are).
Time (`gpu-run time`, 32 lanes): resume 418 s stock, 412 s run-3, 503 + 305 s run-1; run-3 exam regeneration
607 + 275 s.

### Cut inside thinking at 4,096, resumed to 8,192 (predeclared)

Finished = the turn ended on its own by that total. Outcomes at 8,192. "No answer" = it ended with nothing (end token
inside thinking). "Still cut" includes a finished turn whose `calculate` call led to a follow-up turn that was cut
(stock 2, run-3 4).

| Arm | Scenario | Cut-offs | Finished by 6,144 | Finished by 8,192 | Correct at 6,144 | Correct at 8,192 | Wrong | No answer | Still cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | repair | 8 | 4 | 4 | 2 | 2 | 2 | 0 | 4 |
| stock | miss | 27 | 9 | 15 | 2 | 3 | 10 | 0 | 14 |
| stock | empty | 7 | 0 | 3 | 0 | 0 | 3 | 0 | 4 |
| **stock** | **all** | **42** | **13 (31%)** | **22 (52%)** | **4 (10%)** | **5 (12%)** | **15** | **0** | **22** |
| run-3 | repair | 10 | 2 | 4 | 0 | 1 | 2 | 0 | 7 |
| run-3 | miss | 23 | 9 | 13 | 1 | 2 | 9 | 0 | 12 |
| run-3 | empty | 10 | 2 | 5 | 0 | 0 | 4 | 0 | 6 |
| **run-3** | **all** | **43** | **13 (30%)** | **22 (51%)** | **1 (2%)** | **3 (7%)** | **15** | **0** | **25** |
| run-1 | relay | 11 | 6 | 7 | 3 | 4 | 0 | 3 | 4 |
| run-1 | repair | 19 | 4 | 7 | 3 | 6 | 0 | 1 | 12 |
| run-1 | miss | 45 | 3 | 8 | 2 | 2 | 3 | 3 | 37 |
| run-1 | empty | 7 | 2 | 2 | 0 | 0 | 2 | 0 | 5 |
| **run-1** | **all** | **82** | **15 (18%)** | **24 (29%)** | **8 (10%)** | **12 (15%)** | **5** | **7** | **58** |

(Relay has no cut-offs in stock or run-3. No turn was cut after `</think>`; every cut-off was inside thinking.)

- **The wrong answers break the rules.** Stock's 15 wrong: 8 reuse or drop a number (`number_use`), 3 miss the target,
  2 unparseable, 1 extra text, 1 web-search call. Run-3's 15: 10 `number_use`, 3 wrong target, 1 unparseable, 1 other
  tool. Typical: `<answer>78+26-10+(74-74)</answer>` (uses 74 twice to cancel it, and says "the zero contributed by
  (74-74)"); `<answer>8 × (31 - 25)</answer>` for "48 from [8, 25, 31, 41]" ("41 is not needed"). After a long
  failed search the model relaxes the constraint and claims success. Compare: stock turns that finish under 4,096
  on miss are 88% correct (75 of 85); the extended ones 20% (3 of 15).
- **Stock and run-3 cut-offs contain no repetition loops** (0 of 85; 90%+ of the last 512 tokens in a repeated
  20-gram). They are still-searching turns. Run-1 has 19 loops; 8 of them end within 8,192, 1 correctly. Run-1's
  better correct share comes from relay and repair (10 of 30), where the answer is a relay of a known value.
- **Finishing is spread across the whole extra budget,** not front-loaded: of the finished turns, the extra thinking
  (tokens to `</think>`) is p10 / median / p90 = 564 / 1,693 / 3,516 (stock), 543 / 1,857 / 3,726 (run-3),
  428 / 1,237 / 3,353 (run-1). About 30% of cut-offs finish in the first 2,048 extra tokens and another ~21% in the
  next 2,048, so a larger cap (12k, 16k) would finish more, at the same poor accuracy.

### Exam scores as if the cap were larger (mean reward per state; blanks = G2 count, replies with no answer and no call, G1 + G1b)

| Arm | G1 at 4,096 / 6,144 / 8,192 | G1b at 4,096 / 6,144 / 8,192 | Blanks at 4,096 / 6,144 / 8,192 |
|---|---|---|---|
| stock | 0.669 / 0.680 / 0.683 | 0.953 / 0.953 / 0.953 | 60 / 48 / 40 |
| run-3 | 0.798 / 0.801 / 0.806 | 0.962 / 0.962 / 0.962 | 53 / 44 / 35 |
| run-1 | 0.739 / 0.753 / 0.761 | 0.924 / 0.932 / 0.935 | 95 / 84 / 78 |

- G1 gains at 8,192: stock +1.4 points, run-3 +0.8, run-1 +2.2. The blank count falls by a third (stock 60 -> 40,
  run-3 53 -> 35), but for stock 15 of those 20 blanks became wrong answers and 5 became right ones.
- G2 is cap-dependent: at 8,192, run-3 would show 35 blanks vs stock 40 (-5), the same difference as at 4,096
  (53 vs 60, -7). A larger cap does not change the run-3 verdict; it moves both arms.

### Wall-time cost (user latency)

Single-stream decode on the RTX 5090, llama.cpp b9592 (ac4cddeb0) CUDA, the same BF16 GGUF, `llama-bench -n 256`
at context depth 2,048 / 6,144 / 10,240: **194 / 193 / 191 tok/s** (`posttool-longer-bench.log`). So:

- A 4,096-token turn (today's cap) is about 21 s. Raising the cap to 6,144 adds up to 10.7 s per cut turn; to 8,192
  up to 21.4 s (a turn cut again at 8,192 takes ~43 s and still shows nothing).
- Mean extra decode per cut-off turn at 8,192: 2,940 tokens stock (15 s), 3,089 run-3 (16 s). At 6,144: 1,751 (9 s),
  1,815 (10 s).
- Per rescued correct answer: 129 s of extra decode (stock, 8,192), 232 s (run-3). All within GameTerm's 300 s
  `total_timeout`.
- Cap hits are rare exam-wide: 42 of 712 stock episodes (5.9%), 43 of 712 run-3.

### As GameTerm scenarios (a post-tool turn that thought to the cap; cap 4,096 -> 8,192)

| Scenario | Stock today | Stock at 8,192 | Run-3 today | Run-3 at 8,192 |
|---|---|---|---|---|
| miss (Countdown search) | 27 blank after ~21 s | 3 right, 10 wrong (mostly a number reused or dropped, stated as valid), 14 blank after ~43 s | 23 blank | 2 right, 9 wrong, 12 blank |
| repair (Countdown after a rejected call) | 8 blank | 2 right, 2 wrong, 4 blank | 10 blank | 1 right, 2 wrong, 7 blank |
| empty (Countdown, empty result) | 7 blank | 0 right, 3 wrong, 4 blank | 10 blank | 0 right, 4 wrong, 6 blank |
| relay | never happens | never happens | never happens | never happens |

For a Countdown user, 12% of the cap-hit turns become right; 36% become a confident answer that breaks the puzzle's
rule; 52% stay blank after twice the wait. This matches budget forcing and TC-070: once the stock search has run
4,096 tokens without a solution, more budget mostly buys a rationalized wrong answer.

### GameTerm parity: what would change, and events vs tokens

`max_events` counts SSE `data:` events, not tokens (`crates/provider/src/lib.rs`, `event()`, line 892:
`event_count += 1` per event, `LimitExceeded` past `max_events`, line 894). llama-server streams one event per
generated token (reasoning and content deltas alike) plus a few framing events (role, finish, usage, `[DONE]`), so
**events = tokens + 4**, measured: llama-server b9592, this GGUF, thinking on, `max_tokens` 300 streamed 304 `data:`
events (300 one-token reasoning deltas + role/finish/usage/`[DONE]`), 83,889 bytes (276 bytes per event). To give a turn 8,192 tokens GameTerm would change:

1. `Limits::default` `max_events: 4096` (line 56) -> at least 8,192 + a small margin (e.g. 8,256), and the request's
   `max_tokens` (the calculator study sends 4,096, `crates/host/tests/calculator_study.rs:194`) -> 8,192. Today the
   event cap and the token cap coincide, so a capped turn fails the stream instead of ending with `finish_reason:
   length`.
2. Check the other limits it would meet: `max_total_bytes` 8 MiB (line 55; 8,256 events x 276 bytes is 2.3 MB),
   `total_timeout` 300 s (line 63; 8,192 tokens at 191 tok/s is 43 s), `max_events_per_second` 256 (line 57; above
   191 tok/s on this GPU, already a trip wire on faster hardware whatever the cap). `MAX_TEXT_BYTES` 16 KiB
   (`crates/backend/src/lib.rs:29`) caps content and silently truncates kept reasoning (lines 944-948); 8,192
   thinking tokens (~30 KB) overflow it, which matters only for the TH-010 stand-in.
3. The server's context (`-c`) must hold prompt + history + 8,192.

Given the numbers above, none of this is worth doing for accuracy on this exam.

### Decision (the maintainer, 2026-09-27 ~21:15 EDT)

- **Keep the 4,096 turn budget.** 8,192 cuts blanks (stock 60 -> 40, run-3 53 -> 35) mostly into wrong answers,
  often rule-breaking ones (reused or dropped numbers stated as valid). A confident wrong answer is worse for the user
  than a blank.
- **Gate-design lesson:** the blank count alone is a misleading metric here. G2 (fewer blanks) must not be passable by
  converting blanks into wrong answers; a blank gate needs a companion no-new-wrong-answers condition.
- **Next step:** give the model an explicit out. At the budget cut-off, instead of a blank or a forced guess, the
  model reports through a dedicated tool that it could not solve the task (follow-up work on a
  `report_unsolved`-style tool).

Runs: `<runs>/posttool-force-{g1,g1b}-{stock,adapter}-001/longer/resume.xml`,
`posttool-longer-{g1,g1b}-r3-001/{sample.xml,blanks.json,longer/resume.xml}`; logs `posttool-longer-stock.log`,
`posttool-longer-rest.log`, `posttool-longer-bench.log`. Analysis (outside Git, with the runs):
`posttool-longer-analysis/{analyze.py,repro.py}`.
