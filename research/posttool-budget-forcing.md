# Budget forcing on the post-tool exam: do blank replies become answers?

Status: **measured** 2026-09-27 (hypothesis and process predeclared in a private commit, before any measurement). Inference only, no
training. Input to a product decision (the maintainer): should GameTerm force-close thinking when a turn hits its cap?

## Hypothesis

On the held-out post-tool exam (G1 + G1b of [rloo-posttool-calculator.md](rloo-posttool-calculator.md)), most
blank replies are turns that thought until the 4,096-token cap (stock 42 of 60 blanks, run-1 adapter 82 of 95).
Budget forcing (s1, arXiv:2501.19393: end thinking with the end-of-thinking delimiter and let the model answer;
AnytimeReasoner, arXiv:2505.13438: a truncated thought still yields a usable answer) should turn a useful share of
them into answers.

**H:** appending the model's own end-of-thinking token to a turn cut at the cap and letting it answer with a small
budget (256 tokens) turns at least a quarter of those blanks into correct answers, in both arms, and more of them
in miss (Countdown search, where a partial search may already hold the solution) than the model would get by
guessing.

Not a gate: this is a measurement for a product decision. The quarter is the predeclared line for "worth building":
below it, forcing mostly converts blanks into wrong answers.

## Process

- Runs to reproduce: `posttool-g1-{stock,adapter}-001`, `posttool-g1b-{stock,adapter}-001` (states
  `posttool-tasks-002/states.xml`, envelope `rft-posttool-001/envelope.xml`, 4 episodes per state, 4,096 tokens
  per turn, cap 8 turns, temperature 1.0, 32 lanes, seeds 20260925 stock / 20260926 adapter; adapter =
  `posttool-rloo-r1-001/adapter.xml`, sha256 1009ca700ebd...). Their `sample.xml` keeps only a 600-character
  reasoning tail per episode, not the tokens of the cut turn, so each arm is **regenerated** with the same
  command, seed and settings (`rloo_posttool sample --split heldout ... --keep-blanks`), which also writes every
  blank episode's final-turn tokens (`blanks.json`). Reproduction check: per-episode reason and turn lengths vs the
  published `sample.xml`; any mismatch is reported and the regenerated run is the "before".
- Forcing (`rloo_posttool force`, same sampler, same stop tokens, temperature 1.0, 256 tokens): the prompt is the
  exact request of the blank turn (the same rendering, BOS and history) + the model's sampled tokens of that
  turn (a trailing end token removed) + the end-of-thinking token (`</think>`; id checked against the tokenizer
  at run time). One forced continuation per blank episode, seed = the arm's exam seed.
- Which blanks: (a) **cut at the cap** inside thinking (`incomplete`, reasoning not closed) - the main question;
  (b) **reasoning never closed** (the model emitted its end token inside thinking) - forced the same way, reported
  separately; (c) cut at the cap after `</think>` (mid-answer) - not forceable by this rule, counted only;
  (d) voluntary empty replies (closed thinking, no content) - not forced.
- Scoring: the forced turn (original tokens + `</think>` + continuation) is parsed with `posttool.parse_turn` and
  scored by `posttool.episode_reward` (the exam's rule). Outcomes: correct / wrong (incl. unreadable and Countdown
  verifier errors) / still blank (empty, or cut again at 256) / tool call (the forced reply calls a tool instead of
  answering; scored 0 as at the exam, where GameTerm would run it).
- Reported per arm and per category (relay / repair / miss / empty): blanks before; forced correct / wrong / still
  blank / call; the exam score with forced answers substituted (mean reward per state, as G1/G1b) vs the published
  numbers; three forced answers verbatim; and what GameTerm (HT-020) would need to change to do this.

## Finding

**H falsified.** Forcing the end of thinking on a turn cut at the cap turns **0 of 42 (0%)** stock blanks and **5 of
82 (6%)** adapter blanks into correct answers, far under the predeclared quarter. On miss (Countdown) it rescues
**0 of 72** across both arms. The published exam scores barely move. Forcing mostly turns a blank into a wrong
answer or into more reasoning written as the reply. **It is not an accuracy lever.** It is at most a way to show
the user something instead of nothing (the conclusion `research/countdown-thinking-e2e.md` reached for llama-server's
`--reasoning-budget`).

**Reproduction.** The regenerated runs (`posttool-force-{g1,g1b}-{stock,adapter}-001`, DayCare (private commit),
tinygrad-arkey 4322ae9e2) are **identical** to the published exam: 712 of 712 episodes per arm have the same reason
and turn lengths. The "before" column is therefore the published exam. The `</think>` token is id 13 (checked
against the GGUF tokenizer at run time). Time: 4 exam arms 31 min, forcing 8 min (`gpu-run time`).

### Blanks cut at the 4,096 cap, forced with 256 answer tokens (predeclared)

| Arm | Scenario | Blanks before | Correct | Wrong | Still no answer | Tool call instead |
|---|---|---:|---:|---:|---:|---:|
| stock | relay | 0 | - | - | - | - |
| stock | repair | 8 | 0 | 1 | 4 | 3 |
| stock | miss | 27 | 0 | 10 | 16 | 1 |
| stock | empty | 7 | 0 | 0 | 6 | 1 |
| **stock** | **all** | **42** | **0 (0%)** | **11 (26%)** | **26 (62%)** | **5 (12%)** |
| adapter | relay | 11 | 4 | 1 | 5 | 1 |
| adapter | repair | 19 | 1 | 5 | 11 | 2 |
| adapter | miss | 45 | 0 | 16 | 25 | 4 |
| adapter | empty | 7 | 0 | 1 | 6 | 0 |
| **adapter** | **all** | **82** | **5 (6%)** | **23 (28%)** | **47 (57%)** | **7 (9%)** |

- **"Still no answer"** means the model kept reasoning in the reply after `</think>` and used up the 256 tokens again
  (stock 26 of 26, adapter 45 of 47), or ended with nothing (adapter 2).
- **"Wrong"** means a number that fails the rule (Countdown: wrong target or numbers reused; relay: wrong value)
  or an unreadable answer.
- **"Tool call"** scores 0, as at the exam. GameTerm would run the call and continue.
- Every cap blank was still inside thinking, so all of them were forceable. Voluntary empty replies (stock 5,
  adapter 1) were not forced.

**Why forcing rescues nothing for stock: the cut turns are unfinished searches, not finished work.** A turn counts as a
**repetition loop** when 90% or more of its last 512 tokens sit in a repeated 20-gram, and as **still searching**
otherwise:

- None of stock's 42 cut turns is a loop. All are still searching.
- 16 of the adapter's 82 cut turns are loops, such as `0.80? 0.80? 0.80? ...` (relay-unit_price-00) or `1403.24 is
  1403.24.`.
- **All 5 correct forced answers came from loops.** The model already had the answer and was stuck repeating it.
  0 of 66 still-searching cut turns became correct.

Forcing recovers what the adapter's run-1 training broke (the loops the adapter learned on relay). It does not
finish a search.

### Reasoning never closed (the end token came inside thinking), forced the same way

| Arm | Blanks | Correct | Wrong | Still no answer | Tool call |
|---|---:|---:|---:|---:|---:|
| stock (relay 4, repair 8, empty 1) | 13 | 3 (23%) | 7 | 0 | 3 |
| adapter (relay 3, repair 7, miss 2) | 12 | 3 (25%) | 2 | 4 | 3 |

Most of these turns wrote a tool call inside thinking and then stopped (for example
`...<parameter=expression>\n381 - 324.2\n</parameter>\n</function>\n</tool_call>` and then end).

### Exam scores with forced answers substituted (mean reward per state, as G1/G1b; stock -> adapter)

| Exam | As published | Force at the cap | Force at the cap + unclosed |
|---|---|---|---|
| G1 (miss + repair + empty, 93 states) | 0.669 -> 0.739, +7.0 [+1.4, +12.8] | 0.669 -> 0.742, +7.3 [+1.8, +12.9] | 0.672 -> 0.747, +7.5 [+2.1, +13.2] |
| G1b (relay, 85 states) | 0.953 -> 0.924, -2.9 [-6.2, +0.3] | 0.953 -> 0.935, -1.8 [-5.0, +1.5] | 0.959 -> 0.938, -2.1 [-5.0, +0.9] |

(Difference in points, task-clustered 95% bootstrap interval, as `gates.py`.)

- **Stock:** forcing the cap adds 0 correct episodes. Forcing the unclosed turns too adds 3 of 712 (+0.4 points of
  the combined exam).
- **Adapter:** forcing adds 5, or 8 with the unclosed turns, of 712 (+1.1 points). Most of that is relay, where
  forcing removes about a third of the relay regression.
- No gate verdict changes. G2 (fewer blanks) is not rescued either: the forced replies that still carry no answer
  outnumber the correct ones 26:0 (stock) and 47:5 (adapter).

### Post-hoc sensitivity: a 1,024-token answer budget (not predeclared; `posttool-force-*-001/l1024/`)

| Arm | Cap blanks | Correct | Wrong | Still no answer | Tool call |
|---|---:|---:|---:|---:|---:|
| stock | 42 | 2 (5%, both miss) | 14 | 24 | 2 |
| adapter | 82 | 8 (10%: relay 7, miss 1) | 16 | 42 | 16 |

With a larger budget, the "answer" is mostly more search. At 1,024 tokens, G1 stock is 0.675 and adapter 0.742;
G1b adapter is 0.944. The conclusion does not change.

### Three forced answers, verbatim

1. **Stock, miss, forced into a wrong answer** (`miss-4-440026`, "make 53 from [91, 82, 63, 90]"). The last thought
   was `Thus cannot get 92 from 91 and`, then `</think>`. The reply: "My calculations tried many possible ways, but
   using each of 91, 82, 63, and 90 exactly once ... I have not found an expression that equals 53.
   `<answer>(91*82)/(63+90)</answer>`". The model says it failed, then commits a wrong expression anyway.
2. **Stock, miss, forced into a false claim** (`miss-4-440103`, "make 48 from [8, 25, 31, 41]"). The reply:
   `<answer>(41 - 8/25) * 31 = 48</answer>`. This is 1,262.2, not 48.
3. **Adapter, relay, a loop rescued** (`relay-unit_price-00`). The thinking ended in `0.80? 0.80? 0.80? ...` up to
   the cap. After `</think>` the reply was `<answer>0.80</answer>`, which is correct.

### As GameTerm scenarios (what the user sees on a turn that thought to the cap)

| Scenario | Today (stock) | With forcing, 256 tokens (stock) | Today (adapter) | With forcing (adapter) |
|---|---|---|---|---|
| relay | never happens | never happens | 11 blank | 4 right, 1 wrong, 5 reasoning text with no answer, 1 extra tool call |
| repair | 8 blank | 0 right, 1 wrong, 4 reasoning text, 3 tool calls | 19 blank | 1 right, 5 wrong, 11 reasoning text, 2 tool calls |
| miss | 27 blank | 0 right, 10 wrong, 16 reasoning text, 1 tool call | 45 blank | 0 right, 16 wrong, 25 reasoning text, 4 tool calls |
| empty | 7 blank | 0 right, 6 reasoning text, 1 tool call | 7 blank | 0 right, 1 wrong, 6 reasoning text |

For a Countdown user, the trade is a blank reply against a confident wrong expression (example 2). This is the
TC-070 finding again, now on the post-tool turn.

### GameTerm parity: what the product would need

GameTerm does not cap at a token count. The cap lives in the GameTerm beta repo's `crates/provider/src/lib.rs`:

- `Limits::default` sets `max_events: 4096` (line 56) and `total_timeout: 300 s` (line 63). Passing the event cap
  returns `AdapterError::LimitExceeded` (line 894), so the turn fails. HT-020's calculator study also sends the
  envelope's `max_tokens` (4,096, `crates/host/tests/calculator_study.rs:194`).
- `finish_reason` is never read. A turn that ends normally with empty content but with `reasoning_content` is shown
  to the user with the thinking as its text (TH-010 stand-in, lines 1043-1062; its message assumes that thinking was
  switched off). A turn with neither fails with `Protocol` / `ExplicitUserRetry` (line 1065).
- GameTerm talks only to `/v1/chat/completions`.

Budget forcing would therefore need three changes:

1. **Separate the budgets.** Keep the thinking budget below the event cap, so the thinking can be cut without the
   stream failing (TC-070 change 3).
2. **Resume the turn.** Either:
   - (a) re-request with the thinking so far plus `</think>` as a trailing assistant message, which relies on
     llama-server continuing a final assistant message (assistant prefill; not verified on b9592), or
   - (b) use llama-server's `--reasoning-budget`, which is server-side and needs no client change. It was measured
     in `countdown-thinking-e2e.md`: no empty turns, and the forced answers were wrong.
3. **Retire the TH-010 stand-in for thinking-on sessions.** Showing unclosed thinking as the reply scored 0 of 13
   (stock) and 1 of 12 (adapter) correct on the unclosed turns here. Forcing scored 3 of 13 and 3 of 12.

Given the numbers above, (b) is the cheap user-experience option. None of these changes is an accuracy gain on
this exam.

Runs: `<runs>/posttool-force-{g1,g1b}-{stock,adapter}-001/` (`sample.xml`,
`blanks.json`, `force.xml`, `l1024/force.xml`). Log: `posttool-force-001.log`. Analysis (outside Git, with the runs):
`posttool-force-analysis/analyze.py` (`SUB=l1024/` for the sensitivity table).
