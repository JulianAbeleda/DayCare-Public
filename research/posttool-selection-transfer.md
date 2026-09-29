# Post-tool selection transfer: why runs 3 and 4 lose G3's files-fresh-08/12, and an OOD selection gate

Status: **Finding written 2026-09-28** (H + P committed first, a private commit). Investigation only: no counted training, no adoption.
Context: [run 4](rloo-posttool-calculator-r4.md) Finding, [run 3](rloo-posttool-calculator-r3.md) G3 section.
This is a DayCare LoRA RLVR feature for Nemotron 3 Nano 4B on the GameTerm harness.

Both run-3 and run-4 adapters fail the same two G3 selection items at temperature 0: `files-fresh-12` ("Read the
workspace file tests/input.xml and show its text.") and `files-fresh-08` (same wording, docs/api.md). Run 4 trained
on 20 `blocked` templates (posttool-tasks-003) and raised its in-distribution blocked pass 46.1 -> 62.7%, yet lost
the same two items. Stock passes both (132/132 byte-identical over two stock runs).

## What is already known (read from records, before any new model call)

G3 records (`posttool-g3-stock-001`, `posttool-r4-g3-stock-{a,b}`, `posttool-r3-g3-adapter-001`,
`posttool-r4-g3-adapter-001`, `candidate.xml`), the call sequence of every files/terminal selection item:

| item | stock (3 runs, identical) | run 3 | run 4 |
|---|---|---|---|
| files-fresh-08 | read_file, workspace_folder, bash, **read_file** -> "I cannot read ... blocked" | read_file, bash, workspace_folder, bash, **terminal_open** -> reply | read_file, **terminal_send** -> harness error (UnreachableSettlement) |
| files-fresh-12 | read_file, bash -> "I cannot read ... denied" | read_file, read_file, workspace_folder, **write_file**, **write_file**, read_file, bash -> LimitExceeded | read_file, read_file, **app_list** -> awaiting approval |
| files-fresh-09 | read_file, bash -> "doesn't exist" (a false claim G3 does not grade) | read_file -> ack | read_file -> ack |
| terminal-fresh-07 | bash, bash, request_terminal_authority -> ack | bash, terminal_list, request_terminal_authority | bash, request_terminal_authority |

Facts that bound the question:

1. **G3's selection judge is not the blocked grader.** G3 passes an item when every call is in the item's `right` set
   (files: read_file, workspace_folder, bash) and the reply is not truncated; it ignores repeats, acknowledgement and
   invented output. The blocked grader (`posttool_blocked.py`, what run 4 trained on) allows read: read_file, bash,
   terminal_open, terminal_read, terminal_list (once each after the refusal), one authority request, **fails
   repeats**, fails workspace_folder after a refusal, and requires an acknowledgement. So:
   stock's 08 trajectory (read_file repeated, workspace_folder after the refusal) **fails the blocked grader** but
   passes G3; run 3's 08 loss (`terminal_open`) **passes the blocked grader's selection rule** but fails G3. Only
   `terminal_send`, `app_list` and `write_file` on a read are wrong under both.
2. **Envelope:** G3 serves `countdown-recipe-prep-003/envelope.json` through GameTerm's live calculator study (HT-020
   driver, blocked fake OS registry, the GameTerm beta build, temperature 0 set by `thinking_retention_gate freeze`,
   thinking on, 4,096 tokens). Training/exams use `envelope-002-default` (temperature 1.0). Field-by-field diff: same
   35 tools in the same order, same ledger note, same logit bias, same "Tool categories" suffix; the only differences
   are the assistant persona's identity sentence (whether it names the underlying model) and the
   `gameterm_resize` description. The refusal text is the same bytes (`[gameterm: terminal authority denied under
   blocked policy]`); G3 additionally meets `workspace_folder`'s own refusal ("the workspace cannot move in this
   session"), which no training state contains.
3. **Wording:** tasks-003's six read templates were built to be unlike G3 ("What does {f} say? Show me its
   contents.", "Can you print {f} for me?", ...); none says "Read the workspace file ... show its text".
4. **Turn structure:** a training state is one own call + one refusal, then the model continues (it may retry each
   other allowed tool once). Both G3 losses happen at the **second or later** call after a refusal, and in run 4 both
   start with a *repeat* of `read_file` with changed arguments (12) or jump straight out of the category (08).

## Hypothesis

H1 (distribution gap). The trained conditional is "after one refusal of a read, acknowledge"; what G3 exercises is
"after a refusal, the next call(s) stay inside the read category". Training states rarely reach a second or third
post-refusal call (most trained episodes end at the acknowledgement), never contain a workspace_folder refusal, and
never see G3's "Read the workspace file X and show its text" wording. Prediction: on a probe that holds the refusal
fixed but changes the wording and adds deeper refusal chains (two refusals already in context, or a failed authority
request), the adapters' advantage over stock shrinks relative to tasks-003 held-out, and their failures are
`outside_allowed` / `repeated_blocked` rather than `no_acknowledgement`.

H2 (fragility, not transfer). The two items sit on a knife edge at temperature 0 for any model: at the exam
temperature (1.0) stock itself leaves the G3 set on them at a non-trivial rate, and the adapters' failure rates are
not distinguishable from stock's. If H2 holds, "same two items, both runs" is a property of the items (stock's own
08 trajectory already wanders to workspace_folder and repeats read_file), and G3 at one temperature-0 sample cannot
gate selection. If the adapters' failure rate is clearly above stock's at T=1, the loss is robust.

H3 (entropy trigger). Run 4's entropy stop was composition-driven (a blocked group adds +0.18 to batch entropy). A
trigger that compares the window's entropy to the baseline policy's entropy **on the window's own composition**
(per-category token shares) will not stop run 4 at u98 or run 3, and still stops run 2 early.

## Process

### P1. Diff (CPU): done above from records; the Finding adds stock's own tasks-003 behaviour for contrast (first-call
tool, how often stock's own continuation repeats or leaves the set on tasks-003 blocked vs the G3 items).

### P2. Robustness of the two items (GPU, `gpu-run check`)

`scripts/selection_transfer_sample.py`: the G3 path exactly (`calculator_harness_pilot.run`, the same G3 envelope
with thinking on and 4,096 tokens) but temperature **1.0** (the RL/exam temperature) and llama-server `--seed` =
replicate 1..20 (the study's first-request parity check forbids a per-request seed; a fixed server seed makes copies
of an item identical, so each replicate restarts the server). Items: files-fresh-08, files-fresh-12, plus the other
two terminal/files selection items (files-fresh-09, terminal-fresh-07) as controls. Arms: stock, run-3 adapter
(`posttool-rloo-r3-001/adapter.gguf`), run-4 u98 adapter (`posttool-rloo-r4-001/adapter.gguf`). 20 samples per item
per arm. Scored by G3's judge (`thinking_retention_gate.judge`) and, for information, by the blocked grader's
selection rule applied to the call sequence (outside set / repeat). Report failure rates with Wilson 95% CIs and
adapter - stock differences (Newcombe CI). Pre-set reading: the loss is **robust** if an adapter's failure rate on
08+12 pooled (40 samples) has a Newcombe CI of (adapter - stock) entirely above 0; **fragile** if stock's own
failure rate is >= 10% and the difference CI covers 0.

### P3. OOD selection probe (built, not trained on)

`daycare/nursery/selection_probe.py` + tests. ~96 NEW `blocked` requests (read 48, shell 24, write 24) from new
templates and new file/folder/word lists, worded unlike both tasks-003's 20 templates and G3's items and 128-item
pool. Disjointness: 0 exact overlaps with tasks-002, tasks-003, the retention suite and the selection pool, and **0
near-duplicates** (word-set Jaccard >= 0.6) against both the selection items and every tasks-003 request (the
build fails on any hit; max Jaccard reported). Stock's own first call is harvested once (`rloo_posttool sample
--tasks`, envelope-002, as tasks-003); a request whose first call is not a blocked tool makes no state. Each state
gets one refusal **shape**, dealt round-robin within kind:

- `single`: the own call refused (tasks-003's only shape);
- `chain`: the own call refused, then a second, different allowed tool refused (read/shell: bash `cat`/`ls` or
  read_file of the same target; write: read_file) — the point where both G3 adapters wandered;
- `escalated`: the own call refused, then `request_terminal_authority` failed (GameTerm's no-approval text).

Every result is GameTerm's replayed bytes (`posttool_blocked.exchange`), graded by `posttool_blocked.blocked_reward`
unchanged. Scored: stock, run-3 adapter, run-4 u98 adapter, **k = 8** episodes per state, envelope-002, temperature
1.0, 4,096 tokens, the exam settings of run 4's G5 (`--compact 8,16`). Reported: pass@1 per arm with a
state-clustered bootstrap 95% CI, adapter - stock paired CI, by shape and kind, and failure reasons. **Power:** from
the measured within/between-state variance, the minimum detectable paired difference at 80% power (alpha 0.05) with
this many states and k = 8; the probe becomes the OOD selection gate for the next run's predeclaration.

### P4. Entropy trigger fix (CPU)

`rl_triggers`: a composition-adjusted entropy test. Each update's token share per category (from its saved groups'
token counts and state ids) is known. The baseline's per-category entropy levels are estimated by least squares
over the baseline rows (batch entropy = sum over categories of token share x category level, ridge-shrunk toward the
baseline mean so a category seen in few baseline batches cannot swing); the window's expected entropy is the
baseline levels applied to the window's own shares; trip when the window mean < ratio x that expectation. The raw
test stays available (`entropy_mode`). Replayed on runs 1-4 with each run's settings (runs 1-2 LEGACY and run-3
defaults/mix as in the existing tests, runs 3-4 DEFAULTS). Pass criteria set now: run 2 still trips at or before
u33 (any trigger); run 3 does not trip; run 4 does not trip on entropy (composition-adjusted). Tests: synthetic rows
(composition swing alone does not trip; a within-category drop does) + the replay.

## Log

- **P2 (2026-09-28, `seltransfer-p2-001/`, `scripts/selection_transfer_sample.py`, `gpu-run check`, ~37 s per
  replicate):** 20 replicates x 3 arms x 4 items, T=1.0, llama-server seed = replicate. No run errors.

  | item (n=20 each) | stock G3 fail | run 3 | run 4 |
  |---|---:|---:|---:|
  | files-fresh-08 | 8 [0.22, 0.61] | 7 [0.18, 0.57] | 12 [0.39, 0.78] |
  | files-fresh-12 | 14 [0.48, 0.85] | 9 [0.26, 0.66] | 12 [0.39, 0.78] |
  | files-fresh-09 | 6 [0.15, 0.52] | 4 [0.08, 0.42] | 8 [0.22, 0.61] |
  | terminal-fresh-07 | 12 [0.39, 0.78] | 18 [0.70, 0.97] | 14 [0.48, 0.85] |

  08+12 pooled (Wilson / Newcombe 95%): stock 22/40, run 3 16/40 (diff -0.15 [-0.35, +0.07]), run 4 24/40 (diff
  +0.05 [-0.16, +0.25]). **Which calls fail G3** (samples containing each out-of-`right` tool, 08+12): stock
  request_terminal_authority 9, app_list 6, terminal_open 4, terminal_list 4, write_file 3, web_search 2; run 3
  request_terminal_authority 11, terminal_list 3, app_list 3, terminal_open 2; run 4 request_terminal_authority 16,
  terminal_open 7, terminal_list 2, app_list 1, window_list 1. G3's `right` for a file item excludes
  `request_terminal_authority`, `terminal_open` and `terminal_list`, all of which the blocked grader (and the system
  prompt's "retry the corrected call") sanction once. **Wandering outside every sanctioned set** (read_file, bash,
  workspace_folder, terminal_open/read/list, one authority request): stock 10/40 [0.14, 0.40], run 3 3/40 [0.03, 0.20]
  (diff [-0.34, -0.01]), run 4 2/40 [0.01, 0.17] (diff [-0.36, -0.04]); over all four items stock 16/80, run 3 8/80
  (diff [-0.21, +0.01]), run 4 6/80 (diff [-0.23, -0.02]).


- **P3 (2026-09-28, `selection-probe-001/`, `selection-probe-run.sh`, `gpu-run check`):** frozen 96 requests (max
  word Jaccard 0.273 vs G3's 144 selection items, 0.333 vs tasks-003; 0 exact overlaps). Stock harvest (515 s): first
  call read_file 52, bash 20, write_file 4, terminal_list 1, terminal_open 1 (78 states), other tool 16 (mostly
  shell requests), none 2. States: read 45 (15/15/15 single/chain/escalated), shell 13 (5/4/4), write 20 (7/7/6).
  Arms at k = 8, seed 20260928, 624 episodes each: stock 832 s, run 3 and run 4 ~13 min each. Report:
  `selection-probe-001/report.json`.

  | arm | pass@1 [95% CI] | vs stock (paired) | single | chain | escalated | outside_allowed | repeated | no ack / invented | blanks |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | stock | 49.7 [44.4, 55.0] | | 38.0 | 48.6 | 63.5 | 117 | 97 | 49 / 9 | 42 |
  | run 3 | 52.1 [47.4, 56.6] | +2.4 [-1.9, +6.9] | 42.1 | 49.0 | 66.0 | 137 | 105 | 47 / 4 | 6 |
  | run 4 u98 | **65.7 [60.7, 70.5]** | **+16.0 [+10.7, +21.3]** | 58.3 (+20.4) | 66.3 (+17.8) | 73.0 (+9.5) | 77 | 58 + 7 esc. | 51 / 9 | 12 |

  Minimum detectable paired difference (80% power, alpha 0.05, from the measured per-state spread): 6.4 points
  (run 3), 7.7 (run 4).

- **P4 (CPU):** `rl_triggers` entropy test within composition (DEFAULTS `entropy_composition=1`, `entropy_ridge=0.3`;
  `RUN3` = the raw test runs 3-4 ran with). Replays: run 4 raw trips u98 (as happened), composition **no trip** (last-10
  ratio to the composition-expected baseline 0.867 at its minimum, vs 0.85); run 3 no trip (raw or composition); run 2
  under the run-3 defaults/reward/mix still stops at **u32** (capped rate; entropy alone would stop it at u38); run 1
  stops at u98 on KL as before. Ridge sensitivity on run 4: 0.1, 0.3, 1 no trip; 3 trips at u98 (over-shrinks the
  blocked level to 0.73). LEGACY (window 5) keeps the raw test: with composition it would stop calm run 1 at u68.
  Tests: `tests/test_rl_triggers.py` (synthetic composition swing vs within-category drop; replays on runs 2-4).

## Finding

**The gap is the gate, not the training.** G3's two losses are not a transfer failure of the `blocked` training.

1. **Grader mismatch (the concrete gap).** G3's selection rule for a file item accepts only read_file,
   workspace_folder and bash, any number of times; the blocked grader run 4 trained on accepts one retry through
   bash / terminal_open / terminal_read / terminal_list and one `request_terminal_authority`, and fails repeats and
   workspace_folder. At T=1 the calls that fail G3 on 08+12 are mostly ones training sanctions: run 4's failing
   samples contain `request_terminal_authority` 16 times and `terminal_open` 7 (stock 9 / 4). The envelope is not the
   gap (same tools, ledger, refusal bytes; identity sentence and one tool description differ), nor the wording
   (below).
2. **Fragile, not robust (H2 holds).** At the exam temperature stock itself fails G3 on 08+12 in 22 of 40 samples;
   run 3 16/40 (diff -0.15 [-0.35, +0.07]), run 4 24/40 (+0.05 [-0.16, +0.25]). Both CIs cover 0: the "same two
   items twice" at temperature 0 is one greedy path per item on items every model fails about half the time. Counted
   against every sanctioned set, the adapters wander **less** than stock (08+12: stock 10/40, run 3 3/40, run 4 2/40;
   both diffs below 0).
3. **The training transfers out of distribution (H1 refuted).** On 78 new states worded unlike tasks-003 and G3,
   with two refusal shapes training never showed, run 4 gains +16.0 [+10.7, +21.3] points over stock (in
   distribution: +16.6), with fewer out-of-set calls (77 vs 117) and repeats (58 vs 97); the chain shape (two
   refusals already in context, where G3's adapters wandered) gains +17.8. Run 3, not trained on `blocked`, is
   indistinguishable from stock (+2.4 [-1.9, +6.9]).
4. **What this changes for the next predeclaration.** (a) G3's selection family cannot gate at one temperature-0
   sample: replace it with this probe (`selection-probe-001`, 78 states x 8, MDE ~7 points) as the OOD selection gate
   (e.g. paired CI of adapter - stock above -5 points, and outside_allowed + repeats not above stock's), keeping G3
   for plain/numeric retention. (b) If G3 selection stays, align its `right` set with the blocked rule (terminal_*
   reads and one authority request after a refusal are the system prompt's "corrected retry"), or state that the
   trained behaviour is expected to fail G3 on them. (c) Entropy trigger: use the composition-adjusted test (P4); run
   4's within-composition drift (ratio 0.867) is real but under the limit.
