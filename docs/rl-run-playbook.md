# RL run playbook

How to run a counted training experiment (RLOO/RFT/SFT) so its result can be trusted. Worked example:
[rloo-posttool-calculator.md](../research/rloo-posttool-calculator.md). Loop: `daycare/nursery/rloo_tinygrad.py`
(one stack: tinygrad-arkey `exp` samples and trains), post-tool episodes: `daycare/nursery/rloo_posttool.py`.
(These training modules, `rl_triggers.py` and the tests named below are not yet in the public repository; they
will be published with the training code.)

## 1. Predeclare (before any counted update)

Write `research/<name>.md` with **Hypothesis -> Process -> Finding**, Finding left empty:

- Hypothesis: one mechanism and a numeric prediction per gate (e.g. "repair >= +10 points, blanks <= stock").
- Process: tasks and splits (held-out by source task), every fixed setting, the starting weights (**stock**
  unless the file says otherwise; never chain adapters silently), and prior art cited by arXiv id, also in code
  comments where the policy lives.
- Gates, written before seeing data: efficacy = task-clustered bootstrap 95% CI of the paired stock-vs-adapter
  difference, lower bound > 0; a no-regression gate for what is at ceiling; the side-effect gates; retention on
  the GameTerm suite. Stop rule: **a failed gate ends the experiment; no rescue runs.** The next idea is a new
  predeclared run.
- Before predeclaring a new loss, mask or reward policy: **replay it on the previous run's saved records**
  (`update-*/update.xml`) and state which terms move (e.g. sum(A x length) per block; the run-2 audit,
  [rloo-posttool-audit-20260927.md](../research/rloo-posttool-audit-20260927.md), found the lost length brake
  this way, after the fact). **Compute each loss term's gradient share** (per-token weight vs the policy
  gradient's; logged as `kl_weight_ratio`): a coefficient means nothing until its share is known (KL was ~1e-5
  of PG in runs 1-2; `kl_aggregation: matched` fixes it, `legacy_token_mean` reproduces runs 1-2).
- Stop triggers (`daycare/nursery/rl_triggers.py`, `--triggers k=v,...`): means of the last 10 stepped updates vs
  the first 10: KL > 0.025, entropy < 0.85x, finished-episode length > 1.5x, capped-turn rate > 15% or > max(2x
  baseline, 5%), outcome reward < baseline - 0.2, `advantage_length_sum` net positive over the window (or positive
  10 updates in a row); 3 consecutive skipped updates. A working KL beta can keep the KL trigger quiet, so the length
  triggers must stop a runaway on their own. Replayed under run 3's reward and mix: run 2 trips at u32 (collapse
  u45), run 1 at u98, and with KL/entropy silenced run 2 still at u32, run 1 never. Declare any override and replay
  it on prior runs (`python -m daycare.nursery.rl_triggers RUN --run3-reward --categories repair relay`).
- **Calibrate every instrument before predeclaring** (runs 4 and 6 were stopped, and run 5 failed, by instruments sitting
  inside their own noise band). Every gate and every stop trigger states, from existing records and without a new
  training run: its false-alarm rate on known-healthy runs (runs 3-6: at most 10%; "did not trip" is not enough, give
  the margin and how it was estimated) and whether it catches the known-bad run (run 2). Gates are sized from stock's
  measured variance (M >= 3.24 SE); a single-sample gate is not a gate. An instrument that cannot show both numbers
  is reported, not gated.
- The maintainer marks it predeclared. No counted update before that.

## 2. Prove the outputs (P2), before the long run

One update, export, then `verify` in a fresh process: the saved adapter must reload and reproduce bit-exactly.
Adapters were lost twice (no `--export`; a KeyError at save). Keep raw checkpoints (`--keep-every`) and back
up the raw adapter before any conversion.

## 3. Run

- GPU: `gpu-run time <cmd>` (a local GPU-lock wrapper, not published) for counted runs and timings (exclusive), `gpu-run check` for
  correctness runs. Never run GPU work outside it; it caps host RAM at 24 GB (a host OOM crashed the box).
- Outputs go to a run directory outside Git (`<runs>/<run>-NNN/`), never Git or the root disk.
- Never edit code that a running job imports: edit in a worktree, merge after the run.
- Log progress in the file's `## Log` (update ranges, mean reward, capped-turn rate, wall time).
- Stop early only on the predeclared triggers (parity violation, non-finite value, the rolling triggers above;
  the loop stops itself, keeps the adapter and writes `stopped` into the record); record it in the Log.

## 4. Evaluate and report

Score every gate exactly as declared. Write the Finding as GameTerm scenarios, pre (stock) vs post (adapter):

- define each category in one line (relay: the tool result is the answer; repair: the calculator rejected the
  input; miss: the result is not the answer yet; empty: the model gave no answer);
- show each category's share of the exam as a % **and** a question count, then pre -> post score with the CI;
- one real transcript excerpt per category that moved;
- side effects: blanks, capped turns, answer length.

Report failures as failures. A number that contradicts an earlier claim gets corrected in the doc and to the maintainer.

## 5. Land it

Run `python3 sz.py` and the tests of the touched modules, commit to the experiment branch (docs + code, not weights), and
push when approved. The adapter is adopted only if every gate passes, or by an owner exception that the maintainer states in writing, recorded in the run file as an exception (never as a pass) with its evidence and known costs (first: run 5, `posttool-adopted-001`); the final deployable adapter is a staged
curriculum of passed recipes, trained from stock.

## Stop spinning

If two consecutive predeclared runs fail the same gate for the same reason, stop and bring the maintainer the
diagnosis. Do not start a third variant. Check the literature (`research/reports/`) before designing a fix.
