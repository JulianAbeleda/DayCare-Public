# daycare-tui

The DayCare runs folder on a screen, or as JSON. One Go binary, two modes:

- a human runs `daycare-tui` and gets Bubble Tea screens;
- an agent runs `daycare-tui --json <command>` and gets one JSON object and an exit code.

Both modes read through the same seam, `python -m daycare.harness.runs`. Python owns compute and data and is
the only writer of DayCare records. Go owns the screen and process control (start, stop, tail). No code is
shared across the boundary; the contract is a fixture run folder and its expected JSON, pinned by a test on
each side.

## Quickstart (human)

```bash
cd tui && go build -o daycare-tui .            # Go 1.26
export DAYCARE_RUNS=/path/to/runs               # the folder that holds <run>/ folders (outside Git)
export DAYCARE_PYTHON=/path/to/venv/bin/python  # an interpreter with numpy, jinja2 and the RL modules
./daycare-tui
```

Keys: `1` setup, `2` runs, `3` adapters, `4` job, `j`/`k` move or scroll, `enter` open a run, `esc` back,
`s` start the opened run, `x` stop it, `g` generate the selected missing item, `t` plain or technical wording,
`r` refresh, `q` quit. The footer lists them.

The look: a Charm-style palette (pink, purple, cyan, mint) as lipgloss adaptive tokens in
[`internal/ui/theme.go`](internal/ui/theme.go), rounded boxes with gradient titles, glyphs for every outcome
(`✓` pass, `✗` fail, `⚠` crossed or stopped, `⏸` waiting, `○` open, `●` running), a bar for updates n/N and a
"near limit" bar for each trigger reading (full means at the limit), and a small plush in the header whose face
follows the state: `(˘ω˘) zz` when the nursery is empty, `(•ᴗ•)` idle, `(•̀ᴗ•́)✧` while a run trains,
`(≧◡≦)♡` when a run passed, `(•́︿•̀)` when a trigger crossed or a gate failed. Colours degrade through lipgloss
to the terminal's profile and disappear under `NO_COLOR`; the layout fits 80x24 (long cells are cut with `…`).

Screens:

1. **Setup.** Each line is a check, not a guess: a file, an importable module, an executable, a clean checkout.
   A missing item shows the fix. Two items have a generator (`g`): a frozen task set (`posttool_tasks freeze`,
   no private suites, no Countdown, no network) and a predeclaration (template `run5`). The GameTerm envelope
   cannot be generated; the states need a GPU harvest, and the fix text gives both commands.
2. **Runs.** One row per run folder: state, updates done / planned, verdict, and the gates as pass / fail / open.
3. **Run.** The hypothesis, each gate's rule and prediction next to what was measured, the stop-trigger window
   (the last N stepped updates against the first N, with each limit and whether it was crossed), the trip
   reason, the adapter and the bit-exact reload check. Technical mode adds the recipe, the digests and the last
   update rows.
4. **Adapters.** The runs that saved an adapter (or recorded one that is no longer in the folder).
5. **Job.** The process this machine started for the opened run, with the tail of its log.

Plain mode uses short literal words (`drift from stock (KL)`); technical mode uses the record's own names.

## Agent mode: the JSON contract

```
daycare-tui [--json] [--root RUNS] [--repo DIR] [--python PY] [--state DIR] <command>
```

| Command | Prints | Exit |
|---|---|---|
| `status` | `{"kind":"setup","ready":bool,"checks":[{id,label,ok,detail,fix,generate}]}` | 0 ready, 3 not ready |
| `runs` | `{"kind":"runs","root","runs":[summary]}` | 0 |
| `run <id>` | the summary plus `hypothesis, recipe, gate_table, window, verify, updates, adoption` | 0, 1 unknown run |
| `generate tasks [--name N]` | `{"kind":"generated","thing":"tasks","path","output"}` | 0 / 1 |
| `generate predeclaration --name N --protocol P [--template run5] [--states S] [--envelope E] [--gate SPEC]... [-- recipe...]` | `{"kind":"predeclared","path","gates","recipe"}` | 0 / 1 |
| `start <id>` | the job: `{id,pid,alive,log_path,started_at,argv,dir}` | 0, 1 when not predeclared |
| `stop <id>` | the job | 0, 1 when not running |
| `tail <id> [--lines N]` | `{"kind":"job","job":{...},"lines":[...]}` | 0 |
| `score <id> --gate G --diff D [--lo L --hi H] [--note T]` | `{"kind":"scored","gate","passed","verdict"}` | 0 / 1 |
| `adopt <id> --by NAME --exception TEXT` | `{"kind":"adopted","verdict"}` | 0 / 1 |

Every error is `{"kind":"error","error":"...","stderr":"..."}` with exit 1; a usage error exits 2 and prints
the usage on stderr. `status`, `runs`, `run`, `generate predeclaration`, `score` and `adopt` print Python's
bytes unchanged, so the agent reads the same facts the screens were built from.

A run summary:

```json
{"id": "posttool-fixture-001", "state": "stopped", "updates_done": 6, "updates_planned": 102,
 "protocol": "rloo-posttool-fixture-001.md", "counted": true,
 "stopped": {"update": 6, "reason": "entropy 0.44 < 0.4675 (mean of the last 3 stepped updates)"},
 "verdict": "fail", "gates": [{"id": "G1", "result": "pass"}, {"id": "G3", "result": "fail"}],
 "last": {"step": 6, "kl": 0.008, "entropy": 0.4, "mean_reward": 0.86},
 "adapter": {"path": ".../adapter.xml", "present": false, "gguf": false, "sha256": "..."}}
```

`state` is one of `predeclared | in_progress | complete | stopped | failed`; `verdict` is
`none | open | pass | fail | adopted | adopted (exception)`. The full shapes are in
[`testdata/expected/`](testdata/expected/) and typed in [`internal/seam/types.go`](internal/seam/types.go).

## What the seam adds to DayCare

`daycare/harness/runs.py` is the one Python-side addition. It reads the records the loop already writes
(`run.xml`, `progress.xml`, `update-NNN/update.xml`, `verify.xml`; all `daycare.record.v1` XML) and owns one new
record, `predeclaration.xml`, the machine copy of the playbook's step 1:

- `predeclare` writes the gates (id, name, rule, value, prediction), the hypothesis and the fixed recipe before
  the first counted update. It refuses a folder that already holds a predeclaration or a run. Rules:
  `lower_gt`, `lower_ge`, `upper_lt`, `upper_le`, `upper_ge` on a paired CI, `count_le` on a count.
- `score` rules one gate from its declared rule and the measured difference and interval. A scored gate is
  never re-scored: a failed gate ends the experiment (no rescue runs).
- `adopt` records an owner exception, only once every gate is scored. The verdict then reads
  `adopted (exception)` when a gate failed, never `pass`.

The research record `research/<protocol>.md` stays the committed, human-facing authority; the XML is what a
screen can check (coding principle: human-facing and machine-enforced, both).

The stop-trigger window reuses `rl_triggers.Triggers`: the trip and its reason are that class's ruling, replayed
on the saved update records (or on the run record's `losses` rows when a run has no update folders). The
readings beside it are display values from the same object's rows, limits and fitted entropy levels.

`start` runs `python -m daycare.nursery.rloo_posttool train --root <run> <recipe>` detached, with the log and
pid under `--state` (default `~/.local/state/daycare-tui`), never inside the run folder. `stop` sends SIGTERM;
the loop writes `progress.xml` after every update and a raw checkpoint every `--keep-every` updates.

## Tests

```bash
cd tui && go test ./...                                          # screens, contract, jobs
DAYCARE_PYTHON=/path/to/venv/bin/python go test ./...            # also the live seam against the fixture
python -m pytest tests/test_harness_runs.py                      # the Python side of the same contract
```

`testdata/fixture/` is the fixture: a stopped run with update records and a partly scored predeclaration, a
predeclared run, a run in progress, and a sample arm that is not a run. `testdata/expected/*.json` is the
seam's output on it; both test suites compare against those files. `testdata/screens/*.txt` are the rendered
screens (`go test ./internal/ui -update` rewrites them). The live Go test and the Python test skip when the
interpreter lacks numpy or the RL modules (`daycare.artifact.record_xml`, `daycare.nursery.rl_triggers`).

Go tests live beside the code (`_test.go`) because Go's toolchain treats them as part of the package; they
travel with the product on `main`. The Python test follows the branch rule and lives on `dev`.
