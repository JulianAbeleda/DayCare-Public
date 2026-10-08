"""The runs folder as facts: `python -m daycare.harness.runs {setup,list,show,predeclare,score,adopt}`.

This is the one machine-readable seam between DayCare's records and anything that shows them (the Go TUI in
`tui/`, or an agent). Python stays the only writer of DayCare data; the readers get JSON. Every subcommand
prints one JSON object on stdout and exits 0, or prints `{"error": ...}` and exits 1.

    setup       --root RUNS                       what is missing before a counted run is possible, checked
    list        --root RUNS                       one summary per run folder (predeclared, in progress, done)
    show        --root RUNS/<run>                 the run: gates vs results, the stop-trigger window, verify
    predeclare  --root RUNS/<run> --protocol P    write predeclaration.xml (template run5, or your own gates)
    score       --root RUNS/<run> --gate G1 --diff 15.5 --lo 11.2 --hi 20.0     rule one gate as declared
    adopt       --root RUNS/<run> --by NAME --exception TEXT                     record an owner exception

The predeclaration is the machine copy of the playbook's step 1 (docs/rl-run-playbook.md): gates with a numeric
rule and a prediction, the fixed recipe, written before the first counted update. The research record
`research/<protocol>` stays the committed, human-facing authority; this file is what a screen can check. A gate
is ruled here, once, from the declared rule; a failed gate is never re-scored (no rescue runs).

The stop-trigger window reuses `rl_triggers.Triggers` for the ruling (the trip and its reason come from
`Triggers.check`, never from this file); the per-metric readings beside it are display values built from that
object's own rows, limits and fitted entropy levels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = 'daycare.tui.v1'
PREDECLARATION = 'daycare.predeclaration.v1'
RULES = {'lower_gt': 'CI lower bound > value', 'lower_ge': 'CI lower bound >= value',
         'upper_lt': 'CI upper bound < value', 'upper_le': 'CI upper bound <= value',
         'upper_ge': 'CI upper bound >= value', 'count_le': 'count <= value'}
# Run 6's gate table (research/rloo-posttool-calculator-r6.md), with run 5's measured values as the predictions.
TEMPLATES = {'run5': dict(
    hypothesis='From stock, the run-5 recipe lifts held-out repair without more blanks, keeps blocked behaviour '
               'and retention no worse than stock, and trips no stop trigger.',
    gates=[('G1', 'word-problem repair', 'lower_gt', 0.0, '>= +10 points (run 5: +15.5 [+11.2, +20.0])'),
           ('G1b', 'relay', 'upper_ge', 0.0, 'no clear drop (run 5: +0.8 [-0.6, +2.1])'),
           ('G2a', 'relay-family blanks', 'upper_lt', 0.0, 'fewer blanks (run 5: -3.1 [-4.1, -2.1])'),
           ('G2b', 'relay-family wrong answers', 'upper_le', 2.5, 'not up (run 5: -2.9 [-4.5, -1.3])'),
           ('G5L', 'long-number relay/repair', 'lower_ge', -8.0, 'no worse (run 5: +10.8 [+4.2, +18.6])'),
           ('G5B', 'blocked, rule (c)', 'lower_ge', -8.0, 'no worse, expected better (run 5: +12.3 [+7.7, +16.9])'),
           ('G6', 'blocked, OOD selection probe', 'lower_ge', -9.0, 'no worse (run 5: +10.6 [+5.6, +15.4])'),
           ('G7', 'numeric retention, T=1', 'lower_ge', -5.0, 'holds (run 5 at T=1: level with stock)'),
           ('G3', 'plain retention losses, T=0', 'count_le', 0.0, 'zero plain-answer losses')],
    recipe=['--categories', 'repair', 'relay', 'blocked', '--mix', 'repair=0.75,relay=0.15,blocked=0.1',
            '--mask', 'none', '--reward', 'graded', '--wrong', '-1', '--blank', '-1.5',
            '--abstain', 'relay=-0.5,repair=0,miss=0,empty=0,blocked=0', '--length-weight', '1.0',
            '--repetition', 'n=40,penalty=-0.05', '--kl-aggregation', 'matched', '--kl-beta', '0.03',
            '--lanes', '32', '--prompts-per-update', '4', '--compact', '8,16', '--keep-every', '10',
            '--updates', '102'])}


def _read(path: Path):
    from daycare.artifact.record_xml import read
    return read(path)


def _write(path: Path, value, root: str) -> None:
    from daycare.artifact.record_xml import write
    write(path, value, root=root)


def _finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


# --- gates --------------------------------------------------------------------------------------------------

def rule_passes(rule: str, value: float, diff: float, lo: float | None, hi: float | None) -> bool:
    """One place rules a gate. CI rules read the bound they name; count_le reads `diff` as the count."""
    if rule == 'count_le':
        return diff <= value
    if lo is None or hi is None:
        raise ValueError(f'{rule} needs --lo and --hi')
    return {'lower_gt': lo > value, 'lower_ge': lo >= value, 'upper_lt': hi < value,
            'upper_le': hi <= value, 'upper_ge': hi >= value}[rule]


def gate_result(gate: dict) -> str:
    result = gate.get('result')
    return 'open' if not result else 'pass' if result['passed'] else 'fail'


def verdict(predeclaration: dict | None) -> str:
    if not predeclaration or not predeclaration.get('gates'):
        return 'none'
    results = [gate_result(g) for g in predeclaration['gates']]
    if predeclaration.get('adoption'):
        return 'adopted (exception)' if 'fail' in results else 'adopted'
    return 'fail' if 'fail' in results else 'open' if 'open' in results else 'pass'


def gates_view(predeclaration: dict | None) -> list[dict]:
    return [dict(id=g['id'], name=g['name'], rule=g['rule'], rule_text=RULES[g['rule']], value=g['value'],
                 prediction=g['prediction'], result=gate_result(g), measured=g.get('result'))
            for g in (predeclaration or {}).get('gates', [])]


# --- one run ------------------------------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict:
    """The run record (run.xml when the loop finished, else progress.xml) and the predeclaration, if present."""
    record = next((_read(run_dir / name) for name in ('run.xml', 'progress.xml') if (run_dir / name).exists()), None)
    predeclaration = _read(run_dir / 'predeclaration.xml') if (run_dir / 'predeclaration.xml').exists() else None
    if record is None and predeclaration is None:
        raise FileNotFoundError(f'{run_dir} holds no run.xml, progress.xml or predeclaration.xml')
    return dict(record=record, predeclaration=predeclaration)


def run_state(record: dict | None, finished: bool) -> str:
    if record is None:
        return 'predeclared'
    if record.get('parity_failure'):
        return 'failed'
    if not finished:
        return 'in_progress'
    return 'stopped' if record.get('stopped') else 'complete' if record.get('complete') else 'failed'


def summary(run_dir: Path, loaded: dict) -> dict:
    record, predeclaration = loaded['record'], loaded['predeclaration']
    losses = (record or {}).get('losses') or []
    last = losses[-1] if losses else {}
    config = (record or {}).get('config') or {}
    return dict(id=run_dir.name, state=run_state(record, (run_dir / 'run.xml').exists()),
                updates_done=len(losses), updates_planned=config.get('updates'),
                protocol=config.get('protocol') or (predeclaration or {}).get('protocol'),
                counted=(record or {}).get('counted'), stopped=(record or {}).get('stopped'),
                verdict=verdict(predeclaration),
                gates=[dict(id=g['id'], result=gate_result(g)) for g in (predeclaration or {}).get('gates', [])],
                last=dict(step=last.get('step'), kl=_finite(last.get('kl')), entropy=_finite(last.get('entropy')),
                          mean_reward=_finite(last.get('mean_reward'))),
                adapter=dict(path=str(run_dir / 'adapter.xml'), present=(run_dir / 'adapter.xml').exists(),
                             gguf=(run_dir / 'adapter.gguf').exists(), sha256=(record or {}).get('adapter_sha256')))


def window(run_dir: Path, record: dict | None) -> dict:
    """The rolling stop-trigger readings: last `window` stepped updates vs the first `base`, with each limit."""
    if not record:
        return dict(enabled=False)
    limits = (record.get('config') or {}).get('triggers')
    if not limits:
        return dict(enabled=False)
    from daycare.nursery import rl_triggers as rt
    triggers = rt.Triggers(limits)
    paths = sorted(run_dir.glob('update-*/update.xml'))
    if paths:
        rows, source = [rt.row(_read(p)) for p in paths], 'update records'
    else:
        rows, source = [dict(kl=r.get('kl', math.nan), entropy=r.get('entropy', math.nan),
                             capped_turn_rate=r.get('capped_turn_rate', math.nan),
                             mean_reward=r.get('mean_reward', math.nan),
                             finished_length=r.get('finished_length', math.nan),
                             advantage_length_sum=r.get('advantage_length_sum', math.nan),
                             skipped=bool(r.get('skipped')), shares={}) for r in record.get('losses') or []], 'run record'
    tripped = None
    for number, values in enumerate(rows, 1):
        reason = triggers.check(values)
        if reason and tripped is None:
            tripped = dict(update=number, reason=reason)
    lim = triggers.limits

    def mean(key, rows):
        got = [r[key] for r in rows if math.isfinite(r[key])]
        return sum(got) / len(got) if got else math.nan
    now, base = triggers.rows[-lim['window']:], triggers.rows[:lim['base']]
    ready = len(triggers.rows) >= max(lim['base'], lim['window'])
    entropy_limit = lim['entropy_ratio'] * mean('entropy', base)
    if triggers.levels is not None:
        entropy_limit = lim['entropy_ratio'] * rt.expected_entropy(now, triggers.levels)
    tests = [('kl', '>', lim['kl_max']), ('entropy', '<', entropy_limit),
             ('finished_length', '>', (1 + lim['length_growth']) * mean('finished_length', base)),
             ('capped_turn_rate', '>', min(lim['capped_max'],
                                           max(lim['capped_ratio'] * mean('capped_turn_rate', base), lim['capped_floor']))),
             ('mean_reward', '<', mean('mean_reward', base) - lim['reward_drop'])]
    readings = []
    for key, side, limit in tests:
        value = mean(key, now)
        crossed = ready and math.isfinite(value) and math.isfinite(limit) and (value > limit if side == '>' else value < limit)
        readings.append(dict(metric=key, now=_finite(value), base=_finite(mean(key, base)), limit=_finite(limit),
                             side=side, crossed=crossed))
    return dict(enabled=True, source=source, rows=len(triggers.rows), window=lim['window'], base=lim['base'],
                ready=ready, readings=readings, tripped=tripped)


def show(run_dir: Path) -> dict:
    loaded = load_run(run_dir)
    record, predeclaration = loaded['record'], loaded['predeclaration']
    verify = _read(run_dir / 'verify.xml') if (run_dir / 'verify.xml').exists() else None
    losses = (record or {}).get('losses') or []
    tail = [dict(step=r.get('step'), kl=_finite(r.get('kl')), entropy=_finite(r.get('entropy')),
                 mean_reward=_finite(r.get('mean_reward')), finished_length=_finite(r.get('finished_length')),
                 capped_turn_rate=_finite(r.get('capped_turn_rate')), skipped=bool(r.get('skipped')))
            for r in losses[-10:]]
    return dict(schema=SCHEMA, kind='run', **summary(run_dir, loaded),
                hypothesis=(predeclaration or {}).get('hypothesis'), recipe=(predeclaration or {}).get('recipe'),
                predeclared_at=(predeclaration or {}).get('predeclared_at'),
                adoption=(predeclaration or {}).get('adoption'), gate_table=gates_view(predeclaration),
                window=window(run_dir, record), verify=verify, updates=tail,
                revision=(record or {}).get('daycare_revision'), model_sha256=(record or {}).get('model_sha256'))


# --- the folder ---------------------------------------------------------------------------------------------

def is_run(path: Path) -> bool:
    return path.is_dir() and any((path / n).exists() for n in ('run.xml', 'progress.xml', 'predeclaration.xml'))


def list_runs(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f'runs folder not found: {root}')
    runs, unreadable = [], []
    for p in sorted(root.iterdir()):
        if not is_run(p):
            continue
        try:
            runs.append(summary(p, load_run(p)))
        except (ValueError, KeyError, TypeError, OSError) as error:  # one old record must not hide the others
            unreadable.append(dict(id=p.name, error=f'{type(error).__name__}: {error}'))
    return dict(schema=SCHEMA, kind='runs', root=str(root), runs=runs, unreadable=unreadable)


def _check(id: str, label: str, ok: bool, detail: str, fix: str = '', generate: str | None = None) -> dict:
    return dict(id=id, label=label, ok=bool(ok), detail=detail, fix=fix, generate=generate)


def _metal_memory() -> dict:
    """Apple GPUs share system memory; Metal's recommended working set is what the GPU may use."""
    import ctypes
    import ctypes.util
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
    metal = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')  # links the default device
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    device = metal.MTLCreateSystemDefaultDevice()
    if not device:
        raise OSError('Metal has no default device')

    def send(obj, selector, restype):
        call = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
        return call(obj, objc.sel_registerName(selector.encode()))
    name = ctypes.cast(send(send(device, 'name', ctypes.c_void_p), 'UTF8String', ctypes.c_void_p), ctypes.c_char_p)
    total = send(device, 'recommendedMaxWorkingSetSize', ctypes.c_uint64) / 2**30
    return dict(name=name.value.decode(), total_gb=total, free_gb=total, holders=[])


def _nvidia_memory(smi: str) -> dict:
    query = [smi, '--format=csv,noheader,nounits']
    gpu = subprocess.run(query[:1] + ['--query-gpu=name,memory.free,memory.total'] + query[1:], capture_output=True,
                         text=True, check=True, timeout=10).stdout.strip().splitlines()[0]
    apps = subprocess.run(query[:1] + ['--query-compute-apps=pid,process_name,used_memory'] + query[1:],
                          capture_output=True, text=True, check=True, timeout=10).stdout.strip().splitlines()
    name, free, total = (part.strip() for part in gpu.split(','))
    holders = []
    for row in apps:
        parts = [part.strip() for part in row.split(',')]
        if len(parts) == 3 and parts[2].isdigit():
            holders.append(f'{Path(parts[1]).name} (pid {parts[0]}) holds {int(parts[2]) / 1024:.1f} GB')
    return dict(name=name, free_gb=int(free) / 1024, total_gb=int(total) / 1024, holders=holders)


def _rocm_memory(smi: str) -> dict:
    out = subprocess.run([smi, '--showmeminfo', 'vram', '--showproductname', '--json'], capture_output=True, text=True,
                         check=True, timeout=10).stdout
    card = next(iter(json.loads(out).values()))
    total = int(card['VRAM Total Memory (B)']) / 2**30
    used = int(card['VRAM Total Used Memory (B)']) / 2**30
    return dict(name=card.get('Card series') or card.get('Card SKU') or 'AMD GPU', free_gb=total - used,
                total_gb=total, holders=[])


def gpu_memory() -> dict | None:
    """This machine's first GPU and its memory, read from the GPU itself; None when no GPU answers."""
    if smi := shutil.which('nvidia-smi'):
        return _nvidia_memory(smi)
    if smi := shutil.which('rocm-smi'):
        return _rocm_memory(smi)
    if sys.platform == 'darwin':
        return _metal_memory()
    return None


def gpu_check() -> dict:
    """Training needs a GPU with room for the run, on any backend tinygrad drives. Run 5 trained on one 32 GB card
    (docs/rl-training.md); the floor below is that fact, not a measured peak. DAYCARE_TRAIN_GPU_GB overrides it."""
    need = float(os.environ.get('DAYCARE_TRAIN_GPU_GB', '30'))
    label = f'a GPU with {need:g} GB of memory free for training'
    try:
        gpu = gpu_memory()
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, StopIteration) as error:
        return _check('gpu', label, False, f'the GPU could not be read: {type(error).__name__}: {error}',
                      'check the GPU driver')
    if gpu is None:
        return _check('gpu', label, False, 'no GPU found on this machine',
                      f'train on a machine with a GPU of at least {need:g} GB')
    detail = f"{gpu['name']}: {gpu['free_gb']:.1f} GB free of {gpu['total_gb']:.1f} GB"
    if gpu['free_gb'] >= need:
        return _check('gpu', label, True, detail)
    if gpu['total_gb'] < need:
        return _check('gpu', label, False, f"{gpu['name']}: {gpu['total_gb']:.1f} GB · too small to train (needs {need:g} GB)",
                      f'train on a GPU with at least {need:g} GB of memory')
    if gpu['holders']:
        return _check('gpu', label, False, detail, 'free the GPU: ' + '; '.join(gpu['holders']))
    return _check('gpu', label, False, detail, f"free {need - gpu['free_gb']:.1f} GB of GPU memory")


def setup(root: Path, repo: Path) -> dict:
    """Checked, not guessed: each item is a file, a module or a command this machine has or lacks."""
    checks = []
    for name in ('numpy', 'jinja2'):
        try:
            __import__(name)
            checks.append(_check(name, f'{name} importable', True, sys.executable))
        except ImportError:
            checks.append(_check(name, f'{name} importable', False, 'missing', f'pip install {name}'))
    for module in ('daycare.nursery.rloo_posttool', 'daycare.nursery.rl_triggers', 'daycare.nursery.posttool_tasks'):
        try:
            __import__(module)
            checks.append(_check(module, f'{module} importable', True, 'ok'))
        except Exception as error:  # noqa: BLE001 -- the detail is the point
            checks.append(_check(module, f'{module} importable', False, f'{type(error).__name__}: {error}',
                                 'check out the branch that carries the RL loop (docs/rl-training.md)'))
    checks.append(gpu_check())
    trainer = os.environ.get('DAYCARE_TRAIN_TINYGRAD_PATH', '')
    checks.append(_check('trainer', 'DAYCARE_TRAIN_TINYGRAD_PATH is a tinygrad-arkey checkout',
                         bool(trainer) and (Path(trainer) / 'extra').is_dir(), trainer or 'unset',
                         'git clone https://github.com/JulianAbeleda/tinygrad-arkey; export DAYCARE_TRAIN_TINYGRAD_PATH=<checkout>'))
    model = os.environ.get('DAYCARE_BASE_GGUF', '')
    checks.append(_check('model', 'DAYCARE_BASE_GGUF is a file', bool(model) and Path(model).is_file(), model or 'unset',
                         'convert Nemotron 3 Nano 4B to a BF16 GGUF (docs/rl-training.md); export DAYCARE_BASE_GGUF=<file>'))
    runner = os.environ.get('DAYCARE_CALCULATE_RUNNER', '')
    checks.append(_check('runner', 'DAYCARE_CALCULATE_RUNNER is executable',
                         bool(runner) and os.access(runner, os.X_OK), runner or 'unset',
                         'cd tools/calculate-runner && cargo build --release --locked; export DAYCARE_CALCULATE_RUNNER=<binary>'))
    envelope = os.environ.get('DAYCARE_ENVELOPE', '')
    checks.append(_check('envelope', 'DAYCARE_ENVELOPE is the captured GameTerm request record',
                         bool(envelope) and Path(envelope).is_file(), envelope or 'unset',
                         'the envelope is captured from GameTerm (docs/rl-training.md); it cannot be generated'))
    checks.append(_check('runs_root', 'DAYCARE_RUNS folder exists', root.is_dir(), str(root), f'mkdir -p {root}'))
    states = sorted(root.glob('*/states.xml')) if root.is_dir() else []
    tasks = sorted(root.glob('*/tasks.xml')) if root.is_dir() else []
    checks.append(_check('tasks', 'a frozen task set (tasks.xml) exists', bool(tasks),
                         str(tasks[-1]) if tasks else 'none under the runs folder',
                         'python -m daycare.nursery.posttool_tasks freeze --root <runs>/posttool-tasks-001 '
                         '--no-private-suites --countdown-per-size 0', generate='tasks'))
    checks.append(_check('states', 'post-tool states (states.xml) exist', bool(states),
                         str(states[-1]) if states else 'none under the runs folder',
                         'python -m daycare.nursery.rloo_posttool sample --tasks <tasks root>/tasks.xml --root <harvest> '
                         '(GPU), then python -m daycare.nursery.posttool_tasks states --root <tasks root> '
                         '--harvest <harvest>/sample.xml'))
    waiting = [p.name for p in sorted(root.iterdir()) if is_run(p) and not (p / 'progress.xml').exists()
               and not (p / 'run.xml').exists()] if root.is_dir() else []
    checks.append(_check('predeclaration', 'a predeclared run is waiting to start', bool(waiting),
                         ', '.join(waiting) or 'none',
                         'python -m daycare.harness.runs predeclare --root <runs>/<run> --protocol <research file> '
                         '--template run5 --states <states.xml> --envelope <envelope>', generate='predeclaration'))
    try:
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, check=True).stdout.strip()
        checks.append(_check('git_clean', 'checkout has no uncommitted tracked changes (train refuses otherwise)',
                             not dirty, dirty.splitlines()[0] if dirty else 'clean', 'commit or stash the changes'))
    except (OSError, subprocess.CalledProcessError) as error:
        checks.append(_check('git_clean', 'checkout has no uncommitted tracked changes', False, str(error), ''))
    return dict(schema=SCHEMA, kind='setup', ready=all(c['ok'] for c in checks), checks=checks,
                python=sys.executable, repo=str(repo), root=str(root))


# --- writers -------------------------------------------------------------------------------------------------

def predeclare(args) -> dict:
    run_dir: Path = args.root
    if (run_dir / 'predeclaration.xml').exists():
        raise FileExistsError(f'{run_dir} is already predeclared; a new predeclaration is a new run folder')
    if (run_dir / 'progress.xml').exists() or (run_dir / 'run.xml').exists():
        raise FileExistsError(f'{run_dir} already holds a run; no counted update before the predeclaration')
    template = TEMPLATES.get(args.template) if args.template else None
    if args.template and template is None:
        raise ValueError(f'unknown template {args.template!r}; known: {", ".join(TEMPLATES)}')
    gates = [parse_gate(spec) for spec in args.gate] or [
        dict(id=i, name=n, rule=r, value=v, prediction=p, result=None) for i, n, r, v, p in (template or {}).get('gates', [])]
    if not gates:
        raise ValueError('a predeclaration needs gates: --template run5 or --gate ID|NAME|RULE|VALUE|PREDICTION')
    recipe = list(args.recipe) if args.recipe else list((template or {}).get('recipe', []))
    for flag, value in (('--states', args.states), ('--envelope', args.envelope)):
        if value:
            recipe += [flag, str(value)]
    recipe += ['--protocol', args.protocol]
    record = dict(schema=PREDECLARATION, protocol=args.protocol, template=args.template or '',
                  hypothesis=args.hypothesis or (template or {}).get('hypothesis', ''), recipe=recipe, gates=gates,
                  predeclared_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), adoption=None)
    _write(run_dir / 'predeclaration.xml', record, 'predeclaration')
    return dict(schema=SCHEMA, kind='predeclared', path=str(run_dir / 'predeclaration.xml'), gates=len(gates),
                recipe=recipe)


def parse_gate(spec: str) -> dict:
    parts = spec.split('|')
    if len(parts) != 5:
        raise ValueError(f'gate spec needs ID|NAME|RULE|VALUE|PREDICTION: {spec!r}')
    id, name, rule, value, prediction = parts
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}; known: {", ".join(RULES)}')
    return dict(id=id, name=name, rule=rule, value=float(value), prediction=prediction, result=None)


def score(args) -> dict:
    path = args.root / 'predeclaration.xml'
    record = _read(path)
    gate = next((g for g in record['gates'] if g['id'] == args.gate), None)
    if gate is None:
        raise KeyError(f'no gate {args.gate!r}; declared: {", ".join(g["id"] for g in record["gates"])}')
    if gate.get('result'):
        raise FileExistsError(f'gate {args.gate} is already scored; a failed gate ends the experiment (no rescue runs)')
    passed = rule_passes(gate['rule'], gate['value'], args.diff, args.lo, args.hi)
    gate['result'] = dict(diff=args.diff, lo=args.lo, hi=args.hi, passed=passed, note=args.note or '',
                          scored_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    _write(path, record, 'predeclaration')
    return dict(schema=SCHEMA, kind='scored', gate=args.gate, passed=passed, verdict=verdict(record))


def adopt(args) -> dict:
    path = args.root / 'predeclaration.xml'
    record = _read(path)
    if record.get('adoption'):
        raise FileExistsError('already adopted')
    if any(not g.get('result') for g in record['gates']):
        raise ValueError('every gate must be scored before adoption')
    record['adoption'] = dict(by=args.by, exception=args.exception, at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    _write(path, record, 'predeclaration')
    return dict(schema=SCHEMA, kind='adopted', verdict=verdict(record))


# --- CLI -------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('setup', 'list', 'show', 'predeclare', 'score', 'adopt'))
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('DAYCARE_RUNS') or 'runs'),
                        help='setup/list: the runs folder (default $DAYCARE_RUNS); the others: one run folder')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--protocol', help='predeclare: the research record this run belongs to')
    parser.add_argument('--template', help='predeclare: a gate and recipe template (run5)')
    parser.add_argument('--hypothesis', default='')
    parser.add_argument('--gate', action='append', default=[], metavar='ID|NAME|RULE|VALUE|PREDICTION',
                        help=f'predeclare: a gate; RULE in {", ".join(RULES)}. score: the gate id')
    parser.add_argument('--recipe', nargs=argparse.REMAINDER, help='predeclare: the train flags, verbatim (last)')
    parser.add_argument('--states', type=Path, help='predeclare: states.xml for the recipe')
    parser.add_argument('--envelope', type=Path, help='predeclare: the envelope record for the recipe')
    parser.add_argument('--diff', type=float, help='score: the paired difference, or the count for count_le')
    parser.add_argument('--lo', type=float, help='score: CI lower bound')
    parser.add_argument('--hi', type=float, help='score: CI upper bound')
    parser.add_argument('--note', default='')
    parser.add_argument('--by', help='adopt: who records the exception')
    parser.add_argument('--exception', help='adopt: the exception, with its evidence and known costs')
    args = parser.parse_args(argv)
    try:
        if args.action == 'setup':
            out = setup(args.root, args.repo)
        elif args.action == 'list':
            out = list_runs(args.root)
        elif args.action == 'show':
            out = show(args.root)
        elif args.action == 'predeclare':
            if not args.protocol:
                parser.error('predeclare needs --protocol')
            out = predeclare(args)
        elif args.action == 'score':
            if len(args.gate) != 1 or args.diff is None:
                parser.error('score needs --gate ID and --diff (and --lo/--hi for CI rules)')
            args.gate = args.gate[0]
            out = score(args)
        else:
            if not (args.by and args.exception):
                parser.error('adopt needs --by and --exception')
            out = adopt(args)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps(dict(schema=SCHEMA, kind='error', error=f'{type(error).__name__}: {error}')))
        return 1
    print(json.dumps(out, indent=None, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
