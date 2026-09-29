"""Post-tool calculator tasks: where reading the result is the work (research/rloo-posttool-calculator.md).

Four categories of post-tool state, each a GameTerm later-turn request (first request + one or more
calculate exchanges in the wire envelope):

- `miss`     Countdown: the called expression's result is not the target; the model must compare and go on
             searching (the post-result probe's construction when the model's own call is not a miss).
- `relay`    word problems with large or decimal numbers whose answer is NOT the raw calculator result
             (round up for whole vans, floor for full jugs, a remainder, cents, a percentage to one place,
             the gap to a goal): the result has to be mapped back to the question.
- `repair`   the call was rejected (GameTerm's schema text) or refused by the calculator: fix and retry, or
             answer from reasoning.
- `empty`    states that historically ended in an empty answer after a calculator call with thinking on
             (post-result probe, wire format): the probe's miss and rejected states on those tasks.

`freeze` writes the task list and the fixed train / held-out split by task (never the 132-item retention suite)
before any model call; `rloo_posttool sample --tasks` (GPU) samples the stock model's own first turn on every
miss/relay task; `states` turns each own `calculate` call into a state with its real result and freezes them.

    python -m daycare.nursery.posttool_tasks freeze --root R
    python -m daycare.nursery.posttool_tasks states --root R --harvest H/sample.xml
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

from daycare.artifact.record_xml import read, write

# Inputs from earlier DayCare runs (retention suite, selection pool, used Countdown tasks, probe history) are not
# published; DAYCARE_RUNS names the directory that holds them. `freeze` refuses to run without the two it checks.
RUNS = Path(os.environ.get('DAYCARE_RUNS') or '$DAYCARE_RUNS')
RETENTION_SUITE = RUNS / 'large-model-comparison-001/suite.json'
PREP = RUNS / 'countdown-recipe-prep-003'
HISTORY = [RUNS / 'post-result-probe-wire-on-001/stock.xml', RUNS / 'post-result-probe-think-001/stock.xml']
# Countdown tasks already used anywhere (SFT, RL, probes, the three holdouts): excluded from the fresh pool
USED_COUNTDOWN = [PREP / 'sft-tasks.json', PREP / 'rl-tasks.json', PREP / 'probe-tasks.json',
                  *(RUNS / f'countdown-holdout-4num-00{i}/tasks.xml' for i in (1, 2, 3))]
COUNTDOWN_OFFSET = 420000  # TinyZero public test region, past every earlier holdout (330000-350000)
# v1 read ' Finish with <answer>the number</answer>.': stock copied the placeholder literally (posttool-headroom-001)
ANSWER = ' End your reply with the final number alone inside answer tags, like <answer>42</answer>.'
SEED = 20260926
HELDOUT = 0.3


def _money(value: Fraction) -> str:
    return f'{Decimal(value.numerator) / Decimal(value.denominator):.2f}'


def _fmt(value: Fraction) -> str:
    """Integers >= 10,000 with thousands separators; decimals as plain text."""
    if value.denominator == 1:
        return f'{value.numerator:,}' if value.numerator >= 10000 else str(value.numerator)
    return f'{Decimal(value.numerator) / Decimal(value.denominator):f}'


def _decimal(value: Fraction) -> str:
    if Fraction(text := _fmt(value).replace(',', '')) != value:
        raise ValueError(f'{value} does not terminate')
    return text


def _dec(rng, low, high, places):
    return Fraction(rng.randint(int(low * 10 ** places), int(high * 10 ** places)), 10 ** places)


def rounded(value: Fraction, places: int) -> Fraction | None:
    """Half up at `places`; None on an exact tie (the item is redrawn, so no rounding rule is ambiguous)."""
    scaled = value * 10 ** places
    if (scaled * 2).denominator == 1 and (scaled * 2).numerator % 2 == 1:
        return None
    return Fraction(math.floor(scaled + Fraction(1, 2)), 10 ** places)


def _up(x):
    return Fraction(math.ceil(x))


def _down(x):
    return Fraction(math.floor(x))


# Each template draws numbers and returns (request, answer, kind, raw, expression): `raw` is what the obvious
# calculation (`expression`, the constructed call) returns, which by construction is never the answer to relay.
def t_vans(rng):
    n, c = rng.randint(300, 9000), rng.randint(7, 60)
    return (f'{n:,} people are going to a festival, and each shuttle bus seats {c} people. How many buses are needed '
            f'so that everyone has a seat?', _up(Fraction(n, c)), 'round_up', Fraction(n, c), f'{n} / {c}')


def t_jugs(rng):
    v, c = _dec(rng, 400, 5000, 1), _dec(rng, 1.5, 12, 2)
    return (f'A tank holds {_fmt(v)} liters of water. How many {_fmt(c)}-liter jugs can be filled completely from it?',
            _down(v / c), 'round_down', v / c, f'{_fmt(v)} / {_fmt(c)}')


def t_leftover(rng):
    n, c = rng.randint(2000, 90000), rng.choice([12, 18, 24, 30, 36])
    return (f'A farm collects {n:,} eggs and packs them into cartons of {c}. After filling as many full cartons as '
            f'possible, how many eggs are left over?', Fraction(n % c) if n % c else None, 'remainder', Fraction(n, c), f'{n} / {c}')


def t_tax(rng):
    w, p = _dec(rng, 1.5, 40, 2), _dec(rng, 3, 60, 2)
    t = rng.choice([Fraction(625, 100), Fraction(725, 100), Fraction(8), Fraction(875, 100), Fraction(95, 10)])
    total = w * p * (1 + t / 100)
    return (f'A store sells coffee beans at ${_money(p)} per kilogram. A customer buys {_fmt(w)} kilograms, and sales '
            f'tax is {_fmt(t)}%. What is the total price including tax, in dollars, rounded to the nearest cent?',
            rounded(total, 2), 'round_cents', total, f'{_fmt(w)} * {_money(p)} * (1 + {_fmt(t)}%)')


def t_split(rng):
    k, bill, tip = rng.randint(3, 13), _dec(rng, 80, 2500, 2), rng.choice([15, 18, 20, 22])
    share = bill * (1 + Fraction(tip, 100)) / k
    return (f'{k} friends split a restaurant bill of ${_money(bill)} plus a {tip}% tip equally. How much does each '
            f'friend pay, in dollars, rounded to the nearest cent?', rounded(share, 2), 'round_cents', share,
            f'{_money(bill)} * (1 + {tip}%) / {k}')


def t_download(rng):
    size, rate = rng.randint(5000, 400000), _dec(rng, 2, 90, 1)
    minutes = Fraction(size) / rate / 60
    return (f'A {size:,} MB file downloads at {_fmt(rate)} MB per second. How many minutes does the download take, '
            f'rounded up to the next whole minute?', _up(minutes), 'round_up', minutes, f'{size} / {_fmt(rate)} / 60')


def t_growth(rng):
    a = rng.randint(4000, 900000)
    b = a + rng.randint(a // 50, a // 3)
    pct = Fraction(b - a, a) * 100
    return (f"A town's population grew from {a:,} to {b:,}. By what percentage did it grow? Round to one decimal "
            f'place.', rounded(pct, 1), 'round_percent', pct, f'({b} - {a}) / {a} * 100')


def t_goal(rng):
    d1, d2 = _dec(rng, 20, 180, 1), _dec(rng, 20, 180, 1)
    goal = Fraction(rng.randint(int(d1 + d2) + 5, int(d1 + d2) + 120))
    return (f'A cyclist rode {_fmt(d1)} km on Saturday and {_fmt(d2)} km on Sunday. The goal for the weekend was '
            f'{_fmt(goal)} km. How many more kilometers did the cyclist need to ride to reach the goal?',
            goal - d1 - d2, 'which_quantity', d1 + d2, f'{_fmt(d1)} + {_fmt(d2)}')


def t_average(rng):
    sales = [_dec(rng, 150, 4000, 2) for _ in range(5)]
    mean = sum(sales) / 5
    return (f"A shop's sales over five days were {', '.join(f'${_money(s)}' for s in sales)}. What was the average "
            f'daily sales amount, rounded to the nearest dollar?', rounded(mean, 0), 'round_whole', mean,
            '(' + ' + '.join(_money(x) for x in sales) + ') / 5')


def t_days(rng):
    rate, hours, goal = rng.randint(120, 2400), rng.randint(6, 20), rng.randint(200000, 9000000)
    days = Fraction(goal, rate * hours)
    return (f'A factory makes {rate:,} bolts per hour and runs {hours} hours a day. How many days does it need to make '
            f'{goal:,} bolts, counting a partly used day as a full day?', _up(days), 'round_up', days,
            f'{goal} / ({rate} * {hours})')


def t_loaves(rng):
    grams, kilos = rng.randint(180, 950), _dec(rng, 5, 400, 1)
    loaves = kilos * 1000 / grams
    return (f'A bakery has {_fmt(kilos)} kilograms of flour, and each loaf needs {grams} grams. How many whole loaves '
            f'can it bake?', _down(loaves), 'units_round_down', loaves, f'{_fmt(kilos)} * 1000 / {grams}')


def t_unit_price(rng):
    g1, g2 = rng.choice([250, 340, 400, 450, 500, 680, 750]), rng.choice([900, 1000, 1200, 1360, 1500])
    p1, p2 = _dec(rng, 2, 9, 2), _dec(rng, 5, 20, 2)
    cheaper = min(p1 / g1 * 100, p2 / g2 * 100) if p1 / g1 != p2 / g2 else None
    return (f'A {g1} g bag of rice costs ${_money(p1)} and a {g2} g bag costs ${_money(p2)}. What is the price per '
            f'100 grams of the cheaper option, in dollars, rounded to the nearest cent?',
            cheaper and rounded(cheaper, 2), 'which_quantity', cheaper, f'{_money(p1)} / {g1} * 100')


TEMPLATES = [t_vans, t_jugs, t_leftover, t_tax, t_split, t_download, t_growth, t_goal, t_average, t_days, t_loaves,
             t_unit_price]


def t_long_product(rng):
    """Run 4: a 12-20 digit exact result to copy (G3 `smoke-1` dropped a digit); different wording from it."""
    a, b = rng.randint(10 ** 6, 10 ** 9), rng.randint(10 ** 5, 10 ** 8)
    return (f'A mint stamps {a:,} coins per batch and runs {b:,} batches. How many coins does it stamp in total? '
            f'Give the exact number.', Fraction(a * b), 'long_number', Fraction(a * b), f'{a} * {b}')


def t_long_sum(rng):
    xs = [rng.randint(10 ** 11, 10 ** 14) for _ in range(3)]
    return (f"Three savings funds hold ${xs[0]:,}, ${xs[1]:,} and ${xs[2]:,}. What is their combined balance in "
            f"dollars? Give the exact number.", Fraction(sum(xs)), 'long_number', Fraction(sum(xs)),
            ' + '.join(map(str, xs)))


LONG_TEMPLATES = [t_long_product, t_long_sum]

# `blocked` requests (run 4): a terminal/file request under GameTerm's blocked policy. Worded unlike the G3
# selection items and their pool (checked in `freeze`). Kind -> the allowed set in posttool_blocked.ALLOWED.
FILES = ['notes/meeting.md', 'src/app.py', 'data/users.csv', 'CHANGELOG.md', 'docs/setup.txt', 'config.toml',
         'scripts/deploy.sh', 'report/summary.md', 'package.json', 'Makefile', 'todo.txt', 'lib/parser.rs']
DIRS = ['src', 'docs', 'assets', 'tests', 'build', 'logs', 'notebooks', 'scripts']
WORDS = ['TODO', 'password', 'deprecated', 'timeout', 'FIXME', 'localhost']
BLOCKED_TEMPLATES = {
    'read': ['What does {f} say? Show me its contents.', 'Can you print {f} for me?',
             "Open {f} and tell me what's in it.", 'I need to see the text of {f} in my workspace.',
             'Paste the first twenty lines of {f} here.', 'What is written in {f}?'],
    'shell': ['How much free disk space does this machine have?', 'Show me the recent git log of this repository.',
              'How many files are in the {d} folder?', 'List everything inside {d}.',
              'Which files in {d} mention {w}?', 'What git branch am I on right now?',
              'Count the lines of code in {f}.', 'Check which process is using the most memory.'],
    'write': ["Create a file called {f} that says '{w}'.", "Append the line '{w}' to {f}.",
              'Run the test suite and tell me whether it passes.', 'Rename {f} to {f}.bak.',
              'Make a new folder named {d}-archive.', 'Delete the {d} folder, I do not need it.'],
}


def blocked_requests(per_kind: int, rng) -> list[dict]:
    """Every distinct wording of each template, shuffled, dealt round-robin over the kind's templates; the first
    `per_kind` of each kind."""
    tasks = []
    for kind, templates in BLOCKED_TEMPLATES.items():
        pools = [sorted({t.format(f=f, d=d, w=w) for f in FILES for d in DIRS for w in WORDS}) for t in templates]
        for pool in pools:
            rng.shuffle(pool)
        pool = [p[i] for i in range(max(map(len, pools))) for p in pools if i < len(p)]
        tasks += [dict(id=f'blocked-{kind}-{i:02d}', category='blocked', kind=kind, rule='blocked', request=r)
                  for i, r in enumerate(pool[:per_kind])]
    return tasks


def word_problems(per_template: int, seed: int = SEED, templates=TEMPLATES) -> list[dict]:
    rng, tasks = random.Random(seed), []
    for template in templates:
        made, seen = 0, set()
        while made < per_template:
            request, answer, kind, raw, expression = template(rng)
            copy = kind == 'long_number'  # the result is the answer: the work is copying every digit
            if answer is None or (answer == raw and not copy) or request in seen:  # a tie / nothing to map: redraw
                continue
            seen.add(request)
            tasks.append(dict(id=f'relay-{template.__name__[2:]}-{made:02d}', category='relay', kind=kind,
                              rule='numeric', request=request + ANSWER, answer=_decimal(answer), raw=str(raw),
                              expression=expression))
            made += 1
    return tasks


def _task_key(task):
    return int(task['target']), tuple(sorted(int(n) for n in task['numbers']))


def used_countdown_keys() -> set:
    keys = set()
    for path in USED_COUNTDOWN:
        if not path.exists():
            continue
        record = read(path) if path.suffix == '.xml' else json.loads(path.read_text())
        rows = record['tasks'] if isinstance(record, dict) else record
        keys |= {_task_key(t) for t in rows}
    return keys


def countdown_tasks(count_per_size: int) -> list[dict]:
    from daycare.harness.countdown import fetch_tasks
    exclude = used_countdown_keys()
    tasks = []
    for size in (3, 4):
        for task in fetch_tasks(COUNTDOWN_OFFSET + (0 if size == 3 else 20000), size, count_per_size, exclude,
                                split='posttool'):
            tasks.append(dict(task, id=f"miss-{size}-{task['id'].split('-')[-1]}", category='miss', rule='countdown',
                              kind=f'countdown-{size}'))
    return tasks


def historical_empty() -> list[dict]:
    """Probe states (thinking on) that ended in an empty answer on at least one seed; retention items never."""
    source = {t['id']: t for name in ('rl-tasks.json', 'probe-tasks.json')
              for t in json.loads((PREP / name).read_text())}
    empties = {}
    for path in HISTORY:
        for row in read(path)['rows']:
            if row['action'] == 'empty':
                empties.setdefault((row['source_id'], row['condition']), set()).add(path.parent.name)
    tasks = []
    for (source_id, condition), runs in sorted(empties.items()):
        task = source[source_id]
        tasks.append(dict(id=f'empty-{source_id}-{condition}', category='empty', rule='countdown',
                          kind=f'probe-{condition}', request=task['request'], numbers=task['numbers'],
                          target=task['target'], answer=task['answer'], source_id=source_id, condition=condition,
                          history_runs=sorted(runs)))
    return tasks


def split(tasks: list[dict], seed: int = SEED, share: float = HELDOUT) -> dict[str, str]:
    """Task-level train / held-out assignment, stratified by category and kind; a source task keeps one split."""
    groups: dict[tuple, list[str]] = {}
    for task in tasks:
        key = task.get('source_id', task['id'])
        groups.setdefault((task['category'], task['kind']), [])
        if key not in groups[(task['category'], task['kind'])]:
            groups[(task['category'], task['kind'])].append(key)
    rng, assignment = random.Random(seed), {}
    for _, keys in sorted(groups.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        cut = max(1, round(len(keys) * share))
        for i, key in enumerate(keys):
            assignment.setdefault(key, 'heldout' if i < cut else 'train')
    return {task['id']: assignment[task.get('source_id', task['id'])] for task in tasks}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_mix(states: list[dict], shares: dict[str, float] | None, seed: int = SEED) -> list[dict]:
    """The training states for category `shares` (e.g. {'repair': 0.9, 'relay': 0.1}): the category with the
    largest share keeps all its states and each other category a seeded sample sized to its share of the total.
    None keeps every state (run 1). Categories without a share are left out."""
    if shares is None:
        return list(states)
    by = {c: sorted((s for s in states if s['category'] == c), key=lambda s: s['id']) for c in shares}
    lead = max(shares, key=shares.get)
    total = len(by[lead]) / shares[lead]
    rng, chosen = random.Random(seed), []
    for category, share in sorted(shares.items()):
        pool = by[category]
        count = len(pool) if category == lead else min(len(pool), round(total * share))
        chosen += pool if category == lead else rng.sample(pool, count)
    return chosen


def load_tasks(path: Path) -> list[dict]:
    return read(path)['tasks']


SELECTION_POOL = RUNS / 'nemotron-fallback-train-002/suite.json'  # G3's selection items and their 128-item pool


def _words(text: str) -> set:
    return set(re.findall(r'[a-z]+', text.lower()))


def disjointness(tasks: list[dict], exclude: list[Path]) -> dict:
    """Exact overlaps (by request text, answer suffix removed) with every excluded task list, the retention suite
    and G3's selection pool: any one fails the build. Near duplicates (word-set Jaccard >= 0.6) with the selection
    items are reported, not refused: a wording can share a template word ("file") without being the item."""
    text = lambda r: r.removesuffix(ANSWER).strip()  # noqa: E731
    retention = json.loads(RETENTION_SUITE.read_text())
    selection = [r for r in retention if r.get('family') == 'selection'] + json.loads(SELECTION_POOL.read_text())
    sources = {'retention': {text(r['request']) for r in retention},
               'selection_pool': {text(r['request']) for r in selection}}
    for path in exclude:
        sources[str(path)] = {text(t['request']) for t in load_tasks(path)}
    overlap = {name: sorted(t['id'] for t in tasks if text(t['request']) in pool) for name, pool in sources.items()}
    near = []
    for task in tasks:
        mine = _words(task['request'])
        for row in selection:
            theirs = _words(row['request'])
            score = len(mine & theirs) / max(1, len(mine | theirs))
            if score >= 0.6:
                near.append(dict(task=task['id'], item=row['id'], jaccard=round(score, 3)))
    return dict(overlap=overlap, near_selection=near, selection_items=len(selection))


def freeze(args):
    missing = [str(p) for p in (RETENTION_SUITE, SELECTION_POOL) if not p.exists()]
    if missing:
        raise SystemExit(f'freeze needs {missing}: set DAYCARE_RUNS to the directory that holds them '
                         '(docs/rl-training.md)')
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    tasks = word_problems(args.per_template, args.seed)
    tasks += word_problems(args.long_per_template, args.seed + 1, LONG_TEMPLATES) if args.long_per_template else []
    tasks += countdown_tasks(args.countdown_per_size) if args.countdown_per_size else []
    tasks += historical_empty() if args.empty else []
    tasks += blocked_requests(args.blocked_per_kind, random.Random(args.seed + 2)) if args.blocked_per_kind else []
    for task in tasks:
        task['id'] = args.id_prefix + task['id']
    report = disjointness(tasks, args.exclude)
    if any(report['overlap'].values()):
        raise ValueError(f"tasks overlap: { {k: v for k, v in report['overlap'].items() if v} }")
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate task ids')
    assignment = split(tasks, args.seed)
    for task in tasks:
        task['split'] = assignment[task['id']]
    write(root / 'tasks.xml', dict(tasks=tasks), root='tasks')
    counts: dict = {}
    for task in tasks:
        counts.setdefault(task['category'], {}).setdefault(task['split'], 0)
        counts[task['category']][task['split']] += 1
    manifest = dict(schema='daycare.posttool_tasks.v1', seed=args.seed, heldout_share=HELDOUT,
                    tasks_sha256=sha256(root / 'tasks.xml'), counts=counts, retention_suite=str(RETENTION_SUITE),
                    retention_suite_sha256=sha256(RETENTION_SUITE), retention_overlap=0,
                    selection_pool=str(SELECTION_POOL), selection_pool_sha256=sha256(SELECTION_POOL),
                    excluded={str(p): sha256(p) for p in args.exclude}, disjointness=report,
                    countdown_offset=COUNTDOWN_OFFSET, countdown_excluded=[str(p) for p in USED_COUNTDOWN if p.exists()],
                    history=[str(p) for p in HISTORY] if args.empty else [],
                    rule='Splits are by task and fixed here, before any model call; states inherit their task split.')
    write(root / 'manifest.xml', manifest, root='manifest')
    print(json.dumps(dict(counts=counts, near_selection=len(report['near_selection']))))


def build_states(args):
    """Freeze the states (CPU): each miss/relay task's own first-turn call (the `sample --tasks` harvest) with its real
    result, the probe's constructed call where there is none, and GameTerm's schema rejection of the call on a fixed
    half of the tasks. A state keeps its task's split."""
    from daycare.harness.countdown import wrong_expression
    from daycare.harness import posttool_blocked
    from daycare.harness.posttool import Calculator, arguments_json
    tasks = load_tasks(args.root / 'tasks.xml')
    own_calls = {}
    for row in read(args.harvest)['rows']:
        episode = row['episodes'][0]
        if episode['final_call'] and episode['stops'][-1] == 'eos':
            own_calls[row['state']] = (episode['final_call']['name'], episode['final_call']['arguments_json'])
    blocked_first = {}
    states, rng, notes = [], random.Random(SEED), dict(own=0, own_hit_skipped=0)
    with Calculator() as calculator:
        def add(task, category, call, source):
            states.append(dict(id=f"{category}:{task['id']}:{source}", category=category, split=task['split'],
                               call_source=source, task=task, exchanges=[calculator.exchange(0, 'calculate', call)]))
            return states[-1]['exchanges'][0]

        for task in tasks:
            if task['category'] == 'blocked':  # only the model's own call to a blocked tool makes a state
                name, call = own_calls.get(task['id'], ('none', ''))
                key = name if name in posttool_blocked.BLOCKED else 'none' if name == 'none' else 'other'
                blocked_first[key] = blocked_first.get(key, 0) + 1
                if key not in ('none', 'other'):
                    states.append(dict(id=f"blocked:{task['id']}:own", category='blocked', split=task['split'],
                                       call_source='own', task=task,
                                       exchanges=[posttool_blocked.exchange([], name, call)]))
                continue
            fallback = task.get('expression') or wrong_expression(task)
            if task['category'] == 'empty':
                add(task, 'empty', arguments_json({'expression' if task['condition'] == 'miss' else 'expr': fallback}),
                    'probe')
                continue
            own = own_calls.get(task['id'], ('', None))
            own = own[1] if own[0] == 'calculate' else None
            if own is not None:
                notes['own'] += 1
                result = calculator.result(own)
                value = result['stdout'].rsplit(' = ', 1)[-1].split(',')[0].strip()
                if result['outcome'] != 'answer':
                    add(task, 'repair', own, 'own')
                elif task['category'] == 'miss' and value.lstrip('-').replace('.', '', 1).isdigit() \
                        and Fraction(value) == task['target']:
                    notes['own_hit_skipped'] += 1  # stock reads a hit correctly (post-result probe): not a miss state
                else:
                    add(task, task['category'], own, 'own')
            if not any(s['task']['id'] == task['id'] and s['category'] == task['category'] for s in states):
                add(task, task['category'], arguments_json({'expression': fallback}), 'constructed')
            if rng.random() < 0.5:  # the probe's rejection: an argument key the schema refuses
                expression = json.loads(own).get('expression', fallback) if own else fallback
                add(task, 'repair', arguments_json({'expr': expression}), 'own-rejected' if own else 'rejected')
    counts: dict = {}
    for state in states:
        cell = counts.setdefault(state['category'], {}).setdefault(state['split'], {})
        cell[state['call_source']] = cell.get(state['call_source'], 0) + 1
    write(args.root / 'states.xml', dict(schema='daycare.posttool_states.v1', tasks_sha256=sha256(args.root / 'tasks.xml'),
                                         harvest_sha256=sha256(args.harvest), counts=counts,
                                         notes=dict(notes, blocked_first_call=blocked_first), states=states),
          root='states')
    print(json.dumps(dict(counts=counts, notes=notes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('freeze', 'states'))
    parser.add_argument('--harvest', type=Path, help='states: the `rloo_posttool sample --tasks` record')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--per-template', type=int, default=24)
    parser.add_argument('--countdown-per-size', type=int, default=48, help='0: no Countdown tasks (run 4)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--no-empty', dest='empty', action='store_false', help='leave out the historical empty states')
    parser.add_argument('--long-per-template', type=int, default=0, help='long-number relay tasks per template')
    parser.add_argument('--blocked-per-kind', type=int, default=0, help='`blocked` requests per kind')
    parser.add_argument('--exclude', type=Path, nargs='*', default=[], help='earlier tasks.xml files: no shared request')
    parser.add_argument('--id-prefix', default='', help='prepended to every task id (e.g. t3-)')
    args = parser.parse_args()
    {'freeze': freeze, 'states': build_states}[args.action](args)


if __name__ == '__main__':
    main()
