"""The TUI seam (daycare/harness/runs.py): the fixture run folders in tui/testdata/fixture are the contract, pinned as
JSON under tui/testdata/expected; the Go side pins the same files (tui/internal/seam). Writers are tested on a
temporary copy, never on the fixture."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('numpy')
pytest.importorskip('daycare.artifact.record_xml')  # the RL record modules; the seam reads nothing without them
pytest.importorskip('daycare.nursery.rl_triggers')

from daycare.harness import runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path('tui/testdata/fixture')  # relative: the goldens hold these relative paths
EXPECTED = REPO / 'tui' / 'testdata' / 'expected'


def seam(*args, env=None):
    result = subprocess.run([sys.executable, '-m', 'daycare.harness.runs', *args], cwd=REPO, capture_output=True,
                            text=True, env=dict(os.environ, **(env or {})))
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize('name,args', [('list', ('list', '--root', str(FIXTURE))),
                                       ('show-001', ('show', '--root', str(FIXTURE / 'posttool-fixture-001'))),
                                       ('show-002', ('show', '--root', str(FIXTURE / 'posttool-fixture-002'))),
                                       ('show-003', ('show', '--root', str(FIXTURE / 'posttool-fixture-003')))])
def test_fixture_matches_the_pinned_contract(name, args):
    code, got = seam(*args)
    assert code == 0
    assert got == json.loads((EXPECTED / f'{name}.json').read_text())


def test_list_skips_sample_arms_and_orders_runs():
    _, got = seam('list', '--root', str(FIXTURE))
    assert [r['id'] for r in got['runs']] == ['posttool-fixture-001', 'posttool-fixture-002', 'posttool-fixture-003']
    assert [r['state'] for r in got['runs']] == ['stopped', 'predeclared', 'in_progress']
    assert got['runs'][0]['stopped']['update'] == 6 and got['runs'][0]['verdict'] == 'fail'


def test_show_window_rules_with_the_real_triggers():
    _, got = seam('show', '--root', str(FIXTURE / 'posttool-fixture-001'))
    window = got['window']
    assert window['source'] == 'update records' and window['tripped']['update'] == 6
    assert window['tripped']['reason'] == got['stopped']['reason']  # the loop's own ruling, replayed
    crossed = [r['metric'] for r in window['readings'] if r['crossed']]
    assert crossed == ['entropy']
    assert got['verify']['bit_exact'] is True and got['updates'][-1]['step'] == 6


def test_show_reads_the_run_record_when_no_update_folders_exist():
    _, got = seam('show', '--root', str(FIXTURE / 'posttool-fixture-003'))
    assert got['state'] == 'in_progress' and got['window']['source'] == 'run record'
    assert got['window']['ready'] is False and got['window']['tripped'] is None


def test_unknown_run_is_an_error_object_and_exit_1():
    code, got = seam('show', '--root', str(FIXTURE / 'missing'))
    assert code == 1 and got['kind'] == 'error' and 'missing' in got['error']


@pytest.mark.parametrize('rule,value,diff,lo,hi,passed', [
    ('lower_gt', 0, 15.5, 11.2, 20.0, True), ('lower_gt', 0, 1.0, -0.5, 2.5, False),
    ('upper_ge', 0, 0.8, -0.6, 2.1, True), ('upper_lt', 0, -3.1, -4.1, -2.1, True), ('upper_lt', 0, -1, -2, 0, False),
    ('upper_le', 2.5, -2.9, -4.5, -1.3, True), ('lower_ge', -8, -2.7, -10.1, 4.8, False),
    ('count_le', 0, 0, None, None, True), ('count_le', 0, 1, None, None, False)])
def test_rules(rule, value, diff, lo, hi, passed):
    assert runs.rule_passes(rule, value, diff, lo, hi) is passed


def test_ci_rule_needs_both_bounds():
    with pytest.raises(ValueError):
        runs.rule_passes('lower_gt', 0, 1.0, None, None)


def test_predeclare_score_adopt_round_trip(tmp_path):
    root = tmp_path / 'posttool-new-001'
    code, got = seam('predeclare', '--root', str(root), '--protocol', 'rloo-posttool-new.md', '--template', 'run5',
                     '--states', 'S/states.xml', '--envelope', 'E.xml')
    assert code == 0 and got['gates'] == 9 and got['recipe'][-6:] == ['--states', 'S/states.xml', '--envelope',
                                                                      'E.xml', '--protocol', 'rloo-posttool-new.md']
    assert (root / 'predeclaration.xml').exists()
    code, got = seam('predeclare', '--root', str(root), '--protocol', 'x', '--template', 'run5')
    assert code == 1 and 'already predeclared' in got['error']
    _, got = seam('show', '--root', str(root))
    assert got['state'] == 'predeclared' and got['verdict'] == 'open'

    code, got = seam('score', '--root', str(root), '--gate', 'G1', '--diff', '15.5', '--lo', '11.2', '--hi', '20.0')
    assert code == 0 and got['passed'] is True and got['verdict'] == 'open'
    code, got = seam('score', '--root', str(root), '--gate', 'G1', '--diff', '1', '--lo', '0', '--hi', '2')
    assert code == 1 and 'already scored' in got['error']
    code, got = seam('score', '--root', str(root), '--gate', 'G3', '--diff', '1')
    assert code == 0 and got['passed'] is False and got['verdict'] == 'fail'
    code, got = seam('score', '--root', str(root), '--gate', 'G2a', '--diff', '-3')
    assert code == 1 and 'needs --lo and --hi' in got['error']

    code, got = seam('adopt', '--root', str(root), '--by', 'owner', '--exception', 'G3 fragile')
    assert code == 1 and 'every gate must be scored' in got['error']
    for gate in ('G1b', 'G2a', 'G2b', 'G5L', 'G5B', 'G6', 'G7'):
        seam('score', '--root', str(root), '--gate', gate, '--diff', '1', '--lo', '0.5', '--hi', '1.5')
    code, got = seam('adopt', '--root', str(root), '--by', 'owner', '--exception', 'G3 fragile; known costs listed')
    assert code == 0 and got['verdict'] == 'adopted (exception)'
    _, got = seam('show', '--root', str(root))
    assert got['adoption']['by'] == 'owner' and got['verdict'] == 'adopted (exception)'


def test_predeclare_with_own_gates_and_recipe(tmp_path):
    root = tmp_path / 'r'
    code, got = seam('predeclare', '--root', str(root), '--protocol', 'p.md',
                     '--gate', 'G1|repair|lower_gt|0|+10', '--recipe', '--updates', '3')
    assert code == 0 and got['recipe'] == ['--updates', '3', '--protocol', 'p.md']
    code, got = seam('predeclare', '--root', str(tmp_path / 'bad'), '--protocol', 'p.md', '--gate', 'G1|x|nope|0|y')
    assert code == 1 and 'unknown rule' in got['error']
    code, got = seam('predeclare', '--root', str(tmp_path / 'bare'), '--protocol', 'p.md')
    assert code == 1 and 'needs gates' in got['error']


def test_predeclare_refuses_a_started_run(tmp_path):
    root = tmp_path / 'started'
    shutil.copytree(REPO / FIXTURE / 'posttool-fixture-003', root)
    code, got = seam('predeclare', '--root', str(root), '--protocol', 'p.md', '--template', 'run5')
    assert code == 1 and 'already holds a run' in got['error']


def test_setup_checks_files_not_words(tmp_path):
    runner = tmp_path / 'runner'
    runner.write_text('#!/bin/sh\n')
    runner.chmod(0o755)
    (tmp_path / 'tg' / 'extra').mkdir(parents=True)
    env = dict(DAYCARE_RUNS=str(FIXTURE), DAYCARE_TRAIN_TINYGRAD_PATH=str(tmp_path / 'tg'),
               DAYCARE_BASE_GGUF=str(runner), DAYCARE_CALCULATE_RUNNER=str(runner), DAYCARE_ENVELOPE='/nope')
    code, got = seam('setup', env=env)
    assert code == 0 and got['kind'] == 'setup'
    by = {c['id']: c for c in got['checks']}
    assert by['trainer']['ok'] and by['model']['ok'] and by['runner']['ok'] and by['runs_root']['ok']
    assert not by['envelope']['ok'] and 'cannot be generated' in by['envelope']['fix']
    assert not by['tasks']['ok'] and by['tasks']['generate'] == 'tasks'
    assert by['predeclaration']['ok'] and by['predeclaration']['detail'] == 'posttool-fixture-002'
    assert got['ready'] is False
