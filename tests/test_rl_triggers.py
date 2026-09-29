"""Rolling stop triggers (daycare/nursery/rl_triggers.py): synthetic rows, and the replay on post-tool runs 1 and 2."""
import os
from pathlib import Path

import pytest

from daycare.nursery import trainer_env
from daycare.nursery.rl_triggers import LEGACY, RUN3, RUN3_REWARD, Triggers, finished_length, replay, replayed_push

RUNS = Path(os.environ.get('DAYCARE_RUNS') or '$DAYCARE_RUNS')  # the unpublished run records, for the replays


def steady(**change):
    return dict(dict(kl=.005, entropy=.55, finished_length=600., capped_turn_rate=.03, mean_reward=.85,
                     advantage_length_sum=-100., skipped=False), **change)


@pytest.mark.parametrize('change,reason', [(dict(kl=.05), 'kl'), (dict(entropy=.4), 'entropy'),
                                           (dict(finished_length=1300.), 'finished_length'),
                                           (dict(capped_turn_rate=.3), 'capped_turn_rate'),
                                           (dict(mean_reward=.5), 'mean_reward')])
def test_each_limit_trips_on_the_rolling_window_after_the_baseline(change, reason):
    triggers = Triggers(dict(window=5, capped_ratio=float('inf')))  # the absolute limits; the ratio has its own test
    assert not any(triggers.check(steady()) for _ in range(12))
    tripped = [triggers.check(steady(**change)) for _ in range(5)]
    assert tripped[:2] == [None, None] and tripped[-1].startswith(reason)  # the 5-update mean crosses at 3-5


def test_capped_rate_relative_to_baseline_has_a_floor():
    # 2x a near-zero baseline would trip on one capped episode (runs 1-2 replay); the floor is 5%
    triggers = Triggers(dict(window=5))
    assert not any(triggers.check(steady(capped_turn_rate=.01)) for _ in range(10))
    assert not any(triggers.check(steady(capped_turn_rate=.04)) for _ in range(5))  # 4x baseline, under the floor
    triggers = Triggers(dict(window=5))
    assert not any(triggers.check(steady(capped_turn_rate=.04)) for _ in range(10))
    assert triggers.check(steady(capped_turn_rate=.10)) is None
    assert [triggers.check(steady(capped_turn_rate=.10)) for _ in range(4)][-1].startswith('capped_turn_rate')


def test_length_push_triggers():
    # the audit's streak: positive for 10 stepped updates in a row; a negative update resets it
    triggers = Triggers(dict(push_window=0))
    assert not any(triggers.check(steady(advantage_length_sum=5.)) for _ in range(9))
    assert triggers.check(steady()) is None
    assert not any(triggers.check(steady(advantage_length_sum=5.)) for _ in range(9))
    assert 'consecutive' in triggers.check(steady(advantage_length_sum=5.))
    # the window sum: net positive over the last 10, checked once the window no longer overlaps the baseline
    triggers = Triggers(dict(push_positive=0))
    rows = [steady()] * 10 + [steady(advantage_length_sum=300.), steady(advantage_length_sum=-100.)] * 5
    got = [triggers.check(r) for r in rows]
    assert got[:19] == [None] * 19 and 'summed over the last 10' in got[19]
    # a missing push (runs 1-2 never logged it) never trips
    triggers = Triggers()
    assert not any(triggers.check(steady(advantage_length_sum=float('nan'))) for _ in range(30))


@pytest.mark.skipif(not trainer_env.trainer_available(),
                    reason='trainer tinygrad not installed')
def test_replayed_push_scores_the_run3_reward():
    group = [dict(state='repair:a', reward=1.0, reason='correct', final='<answer>3</answer>', final_call=False,
                  tokens=[100]),
             dict(state='repair:a', reward=0.0, reason='incomplete', final='', final_call=False, tokens=[4096]),
             dict(state='repair:a', reward=0.0, reason='wrong', final='<answer>4</answer>', final_call=False, tokens=[300])]
    plain = dict(RUN3_REWARD, length_weight=0)
    # rewards 1, -1.5, -1 -> leave-one-out advantages 2.25, -1.5, -0.75
    assert replayed_push([group], plain) == pytest.approx(2.25 * 100 - 1.5 * 4096 - 0.75 * 300)
    assert replayed_push([group], plain, ('relay',)) == 0.0
    assert replayed_push([group], RUN3_REWARD) < replayed_push([group], plain)  # the length term brakes harder


def test_consecutive_skips_trip_and_a_step_resets_the_count():
    triggers = Triggers(dict(skips_max=3))
    assert triggers.check(steady(skipped=True)) is None and triggers.check(steady(skipped=True)) is None
    assert triggers.check(steady()) is None and triggers.check(steady(skipped=True)) is None
    assert triggers.check(steady(skipped=True)) is None
    assert 'consecutive skipped' in triggers.check(steady(skipped=True))


def test_finished_length_ignores_capped_episodes():
    episodes = [[dict(tokens=[100, 50], stops=['eos', 'eos']), dict(tokens=[4096], stops=['limit'])],
                [dict(tokens=[250], stops=['eos'])]]
    assert finished_length(episodes) == 200.0 and finished_length([]) != finished_length([])


@pytest.mark.skipif(not (RUNS / 'posttool-rloo-r2-001/update-053').exists(), reason='run records not on this host')
def test_defaults_replayed_on_runs_one_and_two():
    """Calibration (research/rloo-posttool-audit-20260927.md, rloo-posttool-calculator-r3.md): run 2 stops well
    before its u45 collapse, run 1 not before its end-of-run drift."""
    assert replay(RUNS / 'posttool-rloo-r2-001', LEGACY)[0] == 33  # runs 1-2 settings
    assert replay(RUNS / 'posttool-rloo-r1-001', LEGACY)[0] == 96
    mix = ('repair', 'relay')  # run-3 defaults, reward and mix: run 2 on the capped rate, run 1 on KL at its end
    two, one = replay(RUNS / 'posttool-rloo-r2-001', None, RUN3_REWARD, mix), replay(RUNS / 'posttool-rloo-r1-001', None, RUN3_REWARD, mix)
    assert two[0] == 32 and two[1].startswith('capped_turn_rate') and one[0] == 98 and one[1].startswith('kl')
    quiet = dict(kl_max=9, entropy_ratio=0)  # a working beta may keep KL and entropy quiet: the length triggers still stop run 2
    assert replay(RUNS / 'posttool-rloo-r2-001', quiet, RUN3_REWARD, mix)[0] == 32
    assert replay(RUNS / 'posttool-rloo-r1-001', quiet, RUN3_REWARD, mix) is None
    # run 2 as trained (filter): the window-sum push trigger fires at the first check; run 1 as trained never
    assert replay(RUNS / 'posttool-rloo-r2-001', dict(quiet, capped_max=9, capped_floor=9, length_growth=9), 'as_run')[0] == 20
    assert replay(RUNS / 'posttool-rloo-r1-001', dict(quiet, capped_max=9, capped_floor=9, capped_ratio=99, length_growth=9), 'as_run') is None


def mixed(entropy_by_category: dict, shares: dict, **change):
    """A row whose batch entropy is the token-share mix of per-category levels."""
    return steady(entropy=sum(shares[c] * entropy_by_category[c] for c in shares), shares=shares, **change)


def test_entropy_compares_within_composition():
    """Run 4's stop: a high-entropy category (blocked) in the baseline batches but not in the window's. The raw test
    trips on composition alone; the composition test does not, and still trips on a within-category drop."""
    levels = dict(repair=0.56, blocked=1.5)
    with_blocked, without = dict(repair=0.75, blocked=0.25), dict(repair=1.0)
    rows = [mixed(levels, with_blocked if i % 2 else without) for i in range(10)] + [mixed(levels, without)] * 10
    raw, adjusted = Triggers(dict(entropy_composition=0)), Triggers()
    assert any(r and r.startswith('entropy') for r in (raw.check(x) for x in rows))
    assert not any(adjusted.check(x) for x in rows)
    dropped = Triggers()
    got = [dropped.check(x) for x in rows[:10] + [mixed(dict(repair=0.45, blocked=1.5), without)] * 10]
    assert got[-1] is not None and got[-1].startswith('entropy')
    # rows without shares (older records) fall back to the raw comparison
    assert Triggers().check(steady()) is None


@pytest.mark.skipif(not (RUNS / 'posttool-rloo-r4-001/update-097').exists(), reason='run records not on this host')
def test_composition_entropy_replayed_on_runs_three_and_four():
    """Run 4 stopped at u98 on raw entropy (composition-driven); within composition neither run 3 nor run 4 trips,
    while run 2 still stops early (u32, capped rate; entropy alone at u38)."""
    assert replay(RUNS / 'posttool-rloo-r4-001', RUN3)[0] == 98
    assert replay(RUNS / 'posttool-rloo-r4-001', None) is None
    assert replay(RUNS / 'posttool-rloo-r3-001', None) is None
    mix = ('repair', 'relay')
    assert replay(RUNS / 'posttool-rloo-r2-001', None, RUN3_REWARD, mix)[0] == 32
    only_entropy = dict(kl_max=9, length_growth=9, capped_max=9, capped_floor=9, capped_ratio=99, reward_drop=9,
                        push_positive=0, push_window=0)
    two = replay(RUNS / 'posttool-rloo-r2-001', only_entropy, RUN3_REWARD, mix)
    assert two[0] == 38 and two[1].startswith('entropy')
