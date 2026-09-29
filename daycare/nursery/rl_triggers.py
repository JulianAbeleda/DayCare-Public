"""Rolling stop triggers for an RL run (research/rloo-posttool-audit-20260927.md, Recommendations).

Each update gives a row: KL and entropy (token means over the trained tokens), the finished-episode mean length
(episodes whose last turn did not hit the token cap), the capped-turn rate, the outcome reward, the sequence term's
length push (`advantage_length_sum`) and whether the update was skipped. Once the baseline (the first `base` stepped
updates) exists, the last `window` stepped updates' means are checked every update; the first limit crossed names
the stop reason.

Defaults (run 3, research/rloo-posttool-calculator-r3.md), window 10: KL > 0.025, entropy < 0.85x, finished length
> 1.5x, capped-turn rate > 15% or > max(2x baseline, 5%), outcome reward < baseline - 0.2, length push summed over
the window > 0 (checked once the window clears the baseline) or positive 10 updates in a row, 3 skips in a row.
Replayed on runs 1-2 under the run-3 reward and mix (`replay(..., RUN3_REWARD, ('repair', 'relay'))`): run 2 trips at
u32 (capped rate; collapse u45), run 1 at u98 (KL); with KL and entropy silenced, run 2 still trips at u32 and run 1
never. The audit's 1.3x length and plain 2x capped limits trip calm run 1 at u25 and u11-17 (noise on a 0-3%
baseline), hence 1.5x and the 5% floor. `LEGACY` is the runs 1-2 setting (u96 / u33, both on KL).
Entropy within composition (research/posttool-selection-transfer.md): run 4 stopped at u98 on the raw entropy test
(`RUN3`); compared on each window's own category shares it does not trip (ridge 0.1-1; ridge 3 under-adjusts and
trips), run 3 never trips, run 2 still stops at u32 (capped rate; entropy alone u38) and run 1 at u98 (KL).

    python -m daycare.nursery.rl_triggers RUN [--limits kl_max=0.025,...]     replay a run's update records
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULTS = dict(window=10, base=10, kl_max=0.025, entropy_ratio=0.85, length_growth=0.5, capped_max=0.15,
                capped_ratio=2.0, capped_floor=0.05, reward_drop=0.2, skips_max=3, push_positive=10, push_window=1,
                entropy_composition=1, entropy_ridge=0.3)
# Entropy is compared within batch composition (research/posttool-selection-transfer.md, P4): run 4 stopped at u98 on
# a raw token-mean entropy that a `blocked` group lifts by ~0.18 per batch. With `entropy_composition`, the baseline's
# per-category entropy levels are fitted on its rows (batch entropy = sum of category token share x level, ridge
# `entropy_ridge` toward the baseline mean) and the window is compared with those levels on its own shares.
# Run 3 adds the audit's length triggers because a working KL beta may keep the KL trigger (calibrated on runs 1-2,
# whose KL term was inert) quiet. Runs 1-2 (window 5, length 2x, capped 0.2, no ratio or push test) are LEGACY.
LEGACY = dict(DEFAULTS, window=5, length_growth=1.0, capped_max=0.2, capped_ratio=float('inf'), capped_floor=0.0,
              push_positive=0, push_window=0, entropy_composition=0)
RUN3 = dict(DEFAULTS, entropy_composition=0)  # the raw entropy test runs 3-4 ran with


def finished_length(episodes) -> float:
    """Mean sampled tokens per episode (all turns) over episodes whose last turn ended before the cap."""
    done = [sum(e['tokens']) for g in episodes or [] for e in g if e['stops'][-1] != 'limit']
    return float(np.mean(done)) if done else float('nan')


def shares(groups) -> dict:
    """Token share of each state category (the id's prefix, `blocked:t3-...` -> blocked) in the trained batch."""
    tokens: dict = {}
    for g in groups or []:
        category = str(g.get('task', '')).split(':')[0]
        tokens[category] = tokens.get(category, 0) + sum(g['tokens'])
    total = sum(tokens.values())
    return {c: n / total for c, n in tokens.items()} if total else {}


def row(result: dict) -> dict:
    m = result['metrics']
    return dict(kl=m['kl'], entropy=m['entropy'], capped_turn_rate=m.get('capped_turn_rate', float('nan')),
                mean_reward=result['mean_reward'], finished_length=finished_length(result.get('episodes')),
                advantage_length_sum=m.get('advantage_length_sum', float('nan')), skipped=bool(m.get('skipped')),
                shares=shares(result.get('groups')))


def category_levels(rows, ridge: float) -> dict:
    """Per-category entropy levels h minimizing sum (H - shares . h)^2 + ridge |h - mean H|^2 over `rows`."""
    cats = sorted({c for r in rows for c in r['shares']})
    shares_ = np.array([[r['shares'].get(c, 0.0) for c in cats] for r in rows])
    entropy = np.array([r['entropy'] for r in rows])
    mean = float(entropy.mean())
    h = np.linalg.solve(shares_.T @ shares_ + ridge * np.eye(len(cats)), shares_.T @ entropy + ridge * mean)
    return dict(zip(cats, h.tolist()), _mean=mean)


def expected_entropy(rows, levels: dict) -> float:
    """The baseline levels on these rows' own composition (a category the baseline never saw gets its mean)."""
    return float(np.mean([sum(s * levels.get(c, levels['_mean']) for c, s in r['shares'].items()) for r in rows]))


def mix_row(episodes, categories) -> dict:
    """capped_turn_rate, finished_length and mean (outcome) reward over the episodes of `categories` only."""
    kept = [g for g in episodes or [] if g[0]['state'].split(':')[0] in categories]
    stops = [s for g in kept for e in g for s in e['stops']]
    rewards = [e['reward'] for g in kept for e in g]
    return dict(capped_turn_rate=float(np.mean([s == 'limit' for s in stops])) if stops else float('nan'),
                finished_length=finished_length(kept),
                mean_reward=float(np.mean(rewards)) if rewards else float('nan'))


def replayed_push(episodes, policy: dict, categories=None) -> float:
    """sum(advantage x tokens) that `policy` (the graded reward, with its length term) would have given a saved
    update's episode summaries (reward, reason, final, final_call, tokens, state): the run-3 advantage_length_sum,
    replayed on runs whose own reward differed. `categories`: only those groups train (the run-3 mix)."""
    from daycare.harness.posttool import BLANKS, group_length_term, is_giveup
    from .rlvr import leave_one_out
    total = 0.0
    for group in episodes or []:
        category = group[0]['state'].split(':')[0]
        if categories and category not in categories or len(group) < 2:
            continue
        correct = [e['reward'] == 1.0 for e in group]
        values = [policy['correct'] if ok else policy['blank'] if e['reason'] in BLANKS else
                  policy['abstain'].get(category, 0.0) if not e['final_call'] and is_giveup(e['final'] or '') else
                  policy['wrong'] for e, ok in zip(group, correct)]
        lengths = [sum(e['tokens']) for e in group]
        if policy.get('length_weight'):
            values = list(np.asarray(values) + policy['length_weight'] * group_length_term(lengths, correct))
        total += float(np.dot(leave_one_out(values), lengths))
    return total


class Triggers:
    def __init__(self, limits: dict | None = None):
        self.limits, self.rows, self.skips, self.pushes = dict(DEFAULTS, **(limits or {})), [], 0, 0
        self.levels = None  # the baseline's per-category entropy levels, fitted once the baseline is complete

    def check(self, values: dict) -> str | None:
        """Add one update's row; the reason to stop, or None."""
        lim = self.limits
        self.skips = self.skips + 1 if values['skipped'] else 0
        if self.skips >= lim['skips_max']:
            return f"{self.skips} consecutive skipped updates (mask left nothing to learn)"
        if values['skipped']:
            return None
        self.rows.append(values)
        push = values.get('advantage_length_sum', float('nan'))
        self.pushes = self.pushes + 1 if np.isfinite(push) and push > 0 else 0
        if lim['push_positive'] and self.pushes >= lim['push_positive']:
            return (f"advantage_length_sum positive for {self.pushes} consecutive stepped updates "
                    f"(the sequence term pushes toward longer episodes)")
        if len(self.rows) < max(lim['base'], lim['window']):
            return None

        def mean(key, rows):
            got = [r[key] for r in rows if np.isfinite(r[key])]
            return float(np.mean(got)) if got else float('nan')
        now, base = self.rows[-lim['window']:], self.rows[:lim['base']]
        entropy_limit = lim['entropy_ratio'] * mean('entropy', base)
        if lim.get('entropy_composition') and all(r.get('shares') for r in base + now):
            if self.levels is None:
                self.levels = category_levels(base, lim['entropy_ridge'])
            entropy_limit = lim['entropy_ratio'] * expected_entropy(now, self.levels)
        if lim['push_window'] and len(self.rows) >= lim['base'] + lim['window']:
            pushed = [r.get('advantage_length_sum', float('nan')) for r in now]
            if all(np.isfinite(pushed)) and sum(pushed) > 0:
                return (f"advantage_length_sum {sum(pushed):.4g} > 0 summed over the last {lim['window']} stepped "
                        f"updates (net push toward longer episodes)")
        tests = (('kl', now, lim['kl_max'], '>'), ('entropy', now, entropy_limit, '<'),
                 ('finished_length', now, (1 + lim['length_growth']) * mean('finished_length', base), '>'),
                 ('capped_turn_rate', now, lim['capped_max'], '>'),
                ('capped_turn_rate', now, max(lim['capped_ratio'] * mean('capped_turn_rate', base), lim['capped_floor']),
                 '>'),
                 ('mean_reward', now, mean('mean_reward', base) - lim['reward_drop'], '<'))
        for key, rows, limit, side in tests:
            value = mean(key, rows)
            if np.isfinite(value) and np.isfinite(limit) and (value > limit if side == '>' else value < limit):
                return f"{key} {value:.4g} {side} {limit:.4g} (mean of the last {lim['window']} stepped updates)"
        return None


def limits_arg(text: str) -> dict:
    return {k: float(v) if '.' in v or 'e' in v else int(v) for k, v in (p.split('=') for p in text.split(','))}


def as_run_push(groups) -> float:
    """sum(advantage x tokens) of the update as trained, from its saved groups (trained rewards and lengths)."""
    from .rlvr import leave_one_out
    return sum(float(np.dot(leave_one_out(g['rewards']), g['tokens'])) for g in groups or [] if len(g['rewards']) >= 2)


def replay(run: Path, limits: dict | None = None, policy: dict | None = None,
           categories=None) -> tuple[int, str] | None:
    """The first update (1-based) at which the thresholds would have stopped a finished run, and why. With
    `policy`, advantage_length_sum is recomputed from the saved episodes under that reward (`replayed_push`)."""
    from daycare.artifact.record_xml import read
    triggers = Triggers(limits)
    for number, path in enumerate(sorted(run.glob('update-*/update.xml')), 1):
        result = read(path)
        values = row(result)
        if categories:  # the run-3 mix: rates from those groups' episodes only
            values.update(mix_row(result.get('episodes'), categories))
        if policy == 'as_run':
            values['advantage_length_sum'] = as_run_push(result.get('groups'))
        elif policy is not None:
            values['advantage_length_sum'] = replayed_push(result.get('episodes'), policy, categories)
        if reason := triggers.check(values):
            return number, reason
    return None


# the run-3 training reward (research/rloo-posttool-calculator-r3.md), for replays on runs 1-2
RUN3_REWARD = dict(name='graded', correct=1.0, wrong=-1.0, blank=-1.5, length_weight=1.0,
                   abstain=dict(relay=-0.5, repair=0.0, miss=0.0, empty=0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--limits', type=limits_arg, default={}, help='overrides, e.g. kl_max=0.02,window=10')
    parser.add_argument('--run3-reward', action='store_true', help='recompute advantage_length_sum under RUN3_REWARD')
    parser.add_argument('--categories', nargs='+', help='with --run3-reward: only these groups train')
    args = parser.parse_args()
    for run in args.runs:
        print(run, replay(run, args.limits, RUN3_REWARD if args.run3_reward else None, args.categories) or 'no trip',
              flush=True)


if __name__ == '__main__':
    main()
