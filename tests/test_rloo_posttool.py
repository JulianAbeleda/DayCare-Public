"""Post-tool episodes (daycare/nursery/rloo_posttool.py) on the one-stack loop, tiny Nemotron-H, in a child process.

The tiny model cannot speak the chat template, so the decode function maps sampled tokens to turns: a rollout
whose first token is odd "calls" calculate, anything else answers; even tokens end a turn. That drives follow-up turns (a new prompt
holding the call and its real calculator result), the episode cap, reward scoring and one RLOO update whose
tokens are exactly the sampled ones: prompts and tool results never enter the loss.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

EXP = Path(os.environ.get("DAYCARE_TRAIN_TINYGRAD_PATH") or "$DAYCARE_TRAIN_TINYGRAD_PATH")  # tinygrad-arkey exp
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_tiny() -> dict:
    import numpy as np
    from test_rloo_tinygrad import tiny_model
    from daycare.harness import posttool
    from daycare.nursery import rloo_posttool as post

    CALL = '</think>\n<tool_call>\n<function=calculate>\n<parameter=expression>\n12 / 5\n</parameter>\n</function>\n</tool_call>'

    def decode(tokens):
        return CALL if tokens and tokens[0] % 2 else '</think>\n<answer>3</answer>'

    class TinyRenderer:  # prompt ids grow with each exchange, as a rendered later-turn request does
        envelope = dict(tools=[dict(type='function', function=dict(name='calculate', parameters=dict(
            type='object', properties=dict(expression=dict(type='string')))))])

        def ids(self, task, exchanges):
            return [3, 17, 5, 42] + [9, 11] * len(exchanges) + [task['seed'], 2]

    cfg = dict(rank=4, alpha=8, lr=1e-2, group=4, prompts_per_update=2, kl_beta=1e-3, entropy_beta=1e-3,
               max_grad_norm=1.0, temperature=1.0, limit=6, logprob_tolerance=0.1, seed=7, lanes=8)
    def make_loop():  # the 4-token "envelope" is the shared prefix, primed once and kept across calls
        return post.one.Loop(tiny_model(), cfg, np.zeros(64, dtype=np.float32), stop=set(range(0, 64, 2)),
                             prefix_capacity=32, piece=8, decode=decode, lanes=8, prompts=8, shared_capacity=4)
    loop = make_loop()
    states = [dict(id=f's{i}', task=dict(id=f't{i}', seed=20 + i, rule='numeric', answer='3'), exchanges=[])
              for i in range(2)]
    with posttool.Calculator() as calculator:
        episodes, stats = post.Episodes(loop, TinyRenderer(), calculator, cap=3).run(states, cfg['group'])
        marks = post.score(episodes)
        sampled = [[post.merged(e) for e in group] for group in episodes]
        result = loop.learn(sampled, marks, {}, dict(stats))
        # the training path: rloo_tinygrad.run driving the episode source for one more update, records written
        record = dict(losses=[], config=cfg)
        root = Path(tempfile.mkdtemp())
        post.one.run(loop, states, None, dict(cfg, updates=1), record, root, None, export=False, keep_every=1,
                     step=post.stepper(loop, post.Episodes(loop, TinyRenderer(), calculator, cap=3), states, 4))
        run_update = record['losses'][0]
        # same seed, fresh loop: the same episodes, token for token
        again = post.Episodes(make_loop(), TinyRenderer(), calculator, cap=3).run(states, cfg['group'])[0]
        # run-1 policies (outcome, none) through the stepper == the direct score-and-learn path, bit for bit
        direct_loop = make_loop()
        eps, st = post.Episodes(direct_loop, TinyRenderer(), calculator, cap=3).run(states, cfg['group'])
        direct = direct_loop.learn([[post.merged(e) for e in g] for g in eps], post.score(eps), {}, dict(st))
        policy_loop = make_loop()
        via = post.stepper(policy_loop, post.Episodes(policy_loop, TinyRenderer(), calculator, cap=3), states, 4,
                           dict(name='outcome'), 'none')([0, 1])
        regression = [direct['metrics'][k] == via['metrics'][k] for k in ('loss', 'policy_loss', 'kl', 'entropy', 'grad_norm')]
        # overlong filter + soft penalty: truncated episodes leave the update but stay in the rates
        filter_loop = make_loop()
        filtered = post.stepper(filter_loop, post.Episodes(filter_loop, TinyRenderer(), calculator, cap=3), states, 4,
                                dict(name='soft_overlong', cap=6, buffer=2), 'overlong_filter')([0, 1])
        eps_f = filtered['episodes']
        truncated = sum(e['stops'][-1] == 'limit' for g in eps_f for e in g)
        kept_groups = [[e for e in g if e['stops'][-1] != 'limit'] for g in eps_f]
        kept_tokens = sum(sum(e['tokens']) for g in kept_groups if len(g) >= 2 for e in g)
        # run 3: graded reward, capped episodes kept; a zero repetition penalty is bit-neutral, a nonzero one moves
        graded = dict(name='graded', correct=1.0, wrong=-1.0, blank=-1.5, abstain=dict(s0=0.25, s1=0.25))
        graded_runs = {}
        for label, repetition, weight in (('plain', None, 0), ('zero', dict(n=1, penalty=0.0), 0),
                                          ('rep', dict(n=1, penalty=-0.05), 0), ('len', None, 1.0)):
            graded_loop = make_loop()
            policy = dict(graded, **({'repetition': dict(repetition, think_end=None)} if repetition else {}),
                          **({'length_weight': weight} if weight else {}))
            graded_runs[label] = post.stepper(graded_loop, post.Episodes(graded_loop, TinyRenderer(), calculator, cap=3),
                                              states, 4, policy, 'none')([0, 1])
        # the dense per-token bonus through learn: zeros are bit-neutral, a penalty on every token moves the loss
        bonus_losses = []
        for value in (None, 0.0, -0.05):
            bonus_loop = make_loop()
            eps_b, st_b = post.Episodes(bonus_loop, TinyRenderer(), calculator, cap=3).run(states, cfg['group'])
            views_b = [[post.merged(e) for e in g] for g in eps_b]
            for view in (v for g in views_b for v in g):
                if value is not None:
                    view['token_bonus'] = np.full(len(view['tokens']), value, dtype=np.float32)
            bonus_losses.append(bonus_loop.learn(views_b, post.score(eps_b), {}, dict(st_b))['metrics']['policy_loss'])
        # slot compaction (research/rloo-slot-refill.md): the step shrinks to the running lanes, parity stays exact
        compact = None
        if 'compact: tuple' in (EXP / 'tinygrad/llm/nemotron_h_sampler.py').read_text():
            compact_loop = post.one.Loop(tiny_model(), dict(cfg, compact=[2, 4]), np.zeros(64, dtype=np.float32),
                                         stop=set(range(0, 64, 2)), prefix_capacity=32, piece=8, decode=decode, lanes=8,
                                         prompts=8, shared_capacity=4)
            eps_c, st_c = post.Episodes(compact_loop, TinyRenderer(), calculator, cap=3).run(states, cfg['group'])
            m_c = compact_loop.learn([[post.merged(e) for e in g] for g in eps_c], post.score(eps_c), {},
                                     dict(st_c))['metrics']
            compact = dict(max_error=m_c['max_logprob_error'], exact=m_c['exact'], checked=m_c['checked'],
                           sampled=sum(len(t['tokens']) for g in eps_c for e in g for t in e['turns']),
                           row_steps=st_c['row_steps'], lane_steps=st_c['lane_steps'], windows=st_c['window_rows'],
                           turns=sum(len(e['turns']) for g in eps_c for e in g))
    same_seed = [[[t['tokens'] for t in e['turns']] for e in g] for g in again] == \
        [[[t['tokens'] for t in e['turns']] for e in g] for g in episodes]
    sampled_tokens = sum(len(t['tokens']) for g in episodes for e in g for t in e['turns'])
    # every group truncated: nothing to learn, the update is skipped instead of raising (learn records it)
    all_capped = post.masked([[dict(stop='limit'), dict(stop='eos')]] * 2, [[(0.0, 'x'), (1.0, 'y')]] * 2,
                             'overlong_filter')
    skipped = loop.learn(*all_capped[:2], {}, {})['metrics']
    graded_summary = {k: dict(loss=v['metrics']['loss'], policy_loss=v['metrics']['policy_loss'],
                              repeated=v['metrics']['repeated_thinking_tokens'], giveup=v['metrics']['giveup_rate'],
                              a_len=v['metrics']['advantage_length_sum'], checked=v['metrics']['checked'],
                              rewards=v['groups'][0]['rewards'] + v['groups'][1]['rewards'],
                              reasons=v['groups'][0]['reasons'] + v['groups'][1]['reasons'],
                              lengths=[sum(e['tokens']) for g in v['episodes'] for e in g])
                      for k, v in graded_runs.items()}
    return dict(compact=compact, stats=stats, metrics=result['metrics'], graded=graded_summary, bonus_losses=bonus_losses, sampled_tokens=sampled_tokens, run_update=run_update,
                all_capped=[len(x) for x in all_capped], skipped=[skipped['skipped'], skipped['stepped']],
                same_seed=same_seed, regression=regression, truncated=truncated, kept_tokens=kept_tokens,
                filtered_checked=filtered['metrics']['checked'], filtered_trained=filtered['metrics']['trained_episodes'],
                kept_episodes=sum(len(g) for g in kept_groups if len(g) >= 2),
                capped_rate=filtered['metrics']['capped_turn_rate'],
                update_files=sorted(p.name for p in (root / 'update-000').iterdir()),
                checkpoint=(root / 'checkpoint-001.npz').exists(),
                turns=[[len(e['turns']) for e in g] for g in episodes],
                results=[[e['exchanges'][-1]['content'] if e['exchanges'] else None for e in g] for g in episodes],
                rewards=[[r for r, _ in m] for m in marks], reasons=[[w for _, w in m] for m in marks],
                calls_ended=[[e['turns'][-1]['call'] is not None for e in g] for g in episodes])


def run_progress_check() -> dict:
    """`Progress` streamed during `sample`'s episode loop, on the same tiny exam as `run_tiny`: with
    `time.perf_counter` pinned to a constant, two fresh loops on the same seed (Progress off, then on) must reach
    the exact same episodes, so their `exam_record` output (sample.xml) is byte-identical either way; and
    progress.jsonl must carry one partial line per state whose rewards match that record's rows."""
    import time
    time.perf_counter = lambda: 42.0
    import numpy as np
    from test_rloo_tinygrad import tiny_model
    from daycare.harness import posttool
    from daycare.nursery import rloo_posttool as post

    CALL = '</think>\n<tool_call>\n<function=calculate>\n<parameter=expression>\n12 / 5\n</parameter>\n</function>\n</tool_call>'

    def decode(tokens):
        return CALL if tokens and tokens[0] % 2 else '</think>\n<answer>3</answer>'

    class TinyRenderer:
        envelope = dict(tools=[dict(type='function', function=dict(name='calculate', parameters=dict(
            type='object', properties=dict(expression=dict(type='string')))))])

        def ids(self, task, exchanges):
            return [3, 17, 5, 42] + [9, 11] * len(exchanges) + [task['seed'], 2]

    cfg = dict(rank=4, alpha=8, lr=1e-2, group=4, prompts_per_update=2, kl_beta=1e-3, entropy_beta=1e-3,
               max_grad_norm=1.0, temperature=1.0, limit=6, logprob_tolerance=0.1, seed=7, lanes=8)

    def make_loop():
        return post.one.Loop(tiny_model(), cfg, np.zeros(64, dtype=np.float32), stop=set(range(0, 64, 2)),
                             prefix_capacity=32, piece=8, decode=decode, lanes=8, prompts=8, shared_capacity=4)

    states = [dict(id=f's{i}', category='repair', split='train',
                  task=dict(id=f't{i}', seed=20 + i, rule='numeric', answer='3'), exchanges=[])
              for i in range(3)]
    meta = dict(k=cfg['group'], cap=3, seed=cfg['seed'], split='train', categories=['repair'],
               adapter='zero (stock)', limit=cfg['limit'], load_s=0.0, source='tiny', source_sha256='deadbeef',
               tinygrad_revision='tiny-rev')

    with posttool.Calculator() as calculator:
        loop_off = make_loop()
        episodes_off, stats_off = post.Episodes(loop_off, TinyRenderer(), calculator, cap=3).run(states, cfg['group'])
        record_off = post.exam_record(states, episodes_off, stats_off, 0.0, **meta)

        log_lines: list[str] = []
        progress_path = Path(tempfile.mkdtemp()) / 'progress.jsonl'
        progress = post.Progress(progress_path, states, cfg['group'], log=log_lines.append)
        loop_on = make_loop()
        episodes_on, stats_on = post.Episodes(loop_on, TinyRenderer(), calculator, cap=3,
                                              on_done=progress.done).run(states, cfg['group'])
        progress.file.close()
        record_on = post.exam_record(states, episodes_on, stats_on, 0.0, **meta)

    path_off, path_on = Path(tempfile.mkdtemp()) / 'a.xml', Path(tempfile.mkdtemp()) / 'b.xml'
    post.write(path_off, record_off, root='sample')
    post.write(path_on, record_on, root='sample')
    xml_equal = path_off.read_bytes() == path_on.read_bytes()

    lines = progress_path.read_text().splitlines()
    rewards_by_state = {row['state']: row['rewards'] for row in record_on['rows']}
    # `Progress` appends episodes in their own finishing order (a tool-call round trip reorders a group relative
    # to its fixed k-index slot), so the two agree as multisets of reward values, not necessarily position for
    # position; `sorted` compares the multiset.
    rewards_match = len(lines) == len(states) and all(
        sorted(json.loads(line)['rewards']) == sorted(rewards_by_state[json.loads(line)['state']]) for line in lines)

    return dict(xml_equal=xml_equal, num_lines=len(lines), num_states=len(states), rewards_match=rewards_match,
               has_partial_log=any('PARTIAL' in line for line in log_lines),
               sample_partial_line=lines[0] if lines else None,
               sample_log_line=next((line for line in log_lines if 'PARTIAL' in line), None))


@pytest.mark.skipif(not (EXP / "tinygrad/llm/nemotron_h_sampler.py").exists()
                    or "follow=None" not in (EXP / "tinygrad/llm/nemotron_h_sampler.py").read_text(),
                    reason="needs tinygrad-arkey exp >= 4322ae9e2 (generate(follow=), shared-prefix reuse)")
@pytest.mark.skipif(not Path(os.environ.get("DAYCARE_CALCULATE_RUNNER") or "$DAYCARE_CALCULATE_RUNNER").exists(),
                    reason="GameTerm calculate runner not built")
def test_progress_streams_partial_states_and_leaves_the_record_unchanged():
    env = dict(os.environ, DAYCARE_TRAIN_TINYGRAD_PATH=str(EXP), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env.setdefault("DEV", "CPU")
    done = subprocess.run([sys.executable, __file__, "progress"], env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["xml_equal"]  # sample.xml is unaffected by whether Progress is streaming alongside it
    assert report["num_lines"] == report["num_states"] == 3
    assert report["rewards_match"]
    assert report["has_partial_log"] and "PARTIAL" in report["sample_log_line"]
    assert json.loads(report["sample_partial_line"])["partial"] is True


@pytest.mark.skipif(not (EXP / "tinygrad/llm/nemotron_h_sampler.py").exists()
                    or "follow=None" not in (EXP / "tinygrad/llm/nemotron_h_sampler.py").read_text(),
                    reason="needs tinygrad-arkey exp >= 4322ae9e2 (generate(follow=), shared-prefix reuse)")
@pytest.mark.skipif(not Path(os.environ.get("DAYCARE_CALCULATE_RUNNER") or "$DAYCARE_CALCULATE_RUNNER").exists(),
                    reason="GameTerm calculate runner not built")
def test_multi_turn_episodes_train_only_sampled_tokens():
    env = dict(os.environ, DAYCARE_TRAIN_TINYGRAD_PATH=str(EXP), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env.setdefault("DEV", "CPU")
    done = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    turns = [n for group in report["turns"] for n in group]
    # some episodes called and continued (a later round), none past the cap; a call at the cap scores 0
    assert report["stats"]["turns"] == sum(turns) and max(turns) <= 3 and any(n > 1 for n in turns)
    for group, ended, rewards, reasons in zip(report["turns"], report["calls_ended"], report["rewards"],
                                              report["reasons"]):
        for n, call, reward, why in zip(group, ended, rewards, reasons):
            assert why in ("correct", "call_at_cap", "incomplete") and reward == float(why == "correct")
            assert why != "call_at_cap" or (call and n == 3)
    assert "correct" in {w for group in report["reasons"] for w in group}
    # the real calculator ran: GameTerm's wire result for 12 / 5
    assert any(r and '"stdout":"12 / 5 = 2.4"' in r for group in report["results"] for r in group)
    metrics = report["metrics"]
    # loss tokens are exactly the sampled tokens of every turn, reproduced bit for bit from the captures
    assert metrics["checked"] == report["sampled_tokens"] == metrics["generated_tokens"]
    assert metrics["max_logprob_error"] == 0.0 and metrics["exact"] == metrics["checked"]
    assert metrics["trajectories"] == 8 and metrics["stepped"]
    run_update = report["run_update"]
    assert run_update["max_logprob_error"] == 0.0 and run_update["checked"] == run_update["generated_tokens"]
    assert {"sample_s", "grad_s", "adam_s", "parity_s", "tool_s", "score_s", "other_tool_rate",
            "capped_turn_rate", "empty_after_tool_rate"} <= set(run_update)
    assert report["update_files"] == ["update.xml"] and report["checkpoint"]
    assert report["same_seed"] and report["stats"]["shared_tokens"] == 4
    assert all(report["regression"])  # run-1 policies reproduce the unrefactored update bit for bit
    # overlong_filter: truncated episodes contribute no token to the loss, yet the capped rate still counts them
    assert report["truncated"] > 0 and report["capped_rate"] > 0
    assert report["all_capped"] == [0, 0, 0] and report["skipped"] == [True, False]
    assert report["filtered_checked"] == report["kept_tokens"] and report["filtered_trained"] == report["kept_episodes"]
    # graded (run 3): capped episodes stay in the loss (every sampled token checked) and score below a wrong answer
    graded = report["graded"]
    for run in graded.values():
        assert run["checked"] == report["sampled_tokens"] and run["giveup"] == 0.0
        for value, why in zip(run["rewards"], run["reasons"]) if run is not graded["len"] else ():
            assert value == {"correct": 1.0, "incomplete": -1.5, "call_at_cap": -1.5}[why]
    assert graded["zero"]["loss"] == graded["plain"]["loss"] and graded["rep"]["repeated"] == graded["zero"]["repeated"]
    assert (graded["rep"]["policy_loss"] != graded["plain"]["policy_loss"]) == (graded["rep"]["repeated"] > 0)
    # the Kimi length term moves each group's rewards by weight x lambda (same seed: the same episodes as `plain`)
    from daycare.harness.posttool import group_length_term
    for g in range(2):
        rows = slice(4 * g, 4 * g + 4)
        plain, lengths = graded["plain"]["rewards"][rows], graded["len"]["lengths"][rows]
        term = group_length_term(lengths, [r == 1.0 for r in plain])
        assert graded["len"]["rewards"][rows] == pytest.approx([r + t for r, t in zip(plain, term)])
    plain, zero, penalized = report["bonus_losses"]
    assert plain == zero and penalized != plain
    if report["compact"] is not None:  # compacted windows ran, and every sampled token is recomputed bit for bit
        compact = report["compact"]
        assert compact["row_steps"] < compact["lane_steps"] and compact["turns"] > 8
        assert compact["max_error"] == 0.0 and compact["exact"] == compact["checked"] == compact["sampled"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "progress":
        print(json.dumps(run_progress_check(), default=float))
    else:
        print(json.dumps(run_tiny(), default=float))


@pytest.mark.skipif(not (EXP / "tinygrad/llm/nemotron_h_sampler.py").exists(), reason="tinygrad-arkey exp not installed")
def test_think_end_id_is_asserted_not_guessed():
    from types import SimpleNamespace
    from daycare.nursery import rloo_posttool as post
    assert post.think_end_id(SimpleNamespace(encode=lambda text: [13])) == 13
    for ids in ([], [14], [60, 13]):  # a changed tokenizer must stop the run, not silently penalize answers
        with pytest.raises(ValueError, match='expected \\[13\\]'):
            post.think_end_id(SimpleNamespace(encode=lambda text, ids=ids: ids))


@pytest.mark.skipif(not (EXP / "tinygrad/llm/nemotron_h_sampler.py").exists(), reason="tinygrad-arkey exp not installed")
def test_graded_policy_carries_the_length_weight():
    from types import SimpleNamespace
    from daycare.nursery import rloo_posttool as post
    args = SimpleNamespace(reward='graded', wrong=-1.0, blank=-1.5, abstain=dict(relay=-0.5), repetition=None,
                           length_weight=1.0)
    assert post.reward_policy(args, 13)['length_weight'] == 1.0
    assert 'length_weight' not in post.reward_policy(SimpleNamespace(**dict(vars(args), length_weight=0.0)), 13)
