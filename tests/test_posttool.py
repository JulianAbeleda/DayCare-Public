"""GameTerm post-tool rules (daycare/harness/posttool.py): parsing a turn, the calculator's wire results, scoring."""
from fractions import Fraction
import json

import pytest

from daycare.harness import posttool

TOOLS = [dict(type='function', function=dict(name='calculate', parameters=dict(type='object', properties=dict(
    expression=dict(type='string'), decimals=dict(type='integer')))))]
CALL = ('I will compute it.</think>\n<tool_call>\n<function=calculate>\n<parameter=expression>\n12 / 5\n</parameter>\n'
        '<parameter=decimals>\n2\n</parameter>\n</function>\n</tool_call>')


def test_parse_turn_splits_reasoning_and_types_arguments_like_llama_server():
    turn = posttool.parse_turn(CALL, TOOLS)
    assert turn['closed'] and turn['reasoning'] == 'I will compute it.' and turn['content'] == ''
    assert turn['call']['arguments_json'] == '{"expression":"12 / 5","decimals":2}'
    # a call inside unfinished reasoning is not a call; an answer is content
    assert posttool.parse_turn(CALL.replace('</think>', ''), TOOLS)['call'] is None
    assert posttool.parse_turn('ok</think>\n<answer>3</answer>', TOOLS)['content'] == '<answer>3</answer>'


def test_later_messages_match_gameterm_wire_shape():
    envelope = dict(messages=[dict(role='system', content='s'), dict(role='user', content='{{REQUEST}}')])
    step = dict(id='call_0', name='calculate', arguments_json='{"expression":"1+1"}', content='{"transport":"x"}')
    messages = posttool.later_messages(envelope, 'q', [step])
    assert messages[1] == dict(role='user', content='q')
    assert messages[2] == {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_0', 'type': 'function',
                           'function': {'name': 'calculate', 'arguments': '{"expression":"1+1"}'}}]}
    assert messages[3] == {'role': 'tool', 'content': '{"transport":"x"}', 'tool_call_id': 'call_0'}


@pytest.mark.skipif(not posttool.RUNNER.exists(), reason='GameTerm calculate runner not built')
def test_calculator_results_follow_terminal_session():
    with posttool.Calculator() as calculator:
        answer = json.loads(calculator.content('{"expression":"12 / 5"}')[0])
        rejected = json.loads(calculator.content('{"expr":"12 / 5"}')[0])
        refused = json.loads(calculator.content('{"expression":"1/0"}')[0])
    assert answer == dict(transport='succeeded', stdout='12 / 5 = 2.4', stderr='', exit_code=None, truncated=False)
    assert rejected['transport'] == 'rejected' and rejected['stderr'] == 'malformed arguments: see the tool schema'
    # a calculator refusal is an answer too: succeeded, the refusal on stdout
    assert refused['transport'] == 'succeeded' and refused['stdout'].startswith('could not calculate `1/0`')


def test_numeric_answer_rule():
    assert posttool.numeric_answer('<answer>$1,119.58</answer>') == Fraction('1119.58')
    assert posttool.numeric_answer('so <answer>27.9%</answer>') == Fraction('27.9')
    assert posttool.numeric_answer('<answer>24 buses</answer>') == 24
    assert posttool.numeric_answer('<answer>24</answer> more text') is None
    assert posttool.numeric_answer('<answer>about 24</answer>') is None
    assert posttool.numeric_answer('24') is None


def test_episode_reward_scores_only_a_final_answer():
    task = dict(rule='numeric', answer='24')

    def episode(content, *, call=None, stop='eos', closed=True):
        return dict(turns=[dict(content=content, call=call, stop=stop, closed=closed)])
    assert posttool.episode_reward(task, episode('<answer>24</answer>')) == (1.0, 'correct')
    assert posttool.episode_reward(task, episode('<answer>23</answer>')) == (0.0, 'wrong')
    assert posttool.episode_reward(task, episode('')) == (0.0, 'empty')
    assert posttool.episode_reward(task, episode('<answer>24</answer>', stop='limit')) == (0.0, 'incomplete')
    assert posttool.episode_reward(task, episode('', call=dict(name='calculate'))) == (0.0, 'call_at_cap')
    countdown = dict(rule='countdown', numbers=[2, 3, 4], target=14)
    assert posttool.episode_reward(countdown, episode('<answer>2 + 3 * 4</answer>'))[0] == 1.0


def test_soft_overlong_penalty_follows_dapo_eq13():
    # DAPO (Yu et al., arXiv:2503.14476) Eq. 13 with cap 4096, buffer 819: flat, then linear to -1 at the cap
    assert posttool.soft_overlong_penalty(100, 4096, 819) == 0.0
    assert posttool.soft_overlong_penalty(4096 - 819, 4096, 819) == 0.0
    assert posttool.soft_overlong_penalty(4096, 4096, 819) == -1.0
    assert abs(posttool.soft_overlong_penalty(4096 - 409.5, 4096, 819) + 0.5) < 1e-12


def test_shaped_reward_policies():
    task = dict(rule='numeric', answer='24')
    short = dict(turns=[dict(content='<answer>24</answer>', call=None, stop='eos', closed=True, tokens=[1] * 100)])
    long = dict(turns=[dict(content='', call=dict(name='calculate'), stop='eos', closed=True, tokens=[1] * 50),
                       dict(content='<answer>24</answer>', call=None, stop='eos', closed=True, tokens=[1] * 3686)])
    outcome, soft = dict(name='outcome'), dict(name='soft_overlong', cap=4096, buffer=819)
    assert posttool.shaped_reward(task, short, outcome) == posttool.episode_reward(task, short) == (1.0, 'correct')
    assert posttool.shaped_reward(task, short, soft) == (1.0, 'correct')
    value, why = posttool.shaped_reward(task, long, soft)  # the longest turn sets the penalty
    assert why == 'correct' and abs(value - (1.0 - (3686 - 3277) / 819)) < 1e-12
    with pytest.raises(ValueError):
        posttool.shaped_reward(task, short, dict(name='length'))


def test_giveup_detection_is_tagless_and_explicit():
    assert posttool.is_giveup("I couldn't find an exact solution. My best guess is 911.")
    assert posttool.is_giveup("I'm not sure; I ran out of room to check it. Best guess: 21 minutes.")
    # a tagged answer is scored by its tag, hedged or not; plain prose without a give-up statement is not a give-up
    assert not posttool.is_giveup("I'm not sure, but <answer>911</answer>")
    assert not posttool.is_giveup('Since we cannot have a fraction of a bus, we need 71 buses.')
    assert not posttool.is_giveup('The population grew by 28.5%.')
    assert not posttool.is_giveup('')


def test_graded_reward_orders_correct_giveup_wrong_blank():
    task = dict(rule='numeric', answer='24')
    policy = dict(name='graded', correct=1.0, wrong=-1.0, blank=-1.5, abstain=dict(relay=0.0, repair=0.25))

    def episode(content, *, state='repair:t:own', stop='eos', closed=True, call=None):
        return dict(state=state, turns=[dict(content=content, call=call, stop=stop, closed=closed, tokens=[1])])
    score = lambda e: posttool.shaped_reward(task, e, policy)
    assert score(episode("I'm not sure, but <answer>24</answer>")) == (1.0, 'correct')
    assert score(episode("I couldn't verify it; my best guess is 24.")) == (0.25, 'giveup')
    assert score(episode("I couldn't verify it; my best guess is 24.", state='relay:t:own')) == (0.0, 'giveup')
    assert score(episode("I'm not sure, but <answer>23</answer>")) == (-1.0, 'wrong')
    assert score(episode('24 buses.')) == (-1.0, 'no_number')
    assert score(episode('<answer>24</answer>', stop='limit')) == (-1.5, 'incomplete')
    assert score(episode('')) == (-1.5, 'empty')
    assert score(episode('', closed=False)) == (-1.5, 'reasoning_not_closed')
    assert score(episode('', call=dict(name='calculate'))) == (-1.5, 'call_at_cap')
    assert score(episode('', call=dict(name='web_search'))) == (-1.0, 'other_tool')


def test_repetition_penalty_marks_repeated_thinking_ngrams():
    # Algorithm 1 of arXiv:2502.03373: every position inside an n-gram seen before
    assert posttool.repeated_positions([1, 2, 3, 1, 2, 3, 4], 3).tolist() == [False] * 3 + [True] * 3 + [False]
    assert not posttool.repeated_positions([1, 2, 3], 5).any()
    end = 9  # the </think> id: the answer after it is never penalized; a turn that never closed is all thinking
    episode = dict(turns=[dict(tokens=[5, 6, 5, 6, end, 5, 6, 5, 6]), dict(tokens=[7, 8, 7, 8])])
    hit = posttool.thinking_repetition(episode, 2, end)
    assert hit.tolist() == [False, False, True, True, False, False, False, False, False, False, False, True, True]


def test_group_length_term_follows_kimi_k15():
    # arXiv:2501.12599 Sec. 2.3.3: lambda = 0.5 - (len - min) / (max - min); correct: lambda, other: min(0, lambda)
    term = posttool.group_length_term([100, 300, 500, 500], [True, True, False, True])
    assert term.tolist() == [0.5, 0.0, -0.5, -0.5]
    assert posttool.group_length_term([100, 500], [False, False]).tolist() == [0.0, -0.5]  # short wrong earns nothing
    assert posttool.group_length_term([7, 7, 7], [True, False, True]).tolist() == [0.0, 0.0, 0.0]
