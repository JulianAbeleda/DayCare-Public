from daycare.harness.countdown import check_expression,score_answer,solve


def test_exact_numbers_and_operations():
    assert check_expression('(4*5)+3',[3,4,5],23)['correct']
    assert check_expression('3/(1-3/4)',[3,1,3,4],12)['correct']
    assert not check_expression('23',[3,4,5],23)['correct']
    assert not check_expression('3+3',[3,4],6)['correct']
    for bad in ['__import__("os")','3**4','3//4','3.0+4','-3+4','True+4']:
        assert not check_expression(bad,[3,4],7)['correct']
    assert not check_expression('3/(4-4)',[3,4,4],7)['correct']


def test_final_expression_and_completion():
    assert score_answer('<answer>4*5+3</answer>',[3,4,5],23)['correct']
    assert not score_answer('I considered 4*5+3. Answer: 23',[3,4,5],23)['correct']
    assert not score_answer('<answer>4*5+3</answer>',[3,4,5],23,truncated=True)['correct']
    assert not score_answer('<answer>4*5+3</answer><answer>4+5+3</answer>',[3,4,5],23)['correct']


def test_solver_witnesses_are_independently_validated():
    for numbers,target in [([3,4,5],23),([99,11,48,46],44),([3,3,1,4],12),([2,2,2],6)]:
        witness=solve(numbers,target)
        assert witness and check_expression(witness,numbers,target)['correct']
    assert solve([1,1,1],100) is None


def test_reward_accepts_notation_without_changing_strict_baseline():
    from daycare.harness.countdown_reward import score
    for text in ['<answer>-(42/(5-12))</answer>', 'Answer: 42 ÷ (12 − 5) = 6',
                 '`42/(12-5)`', '<answer>42/(12-5)', 'Expression: +42 / (12 - 5).']:
        assert score(text,[42,5,12],6)['correct'],text
    assert not score_answer('<answer>-(42/(5-12))</answer>',[42,5,12],6)['correct']


def test_reward_rejects_shortcuts_and_conflicting_final_answers():
    from daycare.harness.countdown_reward import score
    for text in ['6', '42/(12-5) = 7', '42/(12-5)+5-5',
                 '<answer>42/(12-5)</answer> Actually, no.',
                 '42/(12-5) works.\nAnswer: 42+12+5', 'Answer: __import__("os")']:
        assert not score(text,[42,5,12],6)['correct'],text
    assert not score('<answer>42/(12-5)</answer>',[42,5,12],6,truncated=True)['correct']
