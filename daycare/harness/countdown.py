"""Countdown validity and exact arithmetic."""
import ast
from collections import Counter
from fractions import Fraction
import re


def check_expression(expression, numbers, target, *, allow_unary=False):
    """Evaluate only integer leaves and + - * /; enforce the input multiset."""
    if len(expression)>512:return dict(correct=False,error='expression_too_long')
    used=[]
    def visit(node):
        if allow_unary and isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=visit(node.operand)
            return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value>0:
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            return a/b
        raise ValueError('unsupported_expression')
    try:
        value=visit(ast.parse(expression.strip(),mode='eval').body)
    except (ValueError,SyntaxError,ZeroDivisionError,RecursionError) as error:
        return dict(correct=False,error=type(error).__name__)
    valid=Counter(used)==Counter(numbers)
    return dict(correct=valid and value==target,number_use_valid=valid,value=str(value),
                error=None if valid and value==target else 'number_use' if not valid else 'wrong_target')


def score_answer(text,numbers,target,completed=True,truncated=False):
    tags=re.findall(r'<answer>(.*?)</answer>',text,re.S|re.I)
    expression=tags[-1].strip() if tags else None
    result=check_expression(expression,numbers,target) if expression else dict(correct=False,error='missing_answer_tags')
    return dict(result,expression=expression,strict_format=bool(tags),
                correct=bool(result['correct'] and completed and not truncated))


def solve(numbers,target):
    """Exhaustive rational solver; witnesses are never placed in model prompts."""
    seen=set()
    def search(items):
        key=tuple(sorted(v for v,_ in items))
        if key in seen:return None
        seen.add(key)
        if len(items)==1:return items[0][1] if items[0][0]==target else None
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                (a,ea),(b,eb)=items[i],items[j]
                rest=[item for k,item in enumerate(items) if k not in (i,j)]
                choices=[(a+b,f'({ea}+{eb})'),(a*b,f'({ea}*{eb})'),
                         (a-b,f'({ea}-{eb})'),(b-a,f'({eb}-{ea})')]
                if b:choices.append((a/b,f'({ea}/{eb})'))
                if a:choices.append((b/a,f'({eb}/{ea})'))
                for candidate in choices:
                    found=search(rest+[candidate])
                    if found:return found
        return None
    return search([(Fraction(n),str(n)) for n in numbers])


def wrong_expression(task):
    """A valid all-numbers expression that misses the target (the post-result probe's miss construction)."""
    nums = task['numbers']
    for op in ('+', '-', '*'):
        expression = op.join(map(str, nums))
        result = check_expression(expression, nums, task['target'])
        if result.get('number_use_valid') and not result['correct']:
            return expression
    raise ValueError('cannot construct a wrong arithmetic attempt')


# TinyZero's Countdown source; its public test region starts at row 327680.
SOURCE = 'https://datasets-server.huggingface.co/rows?dataset=Jiayi-Pan%2FCountdown-Tasks-3to4&config=default&split=train'


def request_text(numbers, target):
    """The task prompt every Countdown split uses (the Countdown baseline note)."""
    return (f'Using the numbers {numbers}, make {target}. Use every supplied number exactly once, with only +, -, *, / '
            'and parentheses. Fractional and negative intermediate results are allowed. You may use the available '
            'calculator to check your work. Give a brief explanation and finish with <answer>your arithmetic '
            'expression</answer>. Return an expression, not just the target number.')


def fetch_tasks(offset, size, count, exclude, *, split='holdout'):
    """`count` solvable `size`-number tasks from the source at `offset`, skipping `exclude` keys."""
    import json
    import urllib.request
    tasks, keys = [], set(exclude)
    while len(tasks) < count:
        with urllib.request.urlopen(f'{SOURCE}&offset={offset}&length=100', timeout=60) as response:
            page = json.load(response)['rows']
        if not page:
            raise ValueError('source exhausted before enough tasks were found')
        for source in page:
            numbers, target = source['row']['nums'], source['row']['target']
            key = (target, tuple(sorted(numbers)))
            if len(numbers) != size or key in keys or solve(numbers, target) is None:
                continue
            keys.add(key)
            tasks.append(dict(id=f"{split}-{source['row_idx']}", split=split, family='math', numbers=numbers,
                              target=target, answer=str(target), request=request_text(numbers, target)))
            if len(tasks) == count:
                break
        offset += 100
    return tasks
