"""Conservative mathematical answer reward.

Only the stated final expression is scored. No solution search, answer repair,
or selection of a successful discarded intermediate expression.
"""
import re
from fractions import Fraction
from .countdown import check_expression


def score(text,numbers,target,*,completed=True,truncated=False):
    if not completed or truncated:
        return dict(correct=False,error='incomplete_or_truncated',expression=None)
    clean=text.translate(str.maketrans({'×':'*','÷':'/','−':'-','–':'-'})).strip()
    tags=list(re.finditer(r'<answer>(.*?)(?:</answer>|$)',clean,re.S|re.I))
    if tags:
        candidate=tags[-1].group(1).strip()
        if clean[tags[-1].end():].strip():
            return dict(correct=False,error='text_after_final_answer',expression=None)
    else:
        candidate=clean.splitlines()[-1] if clean else ''
        candidate=re.sub(r'^(?:answer|expression|final answer)\s*:\s*','',candidate,flags=re.I)
    candidate=candidate.strip('` $').removeprefix(r'\(').removesuffix(r'\)').strip()
    if candidate.startswith('**') and candidate.endswith('**'):candidate=candidate[2:-2].strip()
    if candidate.endswith('.') and not re.search(r'\d\.\d',candidate):candidate=candidate[:-1]
    parts=candidate.split('=')
    expression=parts[0].strip()
    result=check_expression(expression,numbers,target,allow_unary=True)
    if len(parts)>2:result=dict(correct=False,error='ambiguous_equation')
    elif len(parts)==2:
        try:
            if Fraction(parts[1].strip())!=target:result=dict(correct=False,error='incorrect_stated_equality')
        except (ValueError,ZeroDivisionError):result=dict(correct=False,error='unsupported_equality')
    return dict(result,expression=expression)
