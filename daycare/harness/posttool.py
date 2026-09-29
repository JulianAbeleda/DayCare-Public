"""Post-tool episodes as GameTerm serves them: the turn after a real calculator result.

GameTerm (HT-020's shared loop: `MinimalHarness` + `TerminalSession`, `crates/provider` `encode_request`)
builds every later-turn request the same way, and this module copies each rule:

- The model's turn is split like llama-server's reasoning parser: reasoning before the first `</think>`,
  content after it. Only the first completed tool call counts (the harness stops the stream there), and the
  calling assistant message carries `content: null` and no reasoning (`WireMessage`), so the model's own
  thinking before a call is not shown back to it.
- Arguments travel as llama-server's compact JSON string, values typed by the tool schema
  (`<parameter=decimals>\n2\n</parameter>` becomes `"decimals":2`).
- `calculate` runs through GameTerm's own `calculate::normalize` + `evaluate` (the runner binary wraps
  GameTerm's calculator; docs/rl-training.md): a schema error is a `rejected` result with GameTerm's fixed text on
  stderr (`terminal/result.rs`), a calculator refusal is a `succeeded` result whose stdout is the refusal
  (`TerminalSession::call_tool`), and an answer is `succeeded` with `expr = value` on stdout. Linux has no
  osascript fallback, so an expression fend refuses stays refused (the Air may answer it).
- The tool message is `WireToolResult` (`gameterm_wire.tool_content`).

A parity test against captured GameTerm requests checks these rules (not published: it needs the captures).
"""
from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
import re
import subprocess

import numpy as np

from .gameterm_wire import tool_content

# GameTerm's calculate runner (tools/calculate-runner; JSON lines on stdin/stdout): `--runner` or DAYCARE_CALCULATE_RUNNER.
RUNNER = Path(os.environ.get('DAYCARE_CALCULATE_RUNNER') or '$DAYCARE_CALCULATE_RUNNER')
# GameTerm's fixed `rejected` stderr (tools/calculate-runner rejection()) for each calculate::normalize error
REJECTIONS = {'MalformedArguments': 'malformed arguments: see the tool schema',
              'MissingField': 'missing required field', 'OutOfRange': 'value out of range'}
CALL = re.compile(r'<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>', re.S)
PARAMETER = re.compile(r'<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>', re.S)


class Calculator:
    """GameTerm's calculate tool, one persistent runner process (stdin/stdout JSON lines)."""

    def __init__(self, runner: Path = RUNNER):
        self.runner = Path(runner)
        if not self.runner.is_file():
            raise FileNotFoundError(f'GameTerm calculate runner not found at {self.runner}: build it '
                                    '(tools/calculate-runner: cargo build --release) and pass --runner or set '
                                    'DAYCARE_CALCULATE_RUNNER (docs/rl-training.md)')
        self.process = subprocess.Popen([str(self.runner)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def run(self, arguments_json: str) -> dict:
        if '\n' in arguments_json:  # compact JSON never holds a raw newline; the runner reads one line per call
            raise ValueError('arguments must be one line of JSON')
        self.process.stdin.write(arguments_json + '\n')
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def result(self, arguments_json: str) -> dict:
        """The WireToolResult fields GameTerm returns for this `calculate` call."""
        receipt = self.run(arguments_json)
        if receipt['ok']:
            return dict(transport='succeeded', stdout=receipt['outcome'], stderr='', outcome='answer')
        error = receipt['error']
        if error.startswith('arguments: '):
            return dict(transport='rejected', stdout='', stderr=REJECTIONS[error.removeprefix('arguments: ')],
                        outcome='rejected')
        return dict(transport='succeeded', stdout=error, stderr='', outcome='refused')

    def content(self, arguments_json: str) -> tuple[str, dict]:
        result = self.result(arguments_json)
        return tool_content(result['transport'], result['stdout'], result['stderr']), result

    def exchange(self, index: int, name: str, arguments_json_text: str) -> dict:
        """One executed call as a later request carries it (`later_messages`), with its outcome."""
        content, result = self.content(arguments_json_text)
        return dict(id=f'call_{index}', name=name, arguments_json=arguments_json_text, content=content,
                    transport=result['transport'], outcome=result['outcome'])

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _schema(tools: list[dict]) -> dict[str, dict]:
    return {t['function']['name']: t['function'].get('parameters', {}) for t in tools}


def _typed(value: str, kind: str | None):
    if kind in (None, 'string'):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def arguments_json(arguments: dict) -> str:
    """llama-server's argument string: compact JSON, non-ASCII kept."""
    return json.dumps(arguments, separators=(',', ':'), ensure_ascii=False)


def parse_turn(text: str, tools: list[dict]) -> dict:
    """One sampled assistant turn (the text after the `<think>\\n` generation prompt, without the end token).

    Returns reasoning, content, whether reasoning closed, and the first tool call (name + argument string) or None.
    A call written inside the reasoning is not a call (llama-server parses content only)."""
    closed = '</think>' in text
    reasoning, content = text.split('</think>', 1) if closed else (text, '')
    match = CALL.search(content)
    call = None
    if match:
        name, body = match.group(1, 2)
        props = _schema(tools).get(name, {}).get('properties', {})
        arguments = {}
        for parameter in PARAMETER.finditer(body):
            key, value = parameter.group(1, 2)
            arguments[key] = _typed(value, props.get(key, {}).get('type'))
        call = dict(name=name, arguments=arguments, arguments_json=arguments_json(arguments),
                    known=name in _schema(tools))
        content = content[:match.start()]
    return dict(reasoning=reasoning.strip(), content=content.strip(), closed=closed, call=call)


def call_message(ident: str, name: str, arguments_json_text: str) -> dict:
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': arguments_json_text}}]}


def result_message(ident: str, content: str) -> dict:
    return {'role': 'tool', 'content': content, 'tool_call_id': ident}


def first_messages(envelope: dict, request: str) -> list[dict]:
    """The first request's messages: the envelope with the user's request in the `{{REQUEST}}` slot."""
    messages = [dict(m) for m in envelope['messages']]
    slots = [i for i, m in enumerate(messages) if m.get('content') == '{{REQUEST}}']
    if len(slots) != 1:
        raise ValueError('the envelope needs exactly one {{REQUEST}} message')
    messages[slots[0]] = dict(messages[slots[0]], content=request)
    return messages


def later_messages(envelope: dict, request: str, exchanges: list[dict]) -> list[dict]:
    """A later-turn request's messages: the first request plus each (call, result) pair, in order.

    `exchanges` items hold `id`, `name`, `arguments_json` and `content` (the wire result)."""
    messages = first_messages(envelope, request)
    for step in exchanges:
        messages += [call_message(step['id'], step['name'], step['arguments_json']),
                     result_message(step['id'], step['content'])]
    return messages


# ---------------------------------------------------------------- verification

NUMBER = re.compile(r'^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$')


def numeric_answer(text: str) -> Fraction | None:
    """The number inside the last `<answer>...</answer>` with nothing after it; `$`, `,` and a trailing unit word
    are tolerated, nothing else. None when absent or unreadable."""
    tags = list(re.finditer(r'<answer>(.*?)</answer>', text, re.S | re.I))
    if not tags or text[tags[-1].end():].strip():
        return None
    value = tags[-1].group(1).strip().replace('**', '').strip()
    value = re.sub(r'^\$\s*', '', value)
    value = re.sub(r'\s*(?:%|[A-Za-z][A-Za-z /.]*)$', '', value).strip()
    if not NUMBER.match(value):
        return None
    return Fraction(value.replace(',', ''))


def verdict(task: dict, final: str) -> tuple[bool, str]:
    """The task's deterministic rule on a final answer's content (reasoning excluded)."""
    if not final.strip():
        return False, 'empty'
    if task['rule'] == 'countdown':
        from .countdown_reward import score
        result = score(final, task['numbers'], task['target'])
        return bool(result['correct']), str(result.get('error'))
    if task['rule'] == 'numeric':
        value = numeric_answer(final)
        if value is None:
            return False, 'no_number'
        return value == Fraction(task['answer']), 'correct' if value == Fraction(task['answer']) else 'wrong'
    raise ValueError(f"unknown rule {task['rule']}")


def episode_reward(task: dict, episode: dict) -> tuple[float, str]:
    """1 only for a correct, non-empty final answer by the task's rule; tool calls are neither rewarded nor
    penalized. `episode` is `{turns: [...]}` with each sampled turn's parse and `stop`; the last turn is scored."""
    if task['rule'] == 'blocked':  # the post-refusal rule (posttool_blocked)
        from .posttool_blocked import blocked_reward
        if 'base' not in episode:  # only `Episodes.run` knows where the state's exchanges end
            raise ValueError('a blocked episode needs `base` (the state exchange count); force/resume do not run them')
        return blocked_reward(task, episode, episode['base'])
    last = episode['turns'][-1]
    if last['stop'] != 'eos':
        return 0.0, 'incomplete'
    if not last['closed']:
        return 0.0, 'reasoning_not_closed'
    if last['call'] is not None:
        return 0.0, 'call_at_cap' if last['call']['name'] == 'calculate' else 'other_tool'
    ok, why = verdict(task, last['content'])
    return float(ok), why


def soft_overlong_penalty(length: int, cap: int, buffer: int) -> float:
    """DAPO's Soft Overlong Punishment (Yu et al., arXiv:2503.14476, Eq. 13): 0 while `length <= cap - buffer`,
    then ((cap - buffer) - length) / buffer, reaching -1 at the cap. Added to the outcome reward."""
    return min(0.0, ((cap - buffer) - length) / buffer)


REWARDS = ('outcome', 'soft_overlong', 'graded')

# An honest give-up (run 3, research/rloo-posttool-calculator-r3.md): a completed final turn whose content commits
# no answer (no `<answer>` tag at all) and says in words that the model did not finish or is not sure. A tagged
# answer is always scored by its tag, hedged or not, so a hedged correct answer stays correct and hedging a wrong
# one earns nothing; the abstain reward is reachable only by withholding the committed answer.
GIVEUP = re.compile(
    r"\b(?:i(?:'m| am) not (?:sure|certain|confident)|not (?:sure|certain) (?:of|about|this|that|my|if|whether)|"
    r"i (?:could(?:n't| not)|can(?:'t|not)|was(?:n't| not) able to|am unable to) "
    r"(?:find|finish|solve|determine|work out|figure out|get|reach|make|verify|confirm|complete|pin)|"
    r"i (?:don't|do not) know|no (?:valid |exact )?solution (?:found|exists)|unable to (?:find|solve|determine|finish|verify)|"
    r"my best (?:guess|answer|attempt) (?:is|would be)|best guess)\b", re.I)
ANSWER_TAG = re.compile(r'<answer>', re.I)
BLANKS = ('incomplete', 'reasoning_not_closed', 'empty', 'call_at_cap')  # the GameTerm user sees no answer


def is_giveup(content: str) -> bool:
    return bool(content.strip()) and not ANSWER_TAG.search(content) and bool(GIVEUP.search(content))


def graded_reward(task: dict, episode: dict, policy: dict) -> tuple[float, str]:
    """Correct > honest give-up > wrong > blank (TruthRL, arXiv:2509.25760; abstention reward per arXiv:2601.20126,
    smaller where items are easy, arXiv:2607.10738). `policy`: `correct`, `wrong`, `blank` values and `abstain`,
    a value per state category (the state id's first field). The reason is the outcome's, or `giveup`."""
    value, why = episode_reward(task, episode)
    if value == 1.0:
        return policy['correct'], why
    if why in BLANKS:
        return policy['blank'], why
    last = episode['turns'][-1]
    if task['rule'] != 'blocked' and last['call'] is None and is_giveup(last['content']):
        return policy['abstain'][episode['state'].split(':')[0]], 'giveup'
    return policy['wrong'], why


def repeated_positions(ids, n: int) -> np.ndarray:
    """Demystifying Long CoT (Yeo et al., arXiv:2502.03373, Algorithm 1): every position inside an n-gram that
    already occurred earlier in the sequence."""
    ids, hit, seen = list(ids), np.zeros(len(ids), bool), set()
    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i:i + n])
        if gram in seen:
            hit[i:i + n] = True
        seen.add(gram)
    return hit


def thinking_repetition(episode: dict, n: int, think_end: int | None) -> np.ndarray:
    """Per sampled token of the episode (turns concatenated, as `merged`): True where a thinking token sits in a
    repeated n-gram of its turn's thinking (tokens before the first `think_end`; all of a turn that never closed)."""
    parts = []
    for turn in episode['turns']:
        tokens = list(turn['tokens'])
        end = tokens.index(think_end) if think_end is not None and think_end in tokens else len(tokens)
        parts.append(np.concatenate([repeated_positions(tokens[:end], n), np.zeros(len(tokens) - end, bool)]))
    return np.concatenate(parts) if parts else np.zeros(0, bool)


def group_length_term(lengths, correct) -> np.ndarray:
    """Kimi k1.5's per-group length reward (Kimi Team, arXiv:2501.12599, Sec. 2.3.3): lambda = 0.5 - (len - min) /
    (max - min) over the group's sampled lengths; a correct episode gets lambda, any other min(0, lambda); 0 for a
    group of equal lengths. Added to the reward with a weight (run 3: `length_weight`, set by replay)."""
    lengths = np.asarray(lengths, dtype=np.float64)
    low, high = lengths.min(), lengths.max()
    if high == low:
        return np.zeros(len(lengths))
    lam = 0.5 - (lengths - low) / (high - low)
    return np.where(np.asarray(correct, bool), lam, np.minimum(0.0, lam))


def shaped_reward(task: dict, episode: dict, policy: dict) -> tuple[float, str]:
    """The training reward under a named policy. `outcome`: `episode_reward` as is (run 1). `soft_overlong`:
    outcome + DAPO's soft overlong penalty on the episode's longest sampled turn (every turn has the same cap; run 2).
    `graded`: `graded_reward` (run 3). Evaluation always scores `outcome`."""
    if policy['name'] == 'graded':
        return graded_reward(task, episode, policy)
    value, why = episode_reward(task, episode)
    if policy['name'] == 'outcome':
        return value, why
    if policy['name'] == 'soft_overlong':
        longest = max(len(turn['tokens']) for turn in episode['turns'])
        return value + soft_overlong_penalty(longest, policy['cap'], policy['buffer']), why
    raise ValueError(f"unknown reward policy {policy['name']}; one of {REWARDS}")


__all__ = ['Calculator', 'soft_overlong_penalty', 'group_length_term', 'shaped_reward', 'REWARDS', 'graded_reward', 'is_giveup', 'GIVEUP',
           'BLANKS', 'repeated_positions', 'thinking_repetition', 'REJECTIONS', 'parse_turn', 'arguments_json', 'call_message', 'result_message',
           'first_messages', 'later_messages', 'numeric_answer', 'verdict', 'episode_reward']
