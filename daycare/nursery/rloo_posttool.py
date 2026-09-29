"""Post-tool calculator episodes for the one-stack RLOO loop (research/rloo-posttool-calculator.md).

An episode starts at a GameTerm later-turn state (the first request, the model's own `calculate` call, the
real result in the wire envelope; `daycare/harness/posttool.py`) and samples the next assistant turn with
thinking on. When that turn calls `calculate` again, the call runs through GameTerm's calculator, the next
later-turn request is rendered exactly as GameTerm would send it, and the episode continues, up to `cap`
sampled turns of at most `limit` tokens each. The last turn is scored by the task's deterministic rule
(`posttool.episode_reward`: 1 for a correct, non-empty final answer; tool calls neither rewarded nor penalized).

Every sampled token of every turn trains (its sampler log probability and captured final-block input);
prompts, the rendered history and tool results are prefilled, never sampled, so they never enter the loss.

    python -m daycare.nursery.rloo_posttool sample   --root R --tasks T --envelope E       (own first-turn calls)
    python -m daycare.nursery.rloo_posttool sample   --root R --states S --envelope E      (stock pass@1 / pass@k)
    python -m daycare.nursery.rloo_posttool train    --root R --states S --envelope E [--updates N]
    python -m daycare.nursery.rloo_posttool verify   --run R --envelope E
    python -m daycare.nursery.rloo_posttool force    --root R --blanks B --states S --envelope E [--run A] --limit 256
    python -m daycare.nursery.rloo_posttool resume   --root R --blanks B --states S --envelope E [--run A] --limit 4096
"""
from __future__ import annotations

import os

# The trainer tinygrad (tinygrad-arkey exp) is selected by trainer_env; see rloo_tinygrad.
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from daycare.artifact.record_xml import read, write  # noqa: E402
from daycare.harness import posttool, posttool_blocked  # noqa: E402
from daycare.model.chat_format import SegmentEncoder  # noqa: E402
from . import rloo_tinygrad as one  # noqa: E402  (selects the trainer tinygrad first)
from .rl_triggers import limits_arg  # noqa: E402
from .rloo_tinygrad import CONFIG, MODEL  # noqa: E402
from .trainer_env import trainer_revision, trainer_root  # noqa: E402

CAP = 8  # sampled turns per episode (GameTerm allows 32 tool iterations; this bounds rollout cost)
SHARED_GRAIN = 512  # the flash-attention prefill block: a shared prefix is cut to whole blocks


class Renderer:
    """Token ids of a GameTerm request: llama.cpp's BOS + the chat template's rendering (thinking per envelope)."""

    def __init__(self, chat, tokenizer, envelope: dict, bos: int | None):
        self.chat, self.envelope, self.bos = chat, envelope, bos
        self.encode = SegmentEncoder(tokenizer)

    def messages(self, task: dict, exchanges: list[dict]) -> list[dict]:
        return posttool.later_messages(self.envelope, task['request'], exchanges)

    def ids(self, task: dict, exchanges: list[dict]) -> list[int]:
        text = self.chat.render(self.messages(task, exchanges), self.envelope['tools'], generation=True)
        return ([self.bos] if self.bos is not None else []) + self.encode(text)


class Episodes:
    """Multi-turn post-tool rollouts in one sampling call: when a turn ends in a `calculate` call, the call runs,
    the next request is rendered, and it joins the running batch at once (the sampler's `follow`), so a follow-up
    turn starts as soon as a lane is free instead of after the slowest turn of a round."""

    def __init__(self, loop, renderer: Renderer, calculator, cap: int = CAP, shared: int = 0, on_done=None):
        self.loop, self.renderer, self.calculator, self.cap, self.shared = loop, renderer, calculator, cap, shared
        self.on_done = on_done

    def run(self, states: list[dict], group: int) -> tuple[list[list[dict]], dict]:
        tools = self.renderer.envelope['tools']
        episodes = [[dict(state=s['id'], task=s['task'], exchanges=[dict(e) for e in s['exchanges']], turns=[],
                          base=len(s['exchanges']))
                     for _ in range(group)] for s in states]
        stats = dict(sample_s=0.0, prime_s=0.0, render_s=0.0, tool_s=0.0, parse_s=0.0, generated_tokens=0,
                     prompt_tokens=0, tool_calls=0, turns=0)
        clock = time.perf_counter()
        prompts = [self.renderer.ids(s['task'], s['exchanges']) for s in states]
        stats['render_s'] += time.perf_counter() - clock
        owners: list = [None] * len(states)  # request index -> its episode (first-turn requests: by finishing order)
        finished = [0] * len(states)
        added: list[list[int]] = []

        def follow(request: int, rollout: dict):
            if request < len(states):
                g, k = request, finished[request]
                finished[request] += 1
            else:
                g, k = owners[request]
            episode = episodes[g][k]
            clock = time.perf_counter()
            turn = posttool.parse_turn(rollout['text'], tools)
            stats['parse_s'] += time.perf_counter() - clock
            turn.update(stop=rollout['stop'], tokens=rollout['tokens'], logprobs=rollout['logprobs'], text=rollout['text'])
            if 'hidden' in rollout:
                turn['hidden'] = rollout['hidden']
            stats['generated_tokens'] += len(rollout['tokens'])
            stats['turns'] += 1
            episode['turns'].append(turn)
            call, blocked = turn['call'], episode['task']['rule'] == 'blocked'
            usable = call is not None and (posttool_blocked.continues(episode['task'], episode['exchanges'], call)
                                           if blocked else call['name'] == 'calculate')
            if not (rollout['stop'] == 'eos' and turn['closed'] and usable and len(episode['turns']) < self.cap):
                if self.on_done:
                    self.on_done(g, episode)
                return None
            clock = time.perf_counter()
            step = (posttool_blocked.exchange(episode['exchanges'], call['name'], call['arguments_json']) if blocked
                    else self.calculator.exchange(len(episode['exchanges']), call['name'], call['arguments_json']))
            stats['tool_s'] += time.perf_counter() - clock
            clock = time.perf_counter()
            ids = self.renderer.ids(episode['task'], episode['exchanges'] + [step])
            stats['render_s'] += time.perf_counter() - clock
            if len(ids) - self.shared > self.loop.sampler.prefix_capacity:
                episode['context_cap'] = True  # the history outgrew the prompt slots: the episode ends
                if self.on_done:
                    self.on_done(g, episode)
                return None
            episode['exchanges'].append(step)
            stats['tool_calls'] += 1
            owners.append((g, k))
            added.append(ids)
            return [(ids, 1)]

        prefill = self.loop.prefill
        reused = prefill.saved_tokens is not None and tuple(prompts[0][:len(prefill.saved_tokens)]) == prefill.saved_tokens
        clock = time.perf_counter()
        _, sampler = self.loop.sample(prompts, [group] * len(prompts), follow=follow)
        stats['sample_s'] = time.perf_counter() - clock
        shared = int(sampler.get('shared', 0))
        every = prompts + added
        stats.update(prime_s=sampler.get('prime_s', 0.0), decode_steps=int(sampler.get('steps', 0)),
                     lane_steps=int(sampler.get('lane_steps', 0)), active_lane_steps=int(sampler.get('active_lane_steps', 0)),
                     row_steps=int(sampler.get('row_steps', sampler.get('lane_steps', 0))),
                     lane_moves=int(sampler.get('lane_moves', 0)),
                     window_rows={str(k): int(v) for k, v in sorted(
                         {s: sampler.get('sizes', []).count(s) for s in set(sampler.get('sizes', []))}.items())},
                     window_rows_s={str(k): float(v) for k, v in sorted(sampler.get('size_s', {}).items())},
                     shared_tokens=shared, shared_primed=0 if reused else shared, prompt_tokens=sum(len(p) for p in every),
                     prefill_tokens=(0 if reused else shared) + sum(len(p) - 1 - shared for p in every))
        return episodes, stats


def merged(episode: dict) -> dict:
    """The training view of an episode: its sampled turns' tokens, sampler logprobs and captures, concatenated."""
    turns = episode['turns']
    out = dict(tokens=[t for turn in turns for t in turn['tokens']],
               logprobs=[v for turn in turns for v in turn['logprobs']],
               stop=turns[-1]['stop'], text=turns[-1]['text'])
    if all('hidden' in turn for turn in turns):
        out['hidden'] = np.concatenate([turn['hidden'] for turn in turns]).astype(np.float32)
    return out


def summary(episode: dict, reward: float, why: str, *, text: bool = True) -> dict:
    """What a person needs to check a reward by hand (no hidden states)."""
    return dict(state=episode['state'], reward=reward, reason=why, turns=len(episode['turns']),
                calls=[dict(arguments=e['arguments_json'], outcome=e['outcome'], content=e['content'])
                       for e in episode['exchanges']],
                tokens=[len(t['tokens']) for t in episode['turns']], stops=[t['stop'] for t in episode['turns']],
                final=episode['turns'][-1]['content'] if text else None,
                final_call=episode['turns'][-1]['call'],
                reasoning_tail=episode['turns'][-1]['reasoning'][-600:] if text else None,
                answer=episode['task'].get('answer'), target=episode['task'].get('target'),
                empty_after_tool=not episode['turns'][-1]['content'].strip() and episode['turns'][-1]['call'] is None)


def rates(episodes, marks, limit: int) -> dict:
    """Reported, not rewarded differently: episodes ending in a call to another tool (scored 0; GameTerm would run
    it), sampled turns that hit the token cap, and final turns with no answer and no call."""
    flat = [(e, why) for g, m in zip(episodes, marks) for e, (_, why) in zip(g, m)]
    turns = [t for e, _ in flat for t in e['turns']]
    return dict(other_tool_rate=sum(why == 'other_tool' for _, why in flat) / len(flat),
                capped_turn_rate=sum(len(t['tokens']) >= limit for t in turns) / len(turns),
                empty_after_tool_rate=sum(not e['turns'][-1]['content'].strip() and e['turns'][-1]['call'] is None
                                          for e, _ in flat) / len(flat))


def score(episodes: list[list[dict]]) -> list[list[tuple[float, str]]]:
    return [[posttool.episode_reward(e['task'], e) for e in group] for group in episodes]


class Progress:
    """Streams `sample`'s per-state exam progress while episodes are still running: as `Episodes.run`'s `on_done`,
    counts each state's finished episodes, and once all `k` of a state are in, scores them (`posttool.episode_reward`)
    and appends one partial JSON line to `path`, then logs a small running summary per category."""

    def __init__(self, path: Path, chosen: list[dict], k: int, log=print):
        self.chosen, self.k, self.log = chosen, k, log
        self.file = open(path, 'w')
        self.pending: list[list[dict]] = [[] for _ in chosen]
        self.states_done, self.states_total = 0, len(chosen)
        self.cats: dict[str, dict] = {}
        for state in chosen:
            self.cats.setdefault(state['category'], dict(done=0, total=0, reward_sum=0.0, blanks=0))['total'] += 1
        self.clock = time.perf_counter()

    def done(self, g: int, episode: dict) -> None:
        group = self.pending[g]
        group.append(episode)
        if len(group) < self.k:
            return
        state = self.chosen[g]
        marks = [posttool.episode_reward(state['task'], e) for e in group]
        rewards, reasons = [r for r, _ in marks], [w for _, w in marks]
        blank = [not e['turns'][-1]['content'].strip() and e['turns'][-1]['call'] is None for e in group]
        blank_reasons = [w for w, b in zip(reasons, blank) if b]
        self.states_done += 1
        row = dict(partial=True, state=state['id'], category=state['category'], split=state['split'],
                   rewards=rewards, reasons=reasons, blanks=sum(blank), blank_reasons=blank_reasons,
                   states_done=self.states_done, states_total=self.states_total,
                   wall_s=time.perf_counter() - self.clock)
        self.file.write(json.dumps(row) + '\n')
        self.file.flush()
        cell = self.cats[state['category']]
        cell['done'] += 1
        cell['reward_sum'] += float(np.mean(rewards))
        cell['blanks'] += sum(blank)
        parts = [f"{name} {cell['done']}/{cell['total']} pass1 {cell['reward_sum'] / cell['done']:.3f} "
                f"blanks {cell['blanks']}" for name, cell in self.cats.items() if cell['done']]
        self.log(f"PARTIAL {self.states_done}/{self.states_total} states | " + ' | '.join(parts))


def exam_record(chosen: list[dict], episodes: list[list[dict]], stats: dict, seconds: float, **meta) -> dict:
    """The episode-to-record part of `sample`: per-state rows and per-cell (category/split) means, the exam-wide
    rates, wrapped in the `sample.xml` schema; `meta` supplies every field that does not depend on `episodes`."""
    limit = meta['limit']
    record_rates = rates(episodes, score(episodes), limit)
    rows, cells = [], {}
    for state, group, marks in zip(chosen, episodes, score(episodes)):
        rows.append(dict(state=state['id'], category=state['category'], split=state['split'],
                         rewards=[r for r, _ in marks], episodes=[summary(e, r, why) for e, (r, why) in zip(group, marks)]))
        cell = cells.setdefault(f"{state['category']}/{state['split']}", dict(states=0, pass1=0.0, passk=0.0, empty=0))
        cell['states'] += 1
        cell['pass1'] += float(np.mean(rows[-1]['rewards']))
        cell['passk'] += float(max(rows[-1]['rewards']) > 0)
        cell['empty'] += sum(e['empty_after_tool'] for e in rows[-1]['episodes'])
    for cell in cells.values():
        cell['pass1'], cell['passk'] = cell['pass1'] / cell['states'], cell['passk'] / cell['states']
    return dict(schema='daycare.posttool_sample.v1', k=meta['k'], cap=meta['cap'], seed=meta['seed'],
                split=meta['split'], **record_rates, categories=meta['categories'], adapter=meta['adapter'],
                limit=limit, load_s=meta['load_s'], seconds=seconds, stats=stats, cells=cells, rows=rows,
                source=meta['source'], source_sha256=meta['source_sha256'], tinygrad_revision=meta['tinygrad_revision'],
                envelope=meta.get('envelope', ''), envelope_sha256=meta.get('envelope_sha256', ''))


# ------------------------------------------------------------------------------------------------ setup

def setup(args, cfg: dict):
    """Envelope, model, tokenizer, renderer, logit bias (as rloo_tinygrad.setup, plus BOS and the cached encoder)."""
    from daycare.artifact.record_xml import read
    from daycare.model.chat_format import NativeToolChat
    envelope = read(args.envelope)
    if not envelope['chat_template_kwargs']['enable_thinking']:
        raise ValueError('post-tool episodes run with thinking on (the deployment default)')
    clock = time.perf_counter()
    model, tokenizer, metadata = one.load_model(args.model, cfg['slot_ctx'])
    load_s = time.perf_counter() - clock
    chat = NativeToolChat(metadata['tokenizer.chat_template'], tokenizer, thinking=True)
    bias = np.zeros(model.config.vocab_size, dtype=np.float32)
    for key, value in envelope.get('logit_bias', {}).items():
        bias[int(key)] = float(value)
    bos = tokenizer.bos_id if metadata.get('tokenizer.ggml.add_bos_token') else None
    return envelope, model, tokenizer, Renderer(chat, tokenizer, envelope, bos), bias, load_s


def build_loop(model, cfg, bias, stop, renderer, states, decode, *, capture: bool):
    """A loop whose prefill primes the shared envelope once and keeps it; one prompt slot per lane for follow-up turns."""
    first = [renderer.ids(s['task'], s['exchanges']) for s in states]
    common = one._common(first) if len(first) > 1 else len(first[0]) - 2
    shared = max(SHARED_GRAIN, (common // SHARED_GRAIN) * SHARED_GRAIN)
    own = max(len(p) for p in first) - shared
    prefix = cfg.get('prefix_capacity', 2048)
    if own > prefix:
        raise ValueError(f'a state needs {own} own prompt tokens, over the prefix capacity {prefix}')
    lanes = cfg['lanes']
    return one.Loop(model, cfg, bias, stop, prefix_capacity=prefix, decode=decode, lanes=lanes, capture=capture,
                    shared_capacity=shared, prompts=max(2 * lanes // cfg['group'] + 2, lanes)), shared


def load_states(path: Path, split: str | None = None, categories=None) -> list[dict]:
    states = read(path)['states']
    return [s for s in states if (split is None or s['split'] == split)
            and (categories is None or s['category'] in categories)]


# ----------------------------------------------------------------------------------------------- actions

def sample(args):
    """Episodes without training (capture off): the stock first turn on every miss/relay task (`--tasks`, one
    sample, one turn: the model's own calls, `posttool_tasks.build_states`), or pass@1 / pass@k on a fixed
    sample of `--per-cell` states per category and split (`--states`), or every state of `--split` in
    `--categories` (an evaluation arm; `--run` loads a trained adapter, else the zero adapter = stock)."""
    rng = random.Random(args.seed)
    if args.tasks:
        chosen = [dict(id=t['id'], category=t['category'], split=t['split'], task=t, exchanges=[])
                  for t in read(args.tasks)['tasks'] if t['category'] in ('miss', 'relay', 'blocked')]
        k, cap = 1, 1
    elif args.split:  # every state of a split's categories (the G1 / G1b evaluation arms)
        chosen, k, cap = load_states(args.states, args.split, args.categories), args.k, args.cap
    else:
        chosen, k, cap = [], args.k, args.cap
        for split in ('train', 'heldout'):
            for category in ('miss', 'relay', 'repair', 'empty'):
                pool = load_states(args.states, split, [category])
                rng.shuffle(pool)
                chosen += pool[:args.per_cell]
    chosen = chosen[:args.first] if args.first else chosen
    cfg = dict(CONFIG, group=k, limit=args.limit, lanes=args.lanes, slot_ctx=20480, prefix_capacity=2048,
               compact=args.compact)
    envelope, model, tokenizer, renderer, bias, load_s = setup(args, cfg)
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    loop, shared = build_loop(model, cfg, bias, stop, renderer, chosen, tokenizer.decode, capture=False)
    if args.run:
        one.load_adapter(loop.adapters, args.run / 'adapter.xml', read(args.run / 'run.xml'))
    one.Tensor.manual_seed(args.seed)
    progress = Progress(args.root / 'progress.jsonl', chosen, k) if args.progress else None
    with posttool.Calculator(args.runner) as calculator:
        clock = time.perf_counter()
        episodes, stats = Episodes(loop, renderer, calculator, cap, shared,
                                   on_done=progress.done if progress else None).run(chosen, k)
        seconds = time.perf_counter() - clock
    if progress:
        progress.file.close()
    record = exam_record(chosen, episodes, stats, seconds, k=k, cap=cap, seed=args.seed, split=args.split,
                         categories=args.categories,
                         adapter=str(args.run / 'adapter.xml') if args.run else 'zero (stock)',
                         limit=args.limit, load_s=load_s, source=str(args.tasks or args.states),
                         source_sha256=hashlib.sha256((args.tasks or args.states).read_bytes()).hexdigest(),
                         tinygrad_revision=trainer_revision(),
                         envelope=str(args.envelope), envelope_sha256=hashlib.sha256(args.envelope.read_bytes()).hexdigest())
    write(args.root / 'sample.xml', record, root='sample')
    if args.keep_blanks:  # the final turn's tokens of every episode the user sees no answer from (for `force`)
        blanks = [dict(state=state['id'], category=state['category'], episode=k, reason=why, stop=e['turns'][-1]['stop'],
                       closed=e['turns'][-1]['closed'], exchanges=e['exchanges'], tokens=e['turns'][-1]['tokens'])
                  for state, group, marks in zip(chosen, episodes, score(episodes))
                  for k, (e, (_, why)) in enumerate(zip(group, marks)) if why in posttool.BLANKS]
        (args.root / 'blanks.json').write_text(json.dumps(dict(seed=args.seed, limit=args.limit, blanks=blanks)))
    print(json.dumps(dict(record['cells'], rates=dict(
        other_tool_rate=record['other_tool_rate'], capped_turn_rate=record['capped_turn_rate'],
        empty_after_tool_rate=record['empty_after_tool_rate'])), indent=1), f'{seconds:.0f} s', flush=True)


def forced_prompt(renderer, task: dict, blank: dict, stop: set[int], think_end: int) -> list[int] | None:
    """Budget forcing (s1, arXiv:2501.19393): the blank turn's exact request + its sampled tokens (a trailing end
    token removed) + the end-of-thinking token. None when the turn already closed its thinking (not forceable)."""
    if blank['closed']:
        return None
    body = blank['tokens'][:-1] if blank['tokens'] and blank['tokens'][-1] in stop else blank['tokens']
    return renderer.ids(task, blank['exchanges']) + list(body) + [think_end]


def force(args):
    """Resume every blank of a `sample --keep-blanks` run with `</think>` appended and a small answer budget
    (`--limit`); score the forced turn with the exam's rule (research/posttool-budget-forcing.md)."""
    source = json.loads(args.blanks.read_text())
    states = {s['id']: s for s in load_states(args.states)}
    cfg = dict(CONFIG, group=1, limit=args.limit, lanes=args.lanes, slot_ctx=20480)
    envelope, model, tokenizer, renderer, bias, load_s = setup(args, cfg)
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    think_end = think_end_id(tokenizer)
    if think_end is None or tokenizer.decode([think_end]) != '</think>':
        raise ValueError(f'</think> is not one token: {tokenizer.encode("</think>")}')
    jobs = [(b, forced_prompt(renderer, states[b['state']]['task'], b, stop, think_end)) for b in source['blanks']]
    todo = [(b, p) for b, p in jobs if p is not None]
    shared = max(SHARED_GRAIN, (one._common([p for _, p in todo]) // SHARED_GRAIN) * SHARED_GRAIN)
    own = max(len(p) for _, p in todo) - shared
    loop = one.Loop(model, dict(cfg, prefix_capacity=own), bias, stop, prefix_capacity=own, decode=tokenizer.decode,
                    lanes=args.lanes, capture=False, shared_capacity=shared, prompts=args.lanes)
    if args.run:
        one.load_adapter(loop.adapters, args.run / 'adapter.xml', read(args.run / 'run.xml'))
    one.Tensor.manual_seed(args.seed)
    tools = envelope['tools']
    clock = time.perf_counter()
    results, _ = loop.sample([p for _, p in todo], [1] * len(todo))
    seconds = time.perf_counter() - clock
    rows = []
    for (blank, prompt), (rollout,) in zip(todo, results):
        body = blank['tokens'][:-1] if blank['tokens'] and blank['tokens'][-1] in stop else blank['tokens']
        answer = rollout['tokens'][:-1] if rollout['stop'] == 'eos' else rollout['tokens']
        turn = posttool.parse_turn(tokenizer.decode(list(body) + [think_end] + answer), tools)
        turn['stop'] = rollout['stop']
        task = states[blank['state']]['task']
        reward, why = posttool.episode_reward(task, dict(turns=[turn]))
        rows.append(dict(state=blank['state'], category=blank['category'], episode=blank['episode'],
                         before=blank['reason'], thinking_tokens=len(body), answer_tokens=len(rollout['tokens']),
                         stop=rollout['stop'], reward=reward, reason=why, content=turn['content'], call=turn['call'],
                         answer=task.get('answer'), target=task.get('target'), numbers=task.get('numbers'),
                         reasoning_tail=turn['reasoning'][-400:]))
    skipped = [dict(state=b['state'], category=b['category'], episode=b['episode'], reason=b['reason'])
               for b, p in jobs if p is None]
    record = dict(schema='daycare.posttool_force.v1', blanks=str(args.blanks), seed=args.seed, limit=args.limit,
                  think_end=think_end, adapter=str(args.run / 'adapter.xml') if args.run else 'zero (stock)',
                  seconds=seconds, rows=rows, not_forceable=skipped,
                  tinygrad_revision=trainer_revision())
    write(args.root / 'force.xml', record, root='force')
    print(f'{len(rows)} forced, {sum(r["reward"] for r in rows):.0f} correct, {len(skipped)} not forceable, '
          f'{seconds:.0f} s', flush=True)



def resume(args):
    """A longer turn instead of a cut one (research/posttool-longer-turn.md): every blank of a `sample --keep-blanks`
    run that stopped at the token cap resumes from its exact request + its sampled tokens, with NO `</think>` added,
    for `--limit` more tokens (the turn's total budget = cap + limit), same sampler and temperature. A turn that then
    ends in a `calculate` call runs it through GameTerm's calculator and the episode continues as at the exam (turn
    cap `--cap`, each follow-up turn `--limit` tokens). The last turn is scored by the exam's rule."""
    source = json.loads(args.blanks.read_text())
    states = {s['id']: s for s in load_states(args.states)}
    cfg = dict(CONFIG, group=1, limit=args.limit, lanes=args.lanes, slot_ctx=20480)
    envelope, model, tokenizer, renderer, bias, load_s = setup(args, cfg)
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    tools = envelope['tools']
    todo = [b for b in source['blanks'] if b['stop'] == 'limit']
    if any(len(b['tokens']) != source['limit'] for b in todo):
        raise ValueError('a cut turn is not exactly the cap long')
    if not todo:
        print('no blank stopped at the cap; nothing to resume', flush=True)
        return
    prompts = [renderer.ids(states[b['state']]['task'], b['exchanges']) + list(b['tokens']) for b in todo]
    shared = max(SHARED_GRAIN, (one._common(prompts) // SHARED_GRAIN) * SHARED_GRAIN)
    own = max(len(p) for p in prompts) - shared
    slots = 2 * args.lanes + 2
    loop = one.Loop(model, dict(cfg, prefix_capacity=own), bias, stop, prefix_capacity=own, decode=tokenizer.decode,
                    lanes=args.lanes, capture=False, shared_capacity=shared, prompts=slots)
    if args.run:
        one.load_adapter(loop.adapters, args.run / 'adapter.xml', read(args.run / 'run.xml'))
    one.Tensor.manual_seed(args.seed)
    episodes = [dict(blank=b, task=states[b['state']]['task'], exchanges=[dict(e) for e in b['exchanges']], turns=[])
                for b in todo]
    owners: list[int] = []
    with posttool.Calculator(args.runner) as calculator:
        def follow(request: int, rollout: dict):
            i = request if request < len(todo) else owners[request - len(todo)]
            episode = episodes[i]
            ids = list(rollout['tokens'])
            if not episode['turns']:  # the resumed turn: its cut tokens + the continuation
                ids = list(episode['blank']['tokens']) + ids
                episode['extra'] = len(rollout['tokens'])
            turn = posttool.parse_turn(tokenizer.decode(ids[:-1] if rollout['stop'] == 'eos' else ids), tools)
            turn.update(stop=rollout['stop'], length=len(ids), think_end=ids.index(THINK_END) if THINK_END in ids else None,
                        tail=tokenizer.decode(ids[-200:]))
            episode['turns'].append(turn)
            call = turn['call']
            prefilled = len(states[episode['blank']['state']]['exchanges'])  # the state's own calls are not sampled turns
            turns_so_far = len(episode['blank']['exchanges']) - prefilled + len(episode['turns'])
            if not (rollout['stop'] == 'eos' and turn['closed'] and call is not None and call['name'] == 'calculate'
                    and turns_so_far < args.cap):
                return None
            step = calculator.exchange(len(episode['exchanges']), call['name'], call['arguments_json'])
            nxt = renderer.ids(episode['task'], episode['exchanges'] + [step])
            if len(nxt) - shared > own:
                episode['context_cap'] = True
                return None
            episode['exchanges'].append(step)
            owners.append(i)
            return [(nxt, 1)]

        clock = time.perf_counter()
        _, stats = loop.sample(prompts, [1] * len(prompts), follow=follow)
        seconds = time.perf_counter() - clock
    rows = []
    for e in episodes:
        b, first = e['blank'], e['turns'][0]
        reward, why = posttool.episode_reward(e['task'], e)
        rows.append(dict(state=b['state'], category=b['category'], episode=b['episode'], before=b['reason'],
                         closed_at_cap=b['closed'], extra_tokens=e['extra'], first_stop=first['stop'],
                         first_length=first['length'], think_end=first['think_end'], turns=len(e['turns']),
                         extra_calls=len(e['exchanges']) - len(b['exchanges']), lengths=[t['length'] for t in e['turns']],
                         reward=reward, reason=why, content=e['turns'][-1]['content'], call=e['turns'][-1]['call'],
                         answer=e['task'].get('answer'), target=e['task'].get('target'),
                         numbers=e['task'].get('numbers'), first_tail=first['tail'],
                         empty_after_tool=not e['turns'][-1]['content'].strip() and e['turns'][-1]['call'] is None))
    record = dict(schema='daycare.posttool_resume.v1', blanks=str(args.blanks), seed=args.seed, cap_before=source['limit'],
                  limit=args.limit, cap=args.cap, adapter=str(args.run / 'adapter.xml') if args.run else 'zero (stock)',
                  seconds=seconds, load_s=load_s, rows=rows, stats={k: v for k, v in stats.items() if isinstance(v, (int, float))},
                  tinygrad_revision=trainer_revision())
    write(args.root / 'resume.xml', record, root='resume')
    print(f'{len(rows)} resumed, {sum(r["first_stop"] == "eos" for r in rows)} finished, '
          f'{sum(r["reward"] for r in rows):.0f} correct, {seconds:.0f} s', flush=True)

MASKS = ('none', 'overlong_filter')


def masked(sampled: list[list[dict]], marks: list[list[tuple]], policy: str):
    """The loss-mask policy. `none` (run 1): every episode trains. `overlong_filter`: an episode whose last turn hit
    the token cap is dropped from the update, loss and leave-one-out baseline alike (DAPO's Overlong Filtering, Yu et
    al., arXiv:2503.14476, Sec. 3.4; SimpleTIR, arXiv:2509.02479, excludes a whole trajectory with a void turn the
    same way). It still counts in the rates, which the stepper reports from all episodes. A group left with fewer
    than two episodes carries no leave-one-out signal and is dropped; with none left the update is skipped (learn
    records it, rl_triggers stops the run after `skips_max` in a row)."""
    if policy == 'none':
        return sampled, marks, list(range(len(sampled)))
    if policy != 'overlong_filter':
        raise ValueError(f'unknown loss-mask policy {policy}; one of {MASKS}')
    groups, kept, where = [], [], []
    for index, (rollouts, scores) in enumerate(zip(sampled, marks)):
        pairs = [(r, m) for r, m in zip(rollouts, scores) if r['stop'] != 'limit']
        if len(pairs) >= 2:
            groups.append([r for r, _ in pairs])
            kept.append([m for _, m in pairs])
            where.append(index)
    return groups, kept, where


def stepper(loop, driver, states, group, reward: dict | None = None, mask: str = 'none'):
    """rloo_tinygrad.run's `step`: sample the picked states' episodes, score them, one RLOO step; returns learn's
    result with the per-stage times, the episode statistics and a hand-checkable summary of every episode.
    `reward` and `mask` are the named reward-shaping and loss-mask policies (run 1: outcome, none). The policy
    gradient already sums over tokens with no per-sample 1/|o| length division (rloo_tinygrad.TailSteps._grad),
    the normalization Dr. GRPO (Liu et al., arXiv:2503.20783) removes from GRPO."""
    reward = reward or dict(name='outcome')

    def step(picks):
        clock = time.perf_counter()
        chosen = [states[i] for i in picks]
        episodes, stats = driver.run(chosen, group)
        times = dict(sample_s=stats['sample_s'], sample_prefill_s=stats['prime_s'], render_s=stats['render_s'],
                     tool_s=stats['tool_s'], parse_s=stats['parse_s'])
        mark = time.perf_counter()
        marks = score(episodes)  # the outcome: rates and summaries report it whatever the training reward
        shaped = [[posttool.shaped_reward(e['task'], e, reward) for e in g] for g in episodes]
        if reward.get('length_weight'):  # per-group length term (Kimi k1.5, arXiv:2501.12599 Sec. 2.3.3; run 3)
            shaped = [[(v + reward['length_weight'] * t, why) for (v, why), t in zip(s, posttool.group_length_term(
                [sum(len(turn['tokens']) for turn in e['turns']) for e in g], [o == 1.0 for o, _ in m]))]
                for s, g, m in zip(shaped, episodes, marks)]
        views = [[merged(e) for e in g] for g in episodes]
        repeated = 0
        if reward.get('repetition'):  # dense per-token penalty on repeated thinking n-grams (arXiv:2502.03373)
            rep = reward['repetition']
            for episodes_g, views_g in zip(episodes, views):
                for e, view in zip(episodes_g, views_g):
                    hit = posttool.thinking_repetition(e, rep['n'], rep.get('think_end'))
                    view['token_bonus'] = np.where(hit, np.float32(rep['penalty']), np.float32(0)).astype(np.float32)
                    repeated += int(hit.sum())
        times['score_s'] = time.perf_counter() - mark
        sampled, kept, where = masked(views, shaped, mask)
        result = loop.learn(sampled, kept, times, dict(stats))
        for labelled, index in zip(result['groups'], where):
            labelled['task'] = chosen[index]['id']
        # the log's reward is the outcome over every episode; learn's is the trained (shaped, kept) reward
        result['trained_reward'], result['mean_reward'] = result['mean_reward'], float(np.mean(
            [v for g in marks for v, _ in g]))
        decode_s = max(stats['sample_s'] - stats['prime_s'], 1e-9)
        result['episode_stats'] = dict(stats, decode_s=decode_s, decode_tok_s=stats['generated_tokens'] / decode_s,
                                       ms_per_step=1e3 * decode_s / max(stats['decode_steps'], 1),
                                       lane_utilization=stats['active_lane_steps'] / max(stats['lane_steps'], 1),
                                       row_utilization=stats['active_lane_steps'] / max(stats['row_steps'], 1),
                                       episodes=sum(len(g) for g in episodes))
        result['episodes'] = [[summary(e, r, why) for e, (r, why) in zip(g, m)] for g, m in zip(episodes, marks)]
        # the length push of the sequence term (research/rloo-posttool-audit-20260927.md): sum of advantage x tokens
        a_len = sum(float(np.dot(one.leave_one_out([v for v, _ in m]), [len(r['tokens']) for r in g]))
                    for g, m in zip(sampled, kept) if len(g) >= 2)
        result['metrics'].update(rates(episodes, marks, loop.cfg['limit']), trained_episodes=sum(len(g) for g in kept),
                                 shaped_reward=float(np.mean([v for g in shaped for v, _ in g])),
                                 giveup_rate=float(np.mean([why == 'giveup' for g in shaped for _, why in g])),
                                 advantage_length_sum=a_len, repeated_thinking_tokens=repeated)
        result['states'] = [s['id'] for s in chosen]
        result['times']['driver_s'] = time.perf_counter() - clock
        return result

    return step


def reward_policy(args, think_end: int | None = None) -> dict:
    if args.reward == 'soft_overlong':
        return dict(name='soft_overlong', cap=args.limit, buffer=args.buffer)
    if args.reward == 'graded':  # run 3 (research/rloo-posttool-calculator-r3.md)
        policy = dict(name='graded', correct=1.0, wrong=args.wrong, blank=args.blank,
                      abstain=dict(dict.fromkeys(('relay', 'repair', 'miss', 'empty'), 0.0), **args.abstain))
        ordered = policy['correct'] > max(policy['abstain'].values()) and \
            min(policy['abstain'].values()) > policy['wrong'] > policy['blank']
        if not ordered:
            raise ValueError(f'graded reward must order correct > give-up > wrong > blank: {policy}')
        if args.repetition:
            policy['repetition'] = dict(args.repetition, think_end=think_end)
        if args.length_weight:
            policy['length_weight'] = args.length_weight
        return policy
    return dict(name=args.reward)


THINK_END = 13  # Nemotron 3 Nano's `</think>` (one token)


def think_end_id(tokenizer) -> int:
    """The `</think>` token id, where the repetition penalty's thinking span ends. Refuses anything but the one
    token 13: a silent None would make every turn "all thinking" and penalize the answer's tokens too."""
    ids = list(tokenizer.encode('</think>'))
    if ids != [THINK_END]:
        raise ValueError(f"'</think>' encodes to {ids}, expected [{THINK_END}] (Nemotron 3 Nano)")
    return THINK_END


def train(args):
    """RLOO on the training split's post-tool states; per update: episodes -> scores -> one step (rloo_tinygrad.run)."""
    from .provenance import file_sha256, source_revision
    cfg = dict(CONFIG, updates=args.updates, prompts_per_update=args.prompts_per_update, group=args.group,
               limit=args.limit, lanes=args.lanes or args.group * args.prompts_per_update, bucket=512, tis=True,
               tis_cap=2.0, logprob_tolerance=CONFIG['logprob_tolerance'], slot_ctx=20480, prefix_capacity=2048,
               cap=args.cap, lr=args.lr, protocol=args.protocol, categories=args.categories, mix=args.mix,
               compact=args.compact, seed=args.train_seed or CONFIG['seed'],
               reward=reward_policy(args), mask=args.mask, kl_aggregation=args.kl_aggregation, kl_beta=args.kl_beta,
               triggers=None if args.triggers == 'off' else dict(CONFIG['triggers'], **args.triggers))
    from .posttool_tasks import training_mix
    states = training_mix(load_states(args.states, 'train', args.categories), args.mix)
    random.Random(cfg['seed']).shuffle(states)
    envelope, model, tokenizer, renderer, bias, load_s = setup(args, cfg)
    cfg['reward'] = reward_policy(args, think_end_id(tokenizer))
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    clock = time.perf_counter()
    loop, shared = build_loop(model, cfg, bias, stop, renderer, states, tokenizer.decode, capture=True)
    setup_s = time.perf_counter() - clock
    record = dict(schema='daycare.rloo_tinygrad.v1', episode_source='posttool', complete=False,
                  daycare_revision=source_revision(), runner_sha256=file_sha256(Path(__file__)),
                  model_sha256=file_sha256(args.model), envelope_sha256=file_sha256(args.envelope),
                  states_sha256=file_sha256(args.states), tinygrad=trainer_root(),
                  tinygrad_revision=trainer_revision(),
                  model_profile=dict(architecture='nemotron_h', precision='bf16'), rank=cfg['rank'], alpha=cfg['alpha'],
                  last_k=1, target_map=one.target_map(loop.adapters), lr=cfg['lr'], seed=cfg['seed'],
                  examples=len(states), shared_prefix=shared, training_objective='rloo-tinygrad-posttool',
                  config=cfg, counted=args.protocol.startswith('rloo-posttool-calculator'), load_s=load_s, setup_s=setup_s, losses=[])
    print(f'load {load_s:.1f}s setup {setup_s:.1f}s states {len(states)} shared {shared}', flush=True)
    calculator = posttool.Calculator(args.runner)
    driver = Episodes(loop, renderer, calculator, args.cap, shared)

    try:
        one.run(loop, states, None, cfg, record, args.root, None, export=args.export,
                step=stepper(loop, driver, states, cfg['group'], cfg['reward'], cfg['mask']), keep_every=args.keep_every)
    finally:
        calculator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('sample', 'train', 'verify', 'force', 'resume'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--blanks', type=Path, help='force: a `sample --keep-blanks` blanks.json')
    parser.add_argument('--keep-blanks', action='store_true', help="sample: write blank episodes' final-turn tokens")
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='sample: skip streaming progress.jsonl / PARTIAL log (on by default)')
    parser.add_argument('--states', type=Path)
    parser.add_argument('--envelope', type=Path)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--runner', type=Path, default=posttool.RUNNER)
    parser.add_argument('--limit', type=int, default=CONFIG['limit'])
    parser.add_argument('--lanes', type=int, default=32)
    parser.add_argument('--compact', type=lambda text: [] if text == 'off' else sorted(int(v) for v in text.split(',')),
                        default=[], help='sample/train: step row counts below --lanes the sampler shrinks to as '
                        'episodes finish, e.g. 8,16 (research/rloo-slot-refill.md; default off)')
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--train-seed', type=int, help='train: batch order and sampling seed (default CONFIG seed '
                        '20260924, runs 1-4); the mix subsample keeps posttool_tasks.SEED')
    parser.add_argument('--first', type=int, help='harvest: first N tasks only (output-path check)')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--per-cell', type=int, default=12)
    parser.add_argument('--split', help='sample: every state of this split (with --categories) instead of --per-cell')
    parser.add_argument('--cap', type=int, default=CAP)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--prompts-per-update', type=int, default=4)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--categories', nargs='+', default=['miss', 'relay', 'repair', 'empty'])
    parser.add_argument('--mix', type=lambda text: {k: float(v) for k, v in (p.split('=') for p in text.split(','))},
                        help='train: category shares, e.g. repair=0.9,relay=0.1 (default: every train state)')
    parser.add_argument('--reward', choices=posttool.REWARDS, default='outcome', help='train: reward-shaping policy')
    parser.add_argument('--buffer', type=int, default=819, help='train: soft_overlong buffer before the cap (tokens)')
    parser.add_argument('--wrong', type=float, default=-1.0, help='train: graded reward of a wrong or unreadable answer')
    parser.add_argument('--blank', type=float, default=-1.5, help='train: graded reward of a blank (cap, unclosed, empty)')
    parser.add_argument('--abstain', type=lambda text: {k: float(v) for k, v in (p.split('=') for p in text.split(','))},
                        default={}, help='train: graded give-up reward per category, e.g. relay=0,repair=0.25')
    parser.add_argument('--repetition', type=lambda text: None if text == 'off' else
                        {k: (int(v) if k == 'n' else float(v)) for k, v in (p.split('=') for p in text.split(','))},
                        default=None, help='train: thinking n-gram repetition penalty, e.g. n=40,penalty=-0.05, or off')
    parser.add_argument('--length-weight', type=float, default=0.0,
                        help='train (graded): weight of the per-group Kimi k1.5 length term (run 3: 1.0; 0 = off)')
    parser.add_argument('--mask', choices=MASKS, default='none', help='train: loss-mask policy')
    parser.add_argument('--kl-beta', type=float, default=CONFIG['kl_beta'], help='train: KL coefficient (runs 1-2: 1e-3)')
    parser.add_argument('--kl-aggregation', choices=one.KL_AGGREGATIONS, default=CONFIG['kl_aggregation'],
                        help='train: KL/entropy token aggregation (runs 1-2: legacy_token_mean)')
    parser.add_argument('--triggers', type=lambda t: t if t == 'off' else limits_arg(t), default={},
                        help='train: stop-trigger overrides, e.g. kl_max=0.02,window=10 (rl_triggers.DEFAULTS), or off')
    parser.add_argument('--protocol', default='posttool-loop-iteration-speed.md')
    parser.add_argument('--no-export', dest='export', action='store_false')
    parser.add_argument('--keep-every', type=int, default=10, help='train: raw adapter checkpoint every N updates')
    args = parser.parse_args()
    if args.model is None:
        parser.error('pass --model or set DAYCARE_BASE_GGUF (Nemotron 3 Nano 4B, BF16 GGUF; docs/rl-training.md)')
    if args.root is not None:
        args.root.mkdir(parents=True, exist_ok=True)
    {'sample': sample, 'train': train, 'verify': one.verify, 'force': force, 'resume': resume}[args.action](args)


if __name__ == '__main__':
    main()
