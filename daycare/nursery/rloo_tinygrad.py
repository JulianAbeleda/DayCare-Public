"""RLOO on one stack: tinygrad-arkey samples and trains Nemotron-H, no llama.cpp.

The `exp` tree of tinygrad-arkey carries the Nemotron-H model, a chunked
TinyJit prompt prefill and a continuously batched rollout sampler that returns
each sampled token's log probability. This loop selects that tree as DayCare's
trainer tinygrad (`trainer_env`), puts LoRA on the model's final MLP block
before any graph is captured, and then per update:

    prefill + sample   group rollouts per prompt (sampler logprobs = old logprobs)
    score              the Countdown verifier (`reward`)
    features           the hidden state entering the final block at every sampled
                       token, captured by the sampler (the frozen blocks are then
                       identical by construction)
    update             recompute every sampled token through the tail on the decode
                       step's row count and ops (`Tail`), refuse past 0.1 nats from the
                       sampler (hard stop), sequence RLOO with truncated importance
                       weights min(exp(logp_train - logp_sampler), 2) per token
                       (verl's rollout_is_threshold), KL + entropy, one Adam step in place

The sampler's and prefill's compiled graphs read the adapter's buffers, and
Adam writes those buffers in place, so the next group samples from the updated
adapter with no export or reload. The Countdown RLOO notes (not published) own the
settings and predeclare this variant.

    python -m daycare.nursery.rloo_tinygrad train --root R --envelope E --prep P
"""
from __future__ import annotations

import os

# The trainer tinygrad is tinygrad-arkey's exp tree, selected through the trainer_env
# boundary (DAYCARE_TRAIN_TINYGRAD_PATH, a checkout of that tree) before any DayCare
# module imports tinygrad. See docs/rl-training.md for the pinned commit.

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from .trainer_env import use_train_tinygrad  # noqa: E402

use_train_tinygrad()
try:
    import tinygrad.llm.nemotron_h_sampler  # noqa: E402,F401
except ImportError as error:  # a vendored upstream tinygrad was imported first
    raise RuntimeError("rloo_tinygrad needs tinygrad-arkey exp as the trainer tinygrad: set "
                       "DAYCARE_TRAIN_TINYGRAD_PATH to a checkout of it before importing DayCare's trainer "
                       "modules (docs/rl-training.md)") from error

from tinygrad import Tensor, TinyJit  # noqa: E402
from tinygrad.nn.optim import Adam  # noqa: E402

from .lora import LoRALinear, apply_lora, lora_params, target_map  # noqa: E402
from .rl_triggers import DEFAULTS as TRIGGERS, Triggers, finished_length as rl_finished_length, row  # noqa: E402
from .rlvr import KL_AGGREGATIONS, leave_one_out, regularizer_denominator, step_policy  # noqa: E402,F401
from .rlvr_update import frozen_lora_tail  # noqa: E402

# Countdown protocol settings (the Countdown RLOO note, not published; the llama.cpp loop that first used them is gone).
# The base model: Nemotron 3 Nano 4B as a BF16 GGUF (docs/rl-training.md). `--model` or DAYCARE_BASE_GGUF.
MODEL = Path(os.environ['DAYCARE_BASE_GGUF']) if os.environ.get('DAYCARE_BASE_GGUF') else None
CONFIG = dict(rank=32, alpha=64, lr=2e-5, group=8, prompts_per_update=4, kl_beta=1e-3, entropy_beta=1e-3,
              max_grad_norm=1.0, temperature=1.0, limit=4096, logprob_tolerance=0.1, slot_ctx=16384, seed=20260924,
              kl_aggregation='matched', triggers=TRIGGERS)  # new runs; records without these ran legacy, untriggered
CONFIG_UPDATES = 20


def render(chat, envelope: dict, request: str) -> str:
    messages = [dict(m) for m in envelope["messages"] if m.get("content") != "{{REQUEST}}"]
    messages.append({"role": "user", "content": request})
    return chat.render(messages, envelope["tools"], generation=True)


def reward(task: dict, rollout: dict) -> tuple[float, str]:
    """1 only for a completed answer the Countdown verifier accepts."""
    from daycare.harness.countdown_reward import score
    if rollout["stop"] != "eos":
        return 0.0, "incomplete"
    if "</think>" not in rollout["text"]:
        return 0.0, "reasoning_not_closed"
    verdict = score(rollout["text"].split("</think>", 1)[1], task["numbers"], task["target"])
    return float(bool(verdict["correct"])), str(verdict.get("error"))


def _common(prompts: list[list[int]]) -> int:
    common = len(prompts[0])
    for other in prompts[1:]:
        common = min(common, next((i for i, (x, y) in enumerate(zip(prompts[0], other)) if x != y),
                                  min(len(prompts[0]), len(other))))
    return common


def _export(folder: Path, adapters, record: dict) -> Path:
    from daycare.artifact.lora_gguf import export
    from daycare.artifact.native_lora import write_checkpoint
    from daycare.nursery.provenance import file_sha256
    tensors = {f"lora.{i}.{name}": getattr(adapter, name).numpy().copy()
               for i, adapter in enumerate(adapters) for name in ("A", "B")}
    write_checkpoint(folder / "adapter.xml", tensors, record)
    record["adapter_sha256"] = file_sha256(folder / "adapter.xml")
    export(folder / "adapter.xml", folder / "adapter.gguf", record)
    return folder / "adapter.gguf"


class NemotronLoRALinear(LoRALinear):
    """LoRALinear plus the base `.weight` the Nemotron-H blocks read for dtypes and shapes."""

    @property
    def weight(self):
        return self.base.weight


def attach_lora(model, rank: int, alpha: int) -> tuple[list, list]:
    """Final-block LoRA (rank/alpha as predeclared); returns the adapters and trainable params.

    Call before building the prefill or sampler: both capture the block's tensors."""
    if model.blk[-1].block_type != "mlp":
        raise ValueError("the predeclared tail is the final MLP block")
    adapters = apply_lora(model, r=rank, alpha=alpha, last_k=1)
    for adapter in adapters:
        adapter.__class__ = NemotronLoRALinear
    params = lora_params(adapters)
    for param in params:
        param.is_param_(True)
    return adapters, params


class Loop:
    """One-stack RLOO state: the model with adapters, the prefill and sampler that share them, the optimizer."""

    def __init__(self, model, cfg: dict, bias: np.ndarray, stop: set[int], prefix_capacity: int,
                 decode=None, piece: int = 1024, lanes: int | None = None, capture: bool = True,
                 shared_capacity: int = 0, prompts: int | None = None):
        from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
        from tinygrad.llm.nemotron_h_sampler import NemotronHRolloutSampler
        self.model, self.cfg, self.stop, self.decode = model, cfg, stop, decode
        Tensor.manual_seed(cfg["seed"])
        self.adapters, self.params = attach_lora(model, cfg["rank"], cfg["alpha"])
        self.optimizer = Adam(self.params, lr=cfg["lr"])
        # Adam's state starts as lazy constants; left lazy, the first step's assigns into them wrote NaN
        # into an adapter tensor on tinygrad-arkey exp (tiny config, several gradient slices)
        opt = self.optimizer
        for state in (opt.lr, opt.b1_t, opt.b2_t, *opt.m, *opt.v):
            state.replace(state.contiguous().realize())
        self.lanes = lanes or cfg["group"]
        # `shared_capacity`: the prefix every prompt shares (the tool envelope) is primed once per sampling call and
        # each prompt prefills only its own part (`prefix_capacity`); `prompts`: resident prompt slots (default two
        # slots' worth of rows per lane; one per lane when prompts are sampled once each, as post-tool follow-ups are)
        self.prefill = NemotronHPrefill(model, capacity=shared_capacity + prefix_capacity, piece=piece)
        # `compact` (cfg, row counts below `lanes`): the step shrinks to the running lanes as rollouts finish, the
        # sampling tail still on all `lanes` rows, so the Tail below recomputes the sampled bits as before
        # (research/rloo-slot-refill.md; needs tinygrad-arkey exp with NemotronHRolloutSampler(compact=...))
        extra = dict(compact=tuple(cfg["compact"])) if cfg.get("compact") else {}
        self.sampler = NemotronHRolloutSampler(model, batch=self.lanes, capacity=cfg["limit"],
                                               prefix_capacity=prefix_capacity, rows=cfg["group"], bias=bias,
                                               prefill=self.prefill, capture=len(model.blk) - 1 if capture else None,
                                               shared_capacity=shared_capacity, prompts=prompts, **extra)
        self.policy = Tail(self.sampler)
        self.reference = Tail(self.sampler, block=frozen_lora_tail(model.blk[-1]))
        self.steps = TailSteps(self.policy, self.reference, self.params, cfg)

    def sample(self, prompts: list[list[int]], counts: list[int] | None = None, follow=None
               ) -> tuple[list[list[dict]], dict]:
        """`counts[i]` rollouts of prompt i (default: the group size each). `follow(request, rollout)` may return new
        `(prompt, n)` requests as each rollout finishes (the sampler's `follow`, with the rollout as a dict here);
        their groups are appended to the result in the order they were added."""
        stats: dict = {}
        counts = counts or [self.cfg["group"]] * len(prompts)
        hook = None if follow is None else (lambda request, rollout: follow(request, self._rollout(*rollout)))
        extra = {} if hook is None else dict(follow=hook)  # `follow` needs tinygrad-arkey exp >= 4322ae9e2
        results = self.sampler.generate([(ids, n) for ids, n in zip(prompts, counts)], max_new=self.cfg["limit"],
                                        temperature=self.cfg["temperature"], stop=self.stop, stats=stats, **extra)
        return [[self._rollout(*rollout) for rollout in rollouts] for rollouts in results], stats

    def _rollout(self, tokens, logprobs, *captured) -> dict:
        vocab = self.model.config.vocab_size
        if bad := [(i, t, logprobs[i]) for i, t in enumerate(tokens) if not 0 <= t < vocab]:
            raise ValueError(f"sampler returned token ids outside the vocabulary ({vocab}): "
                             f"(position, id, logprob) {bad[:4]} of {len(tokens)} tokens")
        ended = bool(tokens) and tokens[-1] in self.stop
        body = tokens[:-1] if ended else tokens
        out = dict(tokens=[int(t) for t in tokens], logprobs=[float(v) for v in logprobs],
                   stop="eos" if ended else "limit", text=self.decode(body) if self.decode else "")
        if captured:  # the hidden state entering the final block at each sampled token's step
            out["hidden"] = np.asarray(captured[0], dtype=np.float32)
        return out

    def update(self, prompts: list[list[int]], score) -> dict:
        """Sample, score, recompute and step once; `score(index, rollout) -> (reward, reason)`."""
        times: dict[str, float] = {}
        clock = time.perf_counter()
        sampled, stats = self.sample(prompts)
        times["sample_s"] = time.perf_counter() - clock
        times["sample_prefill_s"] = stats.get("prime_s", 0.0)
        clock = time.perf_counter()
        scored = [[score(index, rollout) for rollout in group] for index, group in enumerate(sampled)]
        times["score_s"] = time.perf_counter() - clock
        return self.learn(sampled, scored, times, stats)

    def learn(self, sampled: list[list[dict]], scored: list[list[tuple]], times: dict, stats: dict) -> dict:
        """One RLOO step on scored groups; each rollout holds the tokens, sampler logprobs and captured final-block
        inputs of every token it sampled (a multi-turn episode concatenates its turns: prompts and tool results
        are never sampled, so they never enter the loss)."""
        groups, summary = [], []
        if not sampled:  # the loss mask left nothing to learn: no step, recorded as skipped (rloo_posttool.masked)
            metrics = dict.fromkeys(("loss", "policy_loss", "kl", "entropy", "max_logprob_error", "rho_mean", "rho_max",
                                     "rho_clipped", "adapter_step_rms", "grad_norm"), 0.0)
            return dict(metrics=dict(metrics, stepped=False, skipped=True, trajectories=0, generated_tokens=0),
                        groups=[], times=times, sampler=stats, mean_reward=float("nan"), mixed_groups=0)
        if not all("hidden" in r for group in sampled for r in group):
            raise ValueError("the sampler did not capture final-block inputs (capture=len(model.blk) - 1)")
        for rollouts, marks in zip(sampled, scored):
            hidden = [r["hidden"] for r in rollouts]
            rewards = [value for value, _ in marks]
            groups.append(dict(advantages=leave_one_out(rewards), rollouts=[
                dict(hidden=h, tokens=np.asarray(r["tokens"], dtype=np.int32),
                     old_logp=np.asarray(r["logprobs"], dtype=np.float32),
                     **({"bonus": np.asarray(r["token_bonus"], dtype=np.float32)} if "token_bonus" in r else {}))
                for r, h in zip(rollouts, hidden)]))
            summary.append(dict(rewards=rewards, reasons=[why for _, why in marks],
                                tokens=[len(r["tokens"]) for r in rollouts], stops=[r["stop"] for r in rollouts]))
        metrics = update(self.steps, self.optimizer, groups, self.cfg, times)
        rewards = [r for g in summary for r in g["rewards"]]
        metrics["features"] = "captured"
        return dict(metrics=metrics, groups=summary, times=times, sampler=stats,
                    mean_reward=float(np.mean(rewards)), mixed_groups=sum(len(set(g["rewards"])) > 1 for g in summary))


class Tail:
    """Log probabilities from the hidden state entering the final block, through the sampler's own tail.

    The sampler captures that hidden state at every sampled token (`capture`) and
    runs the tail with `_stateless` + `_logprobs`; this calls the same code on the
    same row count (the sampler's lanes, zero-padded), so the kernels are the
    decode step's and the sampled log probabilities are reproduced bit for bit.
    `block` defaults to the model's adapted final block; a frozen copy gives the
    KL reference. Unrealized, so gradients reach the adapter."""

    def __init__(self, sampler, block=None):
        self.sampler, self.block, self.rows = sampler, block, sampler.batch

    def forward(self, x: Tensor, temperature: Tensor) -> Tensor:
        """`x` `[rows, 1, dim]` float32 -> log probabilities `[rows, vocab]` (sampler_tail_logprobs' ops)."""
        x = self.sampler._stateless(self.block or self.sampler.model.blk[-1], x)
        return self.sampler._logprobs(x, temperature)

    def pad(self, hidden: np.ndarray) -> Tensor:
        count, dim = hidden.shape
        if count > self.rows:
            raise ValueError("more rows than the decode step runs")
        rows = np.zeros((self.rows, dim), dtype=np.float32)
        rows[:count] = hidden
        return Tensor(rows).reshape(self.rows, 1, dim)

    def logprobs(self, hidden: np.ndarray, temperature: float) -> Tensor:
        return self.forward(self.pad(hidden), Tensor([temperature]))[:len(hidden)]


def flatten(groups) -> dict:
    """Every sampled token of the batch as rows: its final-block input, id, sampler logprob and advantage."""
    parts = [(r, float(g["advantages"][e])) for g in groups for e, r in enumerate(g["rollouts"])]
    return dict(hidden=np.concatenate([r["hidden"] for r, _ in parts]).astype(np.float32),
                tokens=np.concatenate([r["tokens"] for r, _ in parts]).astype(np.int32),
                old_logp=np.concatenate([r["old_logp"] for r, _ in parts]).astype(np.float32),
                # a rollout's per-token `bonus` (a dense shaping term, e.g. the repetition penalty) adds to its advantage
                advantage=np.concatenate([np.full(len(r["tokens"]), a, dtype=np.float32) + r.get("bonus", 0.0)
                                          for r, a in parts]).astype(np.float32),
                trajectories=len(parts))


def _padded(values: np.ndarray, rows: int, dtype) -> Tensor:
    out = np.zeros(rows, dtype=dtype)
    out[:len(values)] = values
    return Tensor(out)


class TailSteps:
    """The tail's per-slice work as compiled graphs (TinyJit), one slice = the sampler's lane rows.

    `selected` recomputes the sampled tokens' log probabilities (the parity check);
    `grad` adds one slice's objective gradient into device accumulators. Each is
    one graph replay per slice, so the host no longer schedules the tail per slice.
    The graphs read the adapter's buffers, which Adam updates in place, and take
    every per-update quantity (weights, advantages, denominators) as inputs."""

    def __init__(self, tail: Tail, reference: Tail, params, cfg: dict):
        self.tail, self.reference, self.params = tail, reference, params
        self.kl_beta, self.entropy_beta = float(cfg.get("kl_beta", 0.0)), float(cfg.get("entropy_beta", 0.0))
        self.kl_aggregation = cfg.get("kl_aggregation", "legacy_token_mean")  # absent from runs 1 and 2
        self.temperature = Tensor([float(cfg["temperature"])]).contiguous().realize()
        self.acc = [Tensor.zeros(*p.shape).contiguous().realize() for p in params]
        self.sums = Tensor.zeros(4).contiguous().realize()
        self.selected, self.grad = TinyJit(self._selected), TinyJit(self._grad)

    def _selected(self, x: Tensor, ids: Tensor) -> Tensor:
        lp = self.tail.forward(x, self.temperature)
        return lp.gather(1, ids.reshape(-1, 1)).reshape(-1).contiguous().realize()

    def _grad(self, x: Tensor, ids: Tensor, scale: Tensor, mask: Tensor) -> None:
        """Sequence RLOO with per-token weights plus KL minus entropy, as rlvr.regularized_token_loss but with
        padded rows masked out: `scale` is weight * advantage / trajectories and `mask` 1 / the kl_aggregation's
        denominator (token_count or trajectories) on real rows, 0 on padding."""
        ref = self.reference.forward(x, self.temperature).detach() if self.kl_beta else None
        with Tensor.train():
            lp = self.tail.forward(x, self.temperature)
            probs = lp.exp()
            pg = -(lp.gather(1, ids.reshape(-1, 1)).reshape(-1) * scale).sum()
            entropy = -((probs * lp).sum(-1) * mask).sum()
            kl = ((probs * (lp - ref)).sum(-1) * mask).sum() if ref is not None else pg * 0
            loss = pg + self.kl_beta * kl - self.entropy_beta * entropy
            grads = loss.gradient(*self.params)
            for acc, g in zip(self.acc, grads):
                acc.assign(acc + g)
            self.sums.assign(self.sums + Tensor.stack(loss, pg, kl, entropy))
            Tensor.realize(self.sums, *self.acc)

    def reset(self):
        for value in (*self.acc, self.sums):
            value.assign(Tensor.zeros(*value.shape)).realize()


def parity_weights(steps: TailSteps, batch: dict, cfg) -> dict:
    """Recompute every sampled token under the current adapter; refuse past the parity tolerance.

    Stores the truncated importance weights min(rho, C), rho = exp(logp_train - logp_sampler)
    (ones with `tis` off), as `batch["weight"]`; returns the statistics."""
    cap = float(cfg.get("tis_cap", 2.0)) if cfg.get("tis", True) else None
    rows, mine = steps.tail.rows, []
    for start in range(0, len(batch["tokens"]), rows):
        part = slice(start, start + rows)
        count = len(batch["tokens"][part])
        values = steps.selected(steps.tail.pad(batch["hidden"][part]), _padded(batch["tokens"][part], rows, np.int32))
        mine.append(values.numpy()[:count])
    mine, old = np.concatenate(mine), batch["old_logp"]
    errors, rho = np.abs(mine - old), np.exp(mine.astype(np.float64) - old)
    batch["weight"] = np.minimum(rho, cap).astype(np.float32) if cap else np.ones_like(old)
    stats = dict(max_logprob_error=float(np.nanmax(errors)), mean_logprob_error=float(np.nanmean(errors)),
                 p99_logprob_error=float(np.nanquantile(errors, 0.99)), exact=int((mine == old).sum()),
                 over_tolerance=int((~(errors <= cfg["logprob_tolerance"])).sum()), checked=int(errors.size),
                 worst_token_logprob=float(old[np.nanargmax(errors)]), tis=cap is not None,
                 rho_mean=float(rho.mean()), rho_max=float(rho.max()), rho_min=float(rho.min()),
                 rho_clipped=float((rho > cap).mean()) if cap else 0.0)
    if stats["over_tolerance"] or not np.isfinite(errors).all():
        raise ParityError(stats)
    return stats


class ParityError(ValueError):
    def __init__(self, stats: dict):
        super().__init__(f"sampler/trainer logprob mismatch: {stats}")
        self.stats = stats


def accumulate(steps: TailSteps, batch: dict) -> dict:
    """Sum the batch's gradient into `p.grad`, one compiled slice graph per `tail.rows` tokens.

    Gradients go to the adapter only (`Tensor.gradient`): `backward` would also give one to
    every live alias of a parameter's buffer (the prefill's graph inputs), and that alias's
    gradient double-writes the parameter's gradient buffer on accumulation."""
    rows, token_count = steps.tail.rows, len(batch["tokens"])
    scale = batch["weight"] * batch["advantage"] / batch["trajectories"]
    denominator = regularizer_denominator(steps.kl_aggregation, token_count, batch["trajectories"])
    steps.reset()
    clock, slices = time.perf_counter(), 0
    for start in range(0, token_count, rows):
        part = slice(start, start + rows)
        count = len(batch["tokens"][part])
        steps.grad(steps.tail.pad(batch["hidden"][part]), _padded(batch["tokens"][part], rows, np.int32),
                   _padded(scale[part], rows, np.float32), _padded(np.full(count, 1.0 / denominator), rows, np.float32))
        slices += 1
    totals = steps.sums.numpy().astype(np.float64)
    seconds = time.perf_counter() - clock
    if not np.isfinite(totals).all():
        raise ValueError(f"nonfinite RL objective {totals}")
    for p, acc in zip(steps.params, steps.acc):
        p.grad = (acc + 0).contiguous().realize()
    mean = denominator / token_count  # kl and entropy are reported as token means under every aggregation
    return dict(loss=float(totals[0]), policy_loss=float(totals[1]), kl=float(totals[2] * mean),
                entropy=float(totals[3] * mean), kl_aggregation=steps.kl_aggregation,
                # the KL term's per-token weight over the policy gradient's mean one (the audit's D1 ratio)
                kl_weight_ratio=steps.kl_beta / denominator / max(float(np.abs(scale).mean()), 1e-30),
                trajectories=batch["trajectories"], generated_tokens=token_count, grad_slices=slices,
                grad_slice_mean_s=seconds / slices)


def update(steps: TailSteps, optimizer, groups, cfg, times: dict | None = None) -> dict:
    """Parity-check the whole batch first, then accumulate once and step once (update_batch's order)."""
    times = {} if times is None else times
    batch = flatten(groups)
    steps.last_batch = batch
    clock = time.perf_counter()
    metrics = parity_weights(steps, batch, cfg)
    times["parity_s"] = time.perf_counter() - clock
    clock = time.perf_counter()
    stepped = bool(cfg.get("kl_beta") or cfg.get("entropy_beta") or batch["advantage"].any())
    metrics.update(accumulate(steps, batch) if stepped else dict(loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0))
    times["grad_s"] = time.perf_counter() - clock
    before = [p.numpy() for p in steps.params]
    adam = time.perf_counter()
    metrics["grad_norm"] = step_policy(optimizer, steps.params, cfg.get("max_grad_norm", 1.0)) if stepped else 0.0
    Tensor.realize(*steps.params)
    times["adam_s"] = time.perf_counter() - adam
    # instrumentation: how far the step moved the adapter, and how much the clip scaled the gradient
    delta = np.concatenate([(p.numpy() - b).ravel() for p, b in zip(steps.params, before)])
    metrics.update(adapter_step_rms=float(np.sqrt(np.mean(np.square(delta)))), adapter_step_max=float(np.abs(delta).max()),
                   clip_scale=float(min(1.0, cfg.get("max_grad_norm", 1.0) / metrics["grad_norm"]))
                   if metrics["grad_norm"] and cfg.get("max_grad_norm", 1.0) else 1.0)
    metrics["stepped"] = stepped
    times["update_s"] = time.perf_counter() - clock
    return metrics


def load_model(path: Path | None, ctx: int):
    from tinygrad.llm.nemotron_h import load
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"base GGUF not found ({path}): pass --model or set DAYCARE_BASE_GGUF "
                                "(docs/rl-training.md)")
    from daycare.model.gguf_tokenizer import tokenizer_from_gguf
    model, metadata = load(str(path), max_context=ctx)
    return model, tokenizer_from_gguf(metadata), metadata


def setup(args, cfg):
    """Envelope, model, tokenizer, prompt renderer and logit bias, as every action uses them."""
    from daycare.artifact.record_xml import read
    from daycare.model.chat_format import NativeToolChat
    envelope = read(args.envelope)
    clock = time.perf_counter()
    model, tokenizer, metadata = load_model(args.model, cfg["slot_ctx"])
    load_s = time.perf_counter() - clock
    chat = NativeToolChat(metadata["tokenizer.chat_template"], tokenizer,
                          thinking=envelope["chat_template_kwargs"]["enable_thinking"])
    bias = np.zeros(model.config.vocab_size, dtype=np.float32)
    for key, value in envelope.get("logit_bias", {}).items():
        bias[int(key)] = float(value)

    def render_prompt(task):
        return tokenizer.encode(render(chat, envelope, task["request"]))
    return envelope, model, tokenizer, render_prompt, bias, load_s


def load_adapter(adapters, path: Path, record: dict | None = None) -> None:
    """Write an exported adapter checkpoint into the attached adapters' buffers in place."""
    from daycare.artifact.native_lora import read_checkpoint
    _, tensors = read_checkpoint(path, record)
    for index, adapter in enumerate(adapters):
        for name in ("A", "B"):
            value, target = tensors[f"lora.{index}.{name}"], getattr(adapter, name)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"adapter tensor lora.{index}.{name} has shape {value.shape}, expected {target.shape}")
            target.assign(Tensor(np.ascontiguousarray(value, dtype=np.float32), device=target.device)).realize()


def check_raw(adapters, path: Path) -> None:
    """The loaded adapter equals the raw tensors saved before the formatted export, bit for bit."""
    with np.load(path) as raw:
        for index, adapter in enumerate(adapters):
            for name in ("A", "B"):
                if not np.array_equal(getattr(adapter, name).numpy(), raw[f"lora.{index}.{name}"]):
                    raise ValueError(f"lora.{index}.{name} differs from the raw save")


def verify(args):
    """Reload a run's exported adapter in a fresh process; its tail must reproduce the probe bit for bit."""
    from daycare.artifact.record_xml import read, write
    run = read(args.run / "run.xml")
    # the probe check runs the tail forward only (no sampling), so the sampler's step compaction does not apply; its
    # 64-token probe loop has too few prompt slots for the run's row counts (research/rloo-posttool-calculator-r4.md P2)
    cfg = {k: v for k, v in run["config"].items() if k != "compact"}
    _, model, tokenizer, _, bias, _ = setup(args, cfg)
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    loop = Loop(model, cfg, bias, stop, prefix_capacity=64, lanes=cfg["lanes"])
    load_adapter(loop.adapters, args.run / "adapter.xml", run)
    check_raw(loop.adapters, args.run / "adapter-raw.npz")
    with np.load(args.run / "probe.npz") as probe:
        values = loop.policy.forward(loop.policy.pad(probe["hidden"]), loop.steps.temperature).numpy()
        saved = probe["logprobs"]
    result = dict(rows=int(saved.shape[0]), vocab=int(saved.shape[1]), bit_exact=bool(np.array_equal(values, saved)),
                  max_abs_error=float(np.abs(values - saved).max()),
                  exact_values=int((values == saved).sum()), values=int(saved.size))
    write(args.run / "verify.xml", result, root="verify")
    print("verify", result, flush=True)
    if not result["bit_exact"]:
        raise SystemExit("reloaded adapter does not reproduce the trained tail")


def evaluate(args):
    """R5: holdout tasks, stock (zero adapter) vs the exported adapter, `seeds` rollouts per task and arm,
    first turn, the training reward; task-clustered bootstrap of the paired difference."""
    from daycare.artifact.record_xml import read, write
    run = read(args.run / "run.xml")
    cfg = dict(run["config"], limit=args.limit, group=args.seeds, lanes=args.lanes or 32)
    envelope, model, tokenizer, render_prompt, bias, _ = setup(args, cfg)
    tasks = read(args.holdout)["tasks"][:args.tasks]  # --tasks: a short end-to-end check, never the gate
    prompts = [render_prompt(task) for task in tasks]
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    # no capture: evaluation needs no final-block inputs, which cost ~12 KB per token of host memory
    loop = Loop(model, cfg, bias, stop, prefix_capacity=max(len(p) for p in prompts), decode=tokenizer.decode,
                lanes=cfg["lanes"], capture=False)
    arms = {}
    for arm in ("stock", "adapter"):
        if arm == "adapter":
            load_adapter(loop.adapters, args.run / "adapter.xml", run)
        Tensor.manual_seed(args.seed + (0 if arm == "stock" else 1))
        clock = time.perf_counter()
        sampled, stats = loop.sample(prompts)
        rows = []
        for task, rollouts in zip(tasks, sampled):
            for k, rollout in enumerate(rollouts):
                value, why = reward(task, rollout)
                rows.append(dict(task=task["id"], rollout=k, reward=value, reason=why, tokens=len(rollout["tokens"]),
                                 stop=rollout["stop"], text=rollout["text"]))
        del sampled
        arms[arm] = dict(rows=rows, seconds=time.perf_counter() - clock, sampler=stats)
        write(args.root / f"{arm}.xml", arms[arm], root="arm")
        print(arm, "correct", sum(r["reward"] for r in rows), "of", len(rows),
              f"({arms[arm]['seconds']:.0f} s)", flush=True)
    ids = [task["id"] for task in tasks]

    def table(arm):
        values = {(r["task"], r["rollout"]): r["reward"] for r in arms[arm]["rows"]}
        return np.array([[values[(i, k)] for k in range(args.seeds)] for i in ids])
    stock, adapter = table("stock"), table("adapter")
    diff = (adapter - stock).mean(axis=1)
    boot = diff[np.random.default_rng(20260923).integers(len(ids), size=(50000, len(ids)))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    result = dict(tasks=len(ids), rollouts_per_arm=int(stock.size), stock_correct=int(stock.sum()),
                  adapter_correct=int(adapter.sum()), stock_mean=float(stock.mean()), adapter_mean=float(adapter.mean()),
                  difference=float(diff.mean()), ci95=[float(low), float(high)], passed=bool(low > 0) and args.tasks is None,
                  task_gains=int((diff > 0).sum()), task_losses=int((diff < 0).sum()),
                  adapter_sha256=run.get("adapter_sha256"), run=str(args.run), limit=args.limit,
                  complete_holdout=args.tasks is None,
                  seconds={arm: arms[arm]["seconds"] for arm in arms})
    write(args.root / "r5.xml", result, root="r5")
    print("R5", result, flush=True)


def train(args):
    from daycare.artifact.record_xml import write
    from .provenance import file_sha256, source_revision
    cfg = dict(CONFIG, updates=args.updates, prompts_per_update=args.prompts_per_update, group=args.group,
               limit=args.limit, lanes=args.lanes or args.group * args.prompts_per_update,
               bucket=512,
               tis=not args.no_tis, tis_cap=2.0, logprob_tolerance=args.logprob_tolerance)
    # only the predeclared settings count toward the Countdown predeclaration
    counted = args.logprob_tolerance == CONFIG["logprob_tolerance"] and not args.no_tis
    cfg["lr"], cfg["protocol"] = args.lr, args.protocol
    revision = source_revision()
    envelope, model, tokenizer, render_prompt, bias, load_s = setup(args, cfg)
    tasks = [t for t in json.loads((args.prep / "sft-tasks.json").read_text()) if len(t["numbers"]) == 4]
    np.random.default_rng(cfg["seed"]).shuffle(tasks)
    prompts = [render_prompt(task) for task in tasks]
    stop = {int(t) for t in (tokenizer.eos_id, tokenizer.eot_id) if t is not None}
    clock = time.perf_counter()
    loop = Loop(model, cfg, bias, stop, prefix_capacity=max(len(p) for p in prompts), decode=tokenizer.decode,
                lanes=cfg["lanes"])
    setup_s = time.perf_counter() - clock
    record = dict(schema="daycare.rloo_tinygrad.v1", complete=False, daycare_revision=revision,
                  runner_sha256=file_sha256(Path(__file__)), model_sha256=file_sha256(args.model),
                  envelope_sha256=file_sha256(args.envelope), tinygrad=os.environ["DAYCARE_TRAIN_TINYGRAD_PATH"],
                  tinygrad_revision=_git_head(os.environ["DAYCARE_TRAIN_TINYGRAD_PATH"]),
                  model_profile=dict(architecture="nemotron_h", precision="bf16"), rank=cfg["rank"],
                  alpha=cfg["alpha"], last_k=1, target_map=target_map(loop.adapters), lr=cfg["lr"],
                  seed=cfg["seed"], examples=len(tasks), common_prefix=_common(prompts),
                  training_objective="rloo-tinygrad-sampler", config=cfg, counted=counted, load_s=load_s, setup_s=setup_s, losses=[])
    print(f"load {load_s:.1f}s setup {setup_s:.1f}s counted {counted}", flush=True)
    run(loop, tasks, prompts, cfg, record, args.root, lambda task, rollout: reward(task, rollout), export=args.export)


# write_checkpoint's metadata fields (native_lora.write_checkpoint / validate_metadata); a one-stack
# record carries each, and `run` exports the initial adapter before any update to prove it.
CHECKPOINT_FIELDS = ("losses", "model_sha256", "rank", "alpha", "last_k", "target_map", "epochs", "model_profile",
                     "schema")


def export_adapter(folder: Path, loop, record: dict) -> dict:
    """Save the adapter: raw tensors (npz + sha256) first, so no formatting failure can lose trained
    weights, then the lossless XML checkpoint and GGUF (`_export`) and the reload probe."""
    from .provenance import file_sha256
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {f"lora.{i}.{name}": getattr(adapter, name).numpy().copy()
               for i, adapter in enumerate(loop.adapters) for name in ("A", "B")}
    np.savez(folder / "adapter-raw.npz", **tensors)
    saved = dict(adapter_raw_sha256=file_sha256(folder / "adapter-raw.npz"))
    (folder / "adapter-raw.sha256").write_text(saved["adapter_raw_sha256"] + "  adapter-raw.npz\n")
    missing = [key for key in CHECKPOINT_FIELDS if key not in record]
    if missing:
        raise ValueError(f"the run record lacks checkpoint fields {missing}")
    saved["adapter"] = str(_export(folder, loop.adapters, dict(record, losses=record["losses"] or [dict(loss=0.0)])))
    saved["adapter_sha256"] = file_sha256(folder / "adapter.xml")
    # the adapter's tail on fixed rows, for the reload check (`verify`)
    batch = getattr(loop.steps, "last_batch", None)
    rows, dim = loop.lanes, loop.model.config.dim
    hidden = batch["hidden"][:rows] if batch is not None else \
        np.random.default_rng(0).standard_normal((rows, dim)).astype(np.float32)
    tokens = batch["tokens"][:rows] if batch is not None else np.zeros(rows, dtype=np.int32)
    values = loop.policy.forward(loop.policy.pad(hidden), loop.steps.temperature).numpy()
    np.savez(folder / "probe.npz", hidden=hidden, tokens=tokens, logprobs=values)
    saved["probe_sha256"] = file_sha256(folder / "probe.npz")
    return saved


def run(loop, tasks, prompts, cfg, record, root: Path, score, export: bool = True, step=None, keep_every: int = 0) -> dict:
    """The update loop: `score(task, rollout) -> (reward, reason)`; records under `root`. `step(picks)`, when given,
    replaces sample-and-score for the picked task indices (a multi-turn episode source) and returns `learn`'s result.
    `keep_every`: also save the raw adapter tensors every that many updates (`checkpoint-NNN.npz`), so a crash in a
    long run loses at most that many updates."""
    from daycare.artifact.record_xml import write
    record.setdefault("epochs", 1)  # one pass over each freshly sampled batch
    if export:  # prove the export path on the initial adapter before any update
        record["initial_export"] = export_adapter(root / "initial-adapter", loop, record)
    cursor, triggers = 0, Triggers(cfg["triggers"]) if cfg.get("triggers") else None
    for update in range(cfg["updates"]):
        folder = root / f"update-{update:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        picks = [(cursor + i) % len(tasks) for i in range(cfg["prompts_per_update"])]
        cursor += len(picks)
        clock = time.perf_counter()
        try:
            result = step(picks) if step is not None else loop.update(
                [prompts[i] for i in picks], lambda index, rollout: score(tasks[picks[index]], rollout))
        except ParityError as error:
            record["parity_failure"] = dict(step=update + 1, **error.stats)
            write(root / "run.xml", record, root="run")
            raise
        result["times"]["total_s"] = time.perf_counter() - clock
        for group, i in zip(result["groups"], picks):
            group.setdefault("task", tasks[i]["id"])  # a step that dropped groups labels its own
        metrics, reason = result["metrics"], None
        if triggers is not None:
            observed = row(result)
            metrics.setdefault("finished_length", observed["finished_length"])
            reason = triggers.check(observed)
        record["skipped_updates"] = record.get("skipped_updates", 0) + bool(metrics.get("skipped"))
        record["losses"].append(dict(step=update + 1, loss=metrics["loss"], mean_reward=result["mean_reward"],
                                     mixed_groups=result["mixed_groups"], **result["times"],
                                     **{k: v for k, v in metrics.items() if k != "loss"}))
        write(folder / "update.xml", result, root="update")
        if reason:  # a rolling stop trigger (rl_triggers): keep this adapter, record why, end the loop cleanly
            record["stopped"] = dict(update=update + 1, reason=reason)
            print(f"stop trigger at update {update + 1}: {reason}", flush=True)
        if reason or keep_every and (update + 1) % keep_every == 0:
            np.savez(root / f"checkpoint-{update + 1:03d}.npz", **{f"lora.{i}.{name}": getattr(a, name).numpy()
                                                                  for i, a in enumerate(loop.adapters) for name in ("A", "B")})
            record["checkpoint"] = update + 1
        write(root / "progress.xml", record, root="progress")
        finished_length = metrics.get("finished_length")
        if finished_length is None and result.get("episodes"):
            finished_length = rl_finished_length(result["episodes"])
        print(f"update {update + 1}: reward {result['mean_reward']:.3f} mixed {result['mixed_groups']} "
              f"loss {metrics['loss']:.5f} max_gap {metrics['max_logprob_error']:.4f} kl {metrics['kl']:.5f} "
              f"entropy {metrics['entropy']:.3f} "
              f"rho {metrics['rho_mean']:.4f}/{metrics['rho_max']:.4f} clipped {metrics['rho_clipped']:.3f} "
              f"step_rms {metrics['adapter_step_rms']:.2e} " + ("SKIPPED (no group to learn from) " if metrics.get("skipped") else "")
              + "".join(f"{k} {metrics[k]:.3f} " for k in ("other_tool_rate", "capped_turn_rate") if k in metrics)
              + (f"finished_length {finished_length:.1f} " if finished_length is not None and not np.isnan(finished_length) else "")
              + f"times {json.dumps({k: round(v, 1) for k, v in result['times'].items()})}", flush=True)
        if reason:
            break
    record["complete"] = True
    if export:
        record.update(export_adapter(root, loop, record))
    write(root / "run.xml", record, root="run")
    return record


def _git_head(path: str) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("train", "verify", "evaluate"))
    parser.add_argument("--root", type=Path, help="train/evaluate: output folder")
    parser.add_argument("--run", type=Path, help="verify/evaluate: a finished training run's folder")
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--prep", type=Path, help="train: the task preparation folder")
    parser.add_argument("--holdout", type=Path, help="evaluate: the holdout tasks.xml")
    parser.add_argument("--seeds", type=int, default=2, help="evaluate: rollouts per task and arm")
    parser.add_argument("--seed", type=int, default=20260925, help="evaluate: sampling seed")
    parser.add_argument("--tasks", type=int, default=None, help="evaluate: first N tasks only (output-path check)")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--updates", type=int, default=CONFIG_UPDATES)
    parser.add_argument("--prompts-per-update", type=int, default=CONFIG["prompts_per_update"])
    parser.add_argument("--group", type=int, default=CONFIG["group"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--lanes", type=int, default=None,
                        help="sampler lanes, also the tail's slice rows (default: group x prompts per update)")
    parser.add_argument("--logprob-tolerance", type=float, default=CONFIG["logprob_tolerance"],
                        help="parity hard stop; any other value marks the run not counted")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Adam learning rate (v1: 2e-5)")
    parser.add_argument("--protocol", default="rloo-tinygrad-countdown.md",
                        help="the predeclaration this run belongs to (recorded)")
    parser.add_argument("--no-tis", action="store_true", help="drop the truncated importance weights")
    parser.add_argument("--no-export", dest="export", action="store_false",
                        help="skip exporting the final adapter (GGUF + checkpoint; R5 evaluates it)")
    args = parser.parse_args()
    if args.model is None:
        parser.error("pass --model or set DAYCARE_BASE_GGUF (Nemotron 3 Nano 4B, BF16 GGUF; docs/rl-training.md)")
    if args.root is not None:
        args.root.mkdir(parents=True, exist_ok=True)
    {"train": train, "verify": verify, "evaluate": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
