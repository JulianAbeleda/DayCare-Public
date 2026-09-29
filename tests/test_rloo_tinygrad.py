"""One-stack RLOO (daycare/nursery/rloo_tinygrad.py) on a tiny Nemotron-H.

tinygrad-arkey exp must be the process's only tinygrad, and this pytest
process may already hold the vendored train tinygrad, so the loop runs in a
child process (DEV=CPU by default) and reports JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

EXP = Path(os.environ.get("DAYCARE_TRAIN_TINYGRAD_PATH") or "$DAYCARE_TRAIN_TINYGRAD_PATH")  # tinygrad-arkey exp


class TailPolicy:
    """Reference tail for update_batch: the last block plus the output head (was rloo_llama.TailPolicy)."""

    def __init__(self, model, bias: np.ndarray):
        self.model, self.bias = model, bias

    def logprobs(self, hidden, temperature: float = 1.0):
        from tinygrad import Tensor
        logits = self.model.output(self.model.output_norm(self.model.blk[-1](hidden.float()))).float()
        return ((logits + Tensor(self.bias, device=logits.device)) / temperature).log_softmax()


def tiny_model():
    from tinygrad import Tensor
    from tinygrad.llm.nemotron_h import NemotronHConfig, NemotronHModel
    config = NemotronHConfig(num_blocks=4, dim=32, vocab_size=64, norm_eps=1e-5, max_context=64,
                             block_types=("mamba", "mlp", "attention", "mlp"),
                             head_counts=(0, 0, 4, 0), kv_head_counts=(0, 0, 2, 0), ffn_dims=(0, 48, 0, 48),
                             head_dim=8, ssm_inner=32, ssm_state=8, ssm_groups=2, ssm_heads=4,
                             conv_kernel=4, scan_chunk=4)
    Tensor.manual_seed(0)
    model = NemotronHModel(config)
    mamba = model.blk[0]
    mamba.ssm_a = -(Tensor.rand(config.ssm_heads, 1) + 0.1)
    mamba.ssm_d = Tensor.rand(config.ssm_heads, 1)
    mamba.ssm_dt = {"bias": Tensor.rand(config.ssm_heads) - 0.5}
    mamba.ssm_conv1d.weight = Tensor.randn(*mamba.ssm_conv1d.weight.shape) * 0.3
    mamba.ssm_norm.weight = Tensor.ones(*mamba.ssm_norm.weight.shape)
    return model


def run_tiny(root: Path) -> dict:
    import numpy as np
    from daycare.nursery.rloo_tinygrad import Loop
    model = tiny_model()
    cfg = dict(rank=4, alpha=8, lr=1e-2, group=4, prompts_per_update=2, kl_beta=1e-3, entropy_beta=1e-3,
               max_grad_norm=1.0, temperature=1.0, limit=12, logprob_tolerance=0.1, seed=7, bucket=8,
               feature_batch=2)
    bias = np.zeros(64, dtype=np.float32)
    bias[5] = -3.0
    loop = Loop(model, cfg, bias, stop={0}, prefix_capacity=16, piece=8)
    prompts = [[3, 17, 5, 42, 9, 11, 2], [7, 7, 1, 60, 2]]
    before = [p.numpy().copy() for p in loop.params]
    updates = []
    for update in range(2):
        # mixed groups: reward rollouts whose token sum is even
        result = loop.update(prompts, lambda _, rollout: (float(sum(rollout["tokens"]) % 2 == 0), "tiny"))
        after = [p.numpy().copy() for p in loop.params]
        result["adapter_delta"] = float(max(np.abs(a - b).max() for a, b in zip(after, before)))
        before = after
        updates.append(result)
    # the adapter must move the policy far beyond the parity gap, or parity on the
    # second update could not tell a stale sampler graph from a fresh one
    hidden = np.random.default_rng(3).standard_normal((4, 32)).astype(np.float32)
    effect = float((loop.policy.logprobs(hidden, 1.0) - loop.reference.logprobs(hidden, 1.0)).abs().max().item())
    return dict(updates=updates, adapter_effect=effect, export=check_export(cfg, bias, root),
                **check_tis(loop, cfg, root))


def check_export(cfg, bias, root: Path) -> dict:
    """rloo_tinygrad.run end to end with export: raw save, XML + GGUF, probe; a fresh process-like
    reload (new model, new loop) must match the raw tensors and reproduce the probe bit for bit."""
    import numpy as np
    from daycare.nursery import rloo_tinygrad as one
    loop = one.Loop(tiny_model(), dict(cfg, updates=2), bias, stop={0}, prefix_capacity=16, piece=8)
    tasks = [dict(id="a"), dict(id="b")]
    prompts = [[3, 17, 5, 42, 9, 11, 2], [7, 7, 1, 60, 2]]
    record = dict(schema="daycare.rloo_tinygrad.v1", rank=cfg["rank"], alpha=cfg["alpha"], last_k=1,
                  target_map=one.target_map(loop.adapters), model_profile=dict(architecture="nemotron_h", precision="bf16"),
                  model_sha256="0" * 64, lr=cfg["lr"], seed=cfg["seed"], examples=2, training_objective="test",
                  config=cfg, losses=[])
    out = root / "run"
    out.mkdir()
    done = one.run(loop, tasks, prompts, dict(cfg, updates=2), record, out,
                   lambda task, rollout: (float(sum(rollout["tokens"]) % 2 == 0), "tiny"))
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    fresh = one.Loop(tiny_model(), cfg, bias, stop={0}, prefix_capacity=16, piece=8)
    one.load_adapter(fresh.adapters, out / "adapter.xml", done)
    one.check_raw(fresh.adapters, out / "adapter-raw.npz")
    with np.load(out / "probe.npz") as probe:
        values = fresh.policy.forward(fresh.policy.pad(probe["hidden"]), fresh.steps.temperature).numpy()
        exact = bool(np.array_equal(values, probe["logprobs"]))
    moved = float(np.abs(fresh.adapters[-1].B.numpy()).max())
    return dict(files=files, probe_exact=exact, updates=len(done["losses"]), trained_b_max=moved,
                initial_export=sorted(done["initial_export"]))


def check_tis(loop, cfg, root: Path) -> dict:
    """TIS off matches rlvr_update.update_batch's gradient; TIS scales each token's policy term by min(rho, C)."""
    import numpy as np
    from daycare.nursery import rloo_tinygrad as one
    from daycare.nursery.provenance import file_sha256
    from daycare.nursery.rlvr_update import TailReference, update_batch
    from tinygrad import Tensor
    from tinygrad.nn.state import get_state_dict
    prompt = [3, 17, 5, 42, 9, 11, 2]
    sampled, _ = loop.sample([prompt])
    rollouts = sampled[0]
    advantages = [1.0, -1.0, 0.5, -0.5]

    def batch(tail, shift, advantages=advantages):
        # the recomputed logprobs, so parity is exact and rho is set by the shift
        rows = []
        for r in rollouts:
            lp = np.concatenate([tail.logprobs(r["hidden"][s:s + tail.rows], 1.0).numpy()
                                 for s in range(0, len(r["tokens"]), tail.rows)])
            rows.append(dict(hidden=r["hidden"], tokens=np.asarray(r["tokens"], dtype=np.int32),
                             old_logp=(lp[np.arange(len(r["tokens"])), r["tokens"]] - shift).astype(np.float32)))
        return one.flatten([dict(advantages=advantages, rollouts=rows)])

    def grads(tail, reference, params, settings, shift, **kw):
        data = batch(tail, shift, **kw)
        steps = one.TailSteps(tail, reference, params, settings)
        stats = one.parity_weights(steps, data, settings)
        stats.update(one.accumulate(steps, data))
        return [p.grad.numpy().copy() for p in params], stats, data

    base = dict(cfg, kl_beta=1e-3, entropy_beta=1e-3, tis=False)
    # rlvr_update.update_batch on the same batch, on a fresh copy of the adapted tail with no prefill
    # (its backward would also grad the prefill's aliases of the adapter buffers): gradients before the step
    fresh = tiny_model()
    _, params = one.attach_lora(fresh, cfg["rank"], cfg["alpha"])
    theirs = get_state_dict(loop.model)
    for name, value in get_state_dict(fresh).items():
        value.assign(Tensor(theirs[name].numpy())).realize()
    from tinygrad.llm.nemotron_h_sampler import NemotronHRolloutSampler
    sampler = NemotronHRolloutSampler(fresh, batch=loop.lanes, capacity=12, prefix_capacity=16, rows=loop.lanes,
                                      bias=loop.sampler.bias.numpy(), capture=len(fresh.blk) - 1)
    tail = one.Tail(sampler)
    reference = one.Tail(sampler, block=one.frozen_lora_tail(fresh.blk[-1]))
    mine, _, data = grads(tail, reference, params, base, 0.0)
    records, start = [], 0
    for episode, r in enumerate(rollouts):
        path, size = root / f"tis-{episode}.npz", len(r["tokens"])
        np.savez(path, hidden=r["hidden"], tokens=np.asarray(r["tokens"], dtype=np.int32),
                 old_logp=data["old_logp"][start:start + size])
        records.append(dict(feature=str(path), feature_sha256=file_sha256(path), episode=episode))
        start += size
    stepped = []

    class NoStep:
        def zero_grad(self):
            for p in params:
                p.grad = None

        def step(self):
            stepped.append([p.grad.numpy().copy() for p in params])

    policy = TailPolicy(fresh, loop.sampler.bias.numpy())
    update_batch(policy, TailReference(policy), NoStep(), params, [dict(records=records, advantages=advantages)],
                 dict(base, max_grad_norm=0.0))
    scale = max(float(np.abs(g).max()) for g in stepped[0])
    reference_gap = float(max(np.abs(a - b).max() for a, b in zip(mine, stepped[0]))) / scale
    plain, _, _ = grads(loop.policy, loop.reference, loop.params, dict(base, kl_beta=0.0, entropy_beta=0.0), 0.05)
    capped, stats, _ = grads(loop.policy, loop.reference, loop.params,
                             dict(base, kl_beta=0.0, entropy_beta=0.0, tis=True, tis_cap=1.02), 0.05)
    tis_gap = float(max(np.abs(c - 1.02 * p).max() for c, p in zip(capped, plain)))
    # kl_aggregation: an explicit legacy_token_mean is the recordless (runs 1-2) update bit for bit; with the
    # advantages zeroed (KL + entropy only, reference = the initial adapter) `matched` is legacy x T / N
    legacy, legacy_stats, data = grads(loop.policy, loop.reference, loop.params, base, 0.0, advantages=[0.0] * 4)
    named, _, _ = grads(loop.policy, loop.reference, loop.params, dict(base, kl_aggregation="legacy_token_mean"), 0.0,
                        advantages=[0.0] * 4)
    matched, matched_stats, _ = grads(loop.policy, loop.reference, loop.params, dict(base, kl_aggregation="matched"),
                                      0.0, advantages=[0.0] * 4)
    ratio = len(data["tokens"]) / data["trajectories"]
    return dict(reference_gap=reference_gap, tis_gap=tis_gap / max(float(np.abs(g).max()) for g in plain),
                tis_stats=stats, legacy_exact=all(np.array_equal(a, b) for a, b in zip(legacy, named)),
                matched_gap=max(float(np.abs(m - ratio * g).max() / np.abs(ratio * g).max()) for m, g in zip(matched, legacy)),
                kl_means=[legacy_stats["kl"], matched_stats["kl"]], **check_skip(loop, cfg, root))


def check_skip(loop, cfg, root: Path) -> dict:
    """An update the mask left empty is skipped (no Adam step); `skips_max` in a row stops the run cleanly."""
    import numpy as np
    from daycare.nursery import rloo_tinygrad as one
    before = [p.numpy().copy() for p in loop.params]
    record, out = dict(losses=[], config=cfg), root / "skips"
    out.mkdir()
    done = one.run(loop, [dict(id="a")], None, dict(cfg, updates=6, prompts_per_update=1, triggers=dict(skips_max=3)),
                   record, out, None, export=False, step=lambda picks: loop.learn([], [], {}, {}))
    return dict(skip_unchanged=all(np.array_equal(b, p.numpy()) for b, p in zip(before, loop.params)),
                skip_stopped=done["stopped"], skip_count=done["skipped_updates"], skip_updates=len(done["losses"]),
                skip_checkpoint=(out / "checkpoint-003.npz").exists())


@pytest.mark.skipif(not (EXP / "tinygrad/llm/nemotron_h_sampler.py").exists(), reason="tinygrad-arkey exp tree missing")
def test_two_one_stack_updates_step_in_place_and_hold_parity(tmp_path):
    env = dict(os.environ, DAYCARE_TRAIN_TINYGRAD_PATH=str(EXP), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env.setdefault("DEV", "CPU")
    done = subprocess.run([sys.executable, __file__, str(tmp_path)], env=env, capture_output=True, text=True,
                          timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    for result in report["updates"]:
        metrics = result["metrics"]
        assert metrics["stepped"] and all(v == v and abs(v) != float("inf")
                                          for v in (metrics["loss"], metrics["grad_norm"]))
        assert metrics["features"] == "captured" and metrics["tis"] and metrics["rho_clipped"] == 0
        # captured inputs through the sampler's own tail: the sampled values, bit for bit
        assert metrics["max_logprob_error"] == 0.0 and metrics["exact"] == metrics["checked"]
        assert abs(metrics["rho_mean"] - 1) < 1e-3
        assert metrics["trajectories"] == 8 and result["adapter_delta"] > 0
    assert report["adapter_effect"] > 1e-2
    export = report["export"]
    assert {"adapter-raw.npz", "adapter-raw.sha256", "adapter.xml", "adapter.gguf", "probe.npz", "run.xml"} <= set(export["files"])
    assert export["probe_exact"] and export["updates"] == 2 and export["trained_b_max"] > 0
    assert "adapter_sha256" in export["initial_export"]
    # the in-memory update reproduces rlvr_update.update_batch; TIS applies min(rho, C) to every token
    assert report["reference_gap"] < 1e-4 and report["tis_gap"] < 1e-5
    assert report["legacy_exact"] and report["matched_gap"] < 1e-4
    assert report["kl_means"][0] > 0 and abs(report["kl_means"][1] / report["kl_means"][0] - 1) < 1e-4  # token means
    assert report["skip_unchanged"] and report["skip_count"] == report["skip_updates"] == 3 and report["skip_checkpoint"]
    assert report["skip_stopped"]["update"] == 3 and "consecutive skipped" in report["skip_stopped"]["reason"]
    assert report["tis_stats"]["rho_clipped"] == 1.0 and abs(report["tis_stats"]["rho_mean"] - np.exp(0.05)) < 1e-4


if __name__ == "__main__":
    print(json.dumps(run_tiny(Path(sys.argv[1])), default=float))
