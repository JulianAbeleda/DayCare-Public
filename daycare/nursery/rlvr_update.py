"""Native frozen-tail reference and batched, replay-checked RLOO updates."""
import copy
from pathlib import Path
import numpy as np
from .rlvr import regularized_token_loss, step_policy
from .provenance import file_sha256
from .lora import LoRALinear
from tinygrad import Tensor


def frozen_lora_tail(block):
    """Share frozen base tensors; copy mutable adapters into a fixed reference."""
    result = copy.copy(block)
    found = False
    for name, layer in vars(block).items():
        if isinstance(layer, LoRALinear):
            found = True
            clone = copy.copy(layer)
            clone.A = Tensor(layer.A.numpy().copy(), device=layer.A.device).is_param_(False)
            clone.B = Tensor(layer.B.numpy().copy(), device=layer.B.device).is_param_(False)
            setattr(result, name, clone)
    if not found:
        raise ValueError('reference requires LoRA in the tokenwise final block')
    return result


class TailReference:
    def __init__(self, sampler):
        self.block = frozen_lora_tail(sampler.model.blk[-1])
        self.output, self.norm = sampler.model.output, sampler.model.output_norm
        self.bias = sampler.bias.copy()

    def logprobs(self, hidden, temperature):
        logits = self.output(self.norm(self.block(hidden.float()))).float()
        return ((logits+Tensor(self.bias, device=logits.device))/temperature).log_softmax().detach()


def update_batch(sampler, reference, optimizer, params, groups, cfg):
    """Collect all groups first; check replay; accumulate once; mutate once.

    Each group supplies records, advantages and a full reward group. Features
    store only generated positions, so observations cannot enter this loss.
    All-zero reward groups still contribute regularization if enabled.
    """
    kl_beta, entropy_beta = cfg.get('kl_beta', 0.), cfg.get('entropy_beta', 0.)
    chunks, max_error, token_count, unchecked = [], 0., 0, 0
    trajectories = sum(len(g['advantages']) for g in groups)
    for group in groups:
        for record in group['records']:
            if file_sha256(Path(record['feature'])) != record['feature_sha256']:
                raise ValueError('rollout feature hash mismatch')
            with np.load(record['feature']) as f:
                size = len(f['tokens'])
                if size == 0 or len(f['hidden']) != size or len(f['old_logp']) != size:
                    raise ValueError('rollout feature lengths differ')
                token_count += size
                for start in range(0, size, 16):
                    h = Tensor(f['hidden'][start:start+16]).unsqueeze(0)
                    tokens = Tensor(f['tokens'][start:start+16])
                    lp = sampler.logprobs(h, cfg['temperature'])[0]
                    selected = lp.gather(1, tokens.reshape(-1, 1)).numpy().ravel()
                    recorded = f['old_logp'][start:start+16]
                    # NaN marks a position the sampler reported no probability for
                    # (llama.cpp: part of a multi-byte character); it is recomputed, unchecked.
                    checked = ~np.isnan(recorded)
                    unchecked += int((~checked).sum())
                    if not checked.any():
                        continue
                    error = float(np.max(np.abs(selected[checked]-recorded[checked])))
                    if not np.isfinite(error) or error > cfg['logprob_tolerance']:
                        raise ValueError(f'rollout/trainer logprob mismatch {error}')
                    max_error = max(max_error, error)
            chunks.append((record, float(group['advantages'][record['episode']])))
    stepped = bool(chunks and (kl_beta or entropy_beta or any(a for _, a in chunks)))
    if kl_beta and reference is None:
        raise ValueError('missing SFT reference policy')
    optimizer.zero_grad()
    metrics = np.zeros(4)
    if stepped:
        for record, advantage in chunks:
            with np.load(record['feature']) as f:
                for start in range(0, len(f['tokens']), 16):
                    h = Tensor(f['hidden'][start:start+16]).unsqueeze(0)
                    tokens = Tensor(f['tokens'][start:start+16])
                    ref = reference.logprobs(h, cfg['temperature'])[0].realize() if kl_beta else None
                    with Tensor.train():
                        lp = sampler.logprobs(h, cfg['temperature'])[0]
                        loss, terms = regularized_token_loss(lp, tokens, advantage, trajectories,
                            ref, token_count, kl_beta=kl_beta, entropy_beta=entropy_beta,
                            kl_aggregation=cfg.get('kl_aggregation', 'legacy_token_mean'))
                        values = np.array([float(x.item()) for x in (loss, *terms)])
                        if not np.isfinite(values).all():
                            raise ValueError('nonfinite RL objective')
                        loss.backward()
                        for p in params:
                            p.grad.realize()
                    metrics += values
        norm = step_policy(optimizer, params, cfg.get('max_grad_norm', 1.))
    else:
        norm = 0.
    return dict(loss=float(metrics[0]), policy_loss=float(metrics[1]),
                kl=float(metrics[2]), entropy=float(metrics[3]), grad_norm=norm,
                stepped=stepped, max_logprob_error=max_error, unchecked_logprobs=unchecked, generated_tokens=token_count,
                trajectories=trajectories)
