"""On-policy RLOO objectives (leave-one-out baseline, token policy loss, KL and entropy terms)."""
from __future__ import annotations
import numpy as np
from .trainer_env import use_train_tinygrad
use_train_tinygrad()
from tinygrad import Tensor


def leave_one_out(rewards):
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1 or len(rewards) < 2 or not np.isfinite(rewards).all():
        raise ValueError('RLOO requires at least two finite scalar rewards')
    return rewards - (rewards.sum() - rewards) / (len(rewards) - 1)


def policy_loss(logits, actions, rewards, reference=None, beta=0.01):
    """Categorical actions sampled from these current logits, exactly one update.

    Environment outcomes and sampled indices are detached. No accepted-answer
    target or cross-entropy imitation loss is supplied to the optimizer.
    """
    actions = np.asarray(actions, dtype=np.int32)
    advantages = leave_one_out(rewards)
    if actions.shape != advantages.shape or np.any(actions < 0) or np.any(actions >= logits.shape[0]):
        raise ValueError('sampled actions do not match rewards or action space')
    logp = logits.log_softmax()
    weights = np.bincount(actions, weights=advantages, minlength=logits.shape[0]) / len(actions)
    loss = -(logp * Tensor(weights.astype(np.float32), device=logits.device)).sum()
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float32)
        if ref.shape != logits.shape or not np.isfinite(ref).all():
            raise ValueError('invalid frozen reference log probabilities')
        loss = loss + beta * (logp.exp() * (logp - Tensor(ref, device=logits.device))).sum()
    return loss


def sample_actions(logits, rng, count):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or count < 2:
        raise ValueError('invalid policy or sample count')
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return rng.choice(len(values), size=count, p=probabilities), probabilities


def token_policy_loss(selected_logprobs, advantage, group_size):
    """One freshly sampled trajectory (or chunk), model-controlled tokens only.

    Sum token log probabilities: this is sequence-level RLOO, not token-mean
    imitation. Sum chunk gradients before the group's single optimizer step.
    """
    if group_size < 2 or not np.isfinite(advantage):
        raise ValueError('invalid trajectory advantage or group size')
    return -float(advantage) * selected_logprobs.sum() / group_size


def step_policy(optimizer, params, max_norm=1.):
    """Apply one finite accumulated policy gradient in the backend's train mode."""
    if not np.isfinite(max_norm) or max_norm<0:raise ValueError('nonnegative gradient limit required')
    norm=float(sum(p.grad.square().sum() for p in params).sqrt().item())
    if not np.isfinite(norm):raise ValueError('non-finite policy gradient')
    with Tensor.train():
        if max_norm and norm>max_norm:
            for p in params:p.grad=p.grad*(max_norm/norm)
        optimizer.step()
    return norm


# How the KL and entropy terms aggregate over tokens (research/rloo-posttool-audit-20260927.md, D1).
# `legacy_token_mean` (post-tool runs 1 and 2): a mean over the batch's T tokens, so per token ~T/N weaker than the
# policy gradient's sum over tokens / N trajectories. `matched`: the policy gradient's own aggregation (sum / N), as
# verl aggregates KL and entropy with the policy loss's loss_agg_mode (verl/workers/utils/losses.py, ppo_loss).
# A record without the field ran `legacy_token_mean`.
KL_AGGREGATIONS = ('legacy_token_mean', 'matched')


def regularizer_denominator(policy, token_count, trajectory_count):
    if policy not in KL_AGGREGATIONS:
        raise ValueError(f'unknown KL aggregation {policy}; one of {KL_AGGREGATIONS}')
    return token_count if policy == 'legacy_token_mean' else trajectory_count


def regularized_token_loss(logprobs, tokens, advantage, trajectory_count, reference_logprobs, token_count, *,
                           kl_beta=0., entropy_beta=0., kl_aggregation='legacy_token_mean'):
    """Fresh on-policy sequence RLOO + KL minus entropy, aggregated per `kl_aggregation`.

    Input rows contain model-controlled positions only. The caller uses the
    SAME batch denominators for every chunk. The reference is the fixed SFT
    policy, detached from gradients, not the most recent rollout policy.
    No importance ratio is needed for one update of a freshly sampled batch.
    This explicit reduction is our adaptation, not an inferred paper default.
    """
    if (token_count < 1 or trajectory_count < 2 or
            not np.isfinite([kl_beta, entropy_beta, advantage]).all() or
            min(kl_beta, entropy_beta) < 0):
        raise ValueError('invalid regularized RLOO configuration')
    if len(logprobs.shape) != 2 or tokens.shape != (logprobs.shape[0],):
        raise ValueError('expected token-by-vocabulary log probabilities')
    selected = logprobs.gather(1, tokens.reshape(-1, 1))
    pg = token_policy_loss(selected, advantage, trajectory_count)
    denominator = regularizer_denominator(kl_aggregation, token_count, trajectory_count)
    entropy = -(logprobs.exp()*logprobs).sum()/denominator
    if kl_beta:
        if reference_logprobs is None or reference_logprobs.shape != logprobs.shape:
            raise ValueError('KL requires matching frozen reference log probabilities')
        kl = (logprobs.exp()*(logprobs-reference_logprobs.detach())).sum()/denominator
    else:
        kl = logprobs.sum()*0
    return pg + kl_beta*kl - entropy_beta*entropy, (pg, kl, entropy)
