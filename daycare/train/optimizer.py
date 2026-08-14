"""The fast-weight lobe: per-turn, reward-gated, decaying weight change.

The primitive the design is premised on (research/state-vs-weights.md): the
model changes *from its own life*, turn by turn -- not only in a batch job. The
frozen substrate cannot provide this (it hands back text, not gradients), so the
lobe lives here, in the brain, over a small trainable adapter.

Two signals, both used:
  correction -> supervised step   (a target beats a scalar; prefer it when it exists)
  reward     -> policy gradient   (drives.reinforcer gates the update -- the lineage's
                                   mechanism: `loss = -advantage * logprob`)

And the load-bearing property: **fast weights fade**. If they persisted they would
just be slow weights with a worse optimizer, and they would drift. They hold the
recent past; sleep (batch consolidation) decides what earns promotion into the slow
weights. Prior art: Ba/Hinton, "Using Fast Weights to Attend to the Recent Past"
(2016), <https://arxiv.org/abs/1610.06258>.

The decision logic below is deliberately framework-free (plain floats) so it is
testable without a GPU; tinygrad is imported lazily, only where tensors are touched.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FastWeightConfig:
    """Bounds are the safety story -- every field here limits how much damage a
    single turn can do (the catastrophic-forgetting risk from research/catastrophic-forgetting.md)."""

    lr: float = 1e-3          # tiny: one turn should nudge, not reshape
    decay: float = 0.05       # per-turn pull toward zero -- the fade
    baseline_beta: float = 0.9  # EMA horizon for the reward baseline
    clip: float = 1.0         # advantage clip (bounds a surprising reward)
    min_signal: float = 0.05  # below this |advantage| there is no signal: skip


class RewardBaseline:
    """EMA baseline: turns a raw reinforcer into an advantage.

    Without it, single-sample policy gradient is pure variance -- every positive
    reward pushes, even when it was merely typical. The advantage asks the useful
    question instead: *better than usual?*
    """

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value = 0.0
        self.n = 0

    def advantage(self, reward: float) -> float:
        adv = reward - self.value
        self.value = self.beta * self.value + (1.0 - self.beta) * reward
        self.n += 1
        return adv


def clip(x: float, c: float) -> float:
    return max(-c, min(c, x))


def should_step(advantage: float, min_signal: float) -> bool:
    return abs(advantage) >= min_signal


def decayed(w, decay: float):
    """Pull toward zero. Works on floats, numpy arrays, or tensors alike."""
    return w * (1.0 - decay)


class FastWeightLobe:
    """Binds the policy above to real trainable params (e.g. lora.lora_params()).

    The base model stays frozen; only this small adapter moves, and it fades.
    """

    def __init__(self, params, config: FastWeightConfig | None = None):
        self.params = list(params)
        self.cfg = config or FastWeightConfig()
        self.baseline = RewardBaseline(self.cfg.baseline_beta)
        self._opt = None

    def _optimizer(self):
        if self._opt is None:
            from tinygrad.nn.optim import Adam  # lazy: keeps this module GPU-free to import

            self._opt = Adam(self.params, lr=self.cfg.lr)
        return self._opt

    @staticmethod
    def _train_ctx():
        # tinygrad's optimizer refuses to step unless Tensor.training is set. The
        # lobe owns the update, so it owns the context -- generation stays outside it.
        from tinygrad import Tensor

        return Tensor.train()

    def learn_from_reward(self, reward: float, logprob_fn) -> dict:
        """Policy gradient on the turn's reinforcer. `logprob_fn()` -> scalar Tensor."""
        adv = clip(self.baseline.advantage(reward), self.cfg.clip)
        if not should_step(adv, self.cfg.min_signal):
            return {"stepped": False, "advantage": adv}
        opt = self._optimizer()
        with self._train_ctx():
            opt.zero_grad()
            loss = -adv * logprob_fn()
            loss.backward()
            opt.step()
        return {"stepped": True, "advantage": adv, "loss": float(loss.item())}

    def learn_from_correction(self, loss_fn) -> dict:
        """Supervised step on a correction. `loss_fn()` -> scalar Tensor."""
        opt = self._optimizer()
        with self._train_ctx():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
        return {"stepped": True, "supervised": True, "loss": float(loss.item())}

    def fade(self) -> None:
        """Decay every fast weight toward zero. Call once per turn.

        Batched into a single realize: this runs every turn, and one dispatch per
        param (112 of them) made the decay cost more than the learning did.
        """
        from tinygrad import Tensor

        Tensor.realize(*[p.assign(decayed(p.detach(), self.cfg.decay)) for p in self.params])
