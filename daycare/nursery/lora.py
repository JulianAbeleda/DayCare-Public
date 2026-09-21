"""LoRA for tinygrad Linear layers (research/name-learning-scope.md).

The trainable delta on a frozen base: y = x@W + (x@A)@B * (alpha/r), with W frozen.
A is small-random, B is zero -> the initial delta is zero, so training starts from
the exact base model. ~15 lines of real logic; no peft.
"""
from __future__ import annotations

from .trainer_env import use_train_tinygrad

use_train_tinygrad()  # self-sufficient: never depend on the importer's order

from tinygrad import Tensor  # noqa: E402
from tinygrad.nn.state import get_parameters  # noqa: E402


class LoRALinear:
    """Wrap a frozen nn.Linear with a trainable low-rank delta."""

    def __init__(self, base, r: int = 16, alpha: int = 32):
        self.base = base  # nn.Linear, weight frozen by freeze_model()
        out_f, in_f = base.weight.shape
        self.A = (Tensor.randn(r, in_f) * (1.0 / in_f) ** 0.5)  # small random
        self.B = Tensor.zeros(out_f, r)                          # zero -> delta=0 at start
        self.scale = alpha / r

    def __call__(self, x: Tensor) -> Tensor:
        return self.base(x) + ((x @ self.A.transpose()) @ self.B.transpose()) * self.scale

    @property
    def params(self) -> list[Tensor]:
        return [self.A, self.B]

    def merged_weight(self) -> Tensor:
        """base W + B@A*scale, for exporting a plain Linear (fp)."""
        return self.base.weight + (self.B @ self.A) * self.scale


# The normalized tinygrad target set for an ordinary decoder block.  These are
# backend surface names, not model-family names; unsupported/fused topologies
# fail when no compatible targets can be discovered.
TARGETS = (
    "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
    # Nemotron-H Mamba-2's two dense projections. Convolution, decay, skip,
    # and normalization parameters remain frozen and keep their exact native
    # shapes; these linears have the same explicit merge contract as every
    # other LoRA target.
    "ssm_in", "ssm_out",
)


def freeze_model(model) -> None:
    """Mark every existing parameter non-trainable (so only LoRA A/B get grads)."""
    for p in get_parameters(model):
        p.is_param_(False)


def apply_lora(model, r: int = 16, alpha: int = 32, targets=TARGETS,
               last_k: int | None = None) -> list[LoRALinear]:
    """Freeze the model, wrap each target Linear, return the adapters.

    If last_k is set, only the last k blocks get adapters (smaller backward graph;
    identity is high-level so top-layer adapters suffice).
    """
    freeze_model(model)
    if r <= 0:
        raise ValueError("adapter rank must be positive")
    if last_k is not None and (last_k <= 0 or last_k > len(model.blk)):
        raise ValueError(f"last_k must be between 1 and {len(model.blk)}, or None")
    loras: list[LoRALinear] = []
    first = 0 if last_k is None else len(model.blk) - last_k
    for layer_index, block in enumerate(model.blk):
        if layer_index < first:
            continue
        for name in targets:
            base = getattr(block, name, None)
            if base is None or not hasattr(base, "weight"):
                continue
            lora = LoRALinear(base, r, alpha)
            # Persist the exact merge destination.  Older adapters inferred this
            # from list order; that was compact but architecture-coupled.
            lora.layer_index = layer_index
            lora.target_name = name
            lora.tensor_name = f"blk.{layer_index}.{name}.weight"
            lora.base_shape = tuple(int(v) for v in base.weight.shape)
            setattr(block, name, lora)  # transparent: block calls getattr(name)(x)
            loras.append(lora)
    if not loras:
        raise ValueError(
            "model exposes no compatible attention/feed-forward linear targets; "
            "add a training backend for this topology"
        )
    return loras


def lora_params(loras: list[LoRALinear]) -> list[Tensor]:
    return [p for lora in loras for p in lora.params]


def target_map(loras: list[LoRALinear]) -> list[dict]:
    """Serializable adapter-index to exact base-tensor mapping."""
    return [
        {
            "index": index,
            "tensor": lora.tensor_name,
            "shape": lora.base_shape,
            "layer": lora.layer_index,
            "target": lora.target_name,
        }
        for index, lora in enumerate(loras)
    ]
