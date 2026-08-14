"""Teach a model its name into the weights (research/catastrophic-forgetting.md).

The end-to-end training path, tinygrad-native, no torch/transformers:

  load Qwen3-0.6B Q8 (fp16, via upstream Transformer) -> apply LoRA ->
  SFT on the name curriculum (cross-entropy on completion tokens) ->
  eval identity-consistency gate-OFF, before vs after.

The forward here is a cache-free training forward: it reuses the loaded block
weights (attn_q/k/v/output, norms, ffn_*, qk-norm, rope) but runs a plain causal
attention with no KV cache, so it is differentiable. Run:

  python -m daycare.nursery.consolidate
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daycare.nursery.trainer_env import use_train_tinygrad

use_train_tinygrad()  # put DayCare's train tinygrad on sys.path BEFORE importing it

from tinygrad import Tensor  # noqa: E402
from tinygrad.nn.optim import Adam  # noqa: E402

from daycare.model import qwen_forward as fwd  # noqa: E402
from daycare.nursery import curriculum, evaluate, lora as loramod  # noqa: E402
from daycare.artifact import save as artifact_save  # noqa: E402

GGUF = fwd.GGUF
MAXLEN = fwd.MAXLEN
# LoRA only the top-K layers: keeps the backward graph small (grads flow through K
# layers, not 28) and lets the frozen lower blocks be realized/chunked.
LORA_LAST_K = int(os.environ.get("DAYCARE_LORA_LAST_K", "8"))


# The forward now lives in daycare/model/qwen_forward.py so lesson mode can share
# it (research/fast-weight-lobe-scope.md F1). Thin aliases keep this module readable.
def logits_all(model, freqs, ids):
    return fwd.logits_all(model, freqs, ids, grad_last_k=LORA_LAST_K)


def build(tok, prompt, completion):
    return fwd.build(tok, prompt, completion)


def example_loss(model, freqs, p_ids, c_ids):
    return fwd.completion_loss(model, freqs, p_ids, c_ids, grad_last_k=LORA_LAST_K)


def greedy(model, tok, freqs, prompt, max_new: int = 24):
    return fwd.greedy(model, tok, freqs, prompt, max_new, grad_last_k=LORA_LAST_K)


def identity_rate(model, tok, freqs, name: str, prompts: list[str]) -> tuple[int, int, list[str]]:
    outs = [greedy(model, tok, freqs, p) for p in prompts]
    hits = sum(curriculum.scores_identity(name, o) for o in outs)
    return hits, len(prompts), outs


class _GreedySub:
    """Tiny shim so evaluate.forgetting_probe (which wants a `.generate(...)`
    substrate) can run against the in-training (model, tok, freqs) triple without
    loading a second model."""

    def __init__(self, model, tok, freqs):
        self.model, self.tok, self.freqs = model, tok, freqs

    def generate(self, prompt, temperature: float = 0.0, max_tokens: int = 16):
        return greedy(self.model, self.tok, self.freqs, prompt, max_tokens)


def main() -> None:
    name = os.environ.get("DAYCARE_MODEL_NAME", "Ada")
    epochs = int(os.environ.get("DAYCARE_EPOCHS", "6"))
    lr = float(os.environ.get("DAYCARE_LR", "1e-3"))
    r = int(os.environ.get("DAYCARE_LORA_R", "16"))
    eval_n = int(os.environ.get("DAYCARE_EVAL_N", "5"))

    model, tok, freqs = fwd.load(GGUF, 2048)

    # forward sanity before we train through it
    check = greedy(model, tok, freqs, "What is the capital of France? Answer in one word.", 12)
    print(f"[forward check] capital of France -> {check!r}")
    assert "paris" in check.lower(), "training forward looks wrong; aborting"

    eval_prompts = curriculum.eval_prompts(name)[:eval_n]
    b_hit, b_tot, b_out = identity_rate(model, tok, freqs, name, eval_prompts)
    print(f"[before] identity {b_hit}/{b_tot}  e.g. {b_out[0]!r}")
    # Measure forgetting BEFORE training too: without a baseline, "regressed" is an
    # assumption sitting inside a gate. Now the halt is evidence, not inference.
    bf_hit, bf_tot, _ = evaluate.forgetting_probe(_GreedySub(model, tok, freqs))
    print(f"[before] forgetting {bf_hit}/{bf_tot}  (the capability training must not cost)")

    loras = loramod.apply_lora(model, r=r, alpha=2 * r, last_k=LORA_LAST_K)
    params = loramod.lora_params(loras)
    opt = Adam(params, lr=lr)
    # mixed_curriculum, NOT train_examples: a single-intent curriculum is what
    # collapsed her (research/catastrophic-forgetting.md). Ballast + counter-examples
    # are the mitigation, and a mitigation that isn't called is decoration.
    ratio = int(os.environ.get("DAYCARE_BALLAST_RATIO", "4"))
    data = [build(tok, ex.prompt, ex.completion) for ex in curriculum.mixed_curriculum(name, ratio)]
    print(f"[train] {len(loras)} LoRA adapters, {len(params)} tensors, {len(data)} examples, "
          f"r={r} lr={lr} epochs={epochs}")

    gate_every = int(os.environ.get("DAYCARE_GATE_EVERY", "1"))
    forget_floor = int(os.environ.get("DAYCARE_FORGET_FLOOR", str(len(evaluate.PROBE) - 1)))
    sub = _GreedySub(model, tok, freqs)

    losses = []
    halted = False
    with Tensor.train():
        for epoch in range(epochs):
            total = 0.0
            for p_ids, c_ids in data:
                opt.zero_grad()
                loss = example_loss(model, freqs, p_ids, c_ids)
                loss.backward()
                opt.step()
                total += loss.item()
            losses.append(total / len(data))

            gate_msg = ""
            regressed = False
            if gate_every > 0 and (epoch + 1) % gate_every == 0:
                f_hit, f_tot, _ = evaluate.forgetting_probe(sub)
                gate_msg = f"  forgetting={f_hit}/{f_tot}"
                regressed = f_hit < forget_floor
            print(f"  epoch {epoch}  loss={losses[-1]:.4f}{gate_msg}")
            if regressed:
                print(f"[gate] HALTED at epoch {epoch}: general capability regressed "
                      f"({f_hit}/{f_tot} < floor {forget_floor}) -- training stopped early "
                      f"to avoid teaching the name at the cost of everything else.")
                halted = True
                break
    if halted:
        print("[gate] proceeding to final eval with the model as it stands.")

    a_hit, a_tot, a_out = identity_rate(model, tok, freqs, name, eval_prompts)
    print(f"[after]  identity {a_hit}/{a_tot}  e.g. {a_out[0]!r}")

    # Save the adapter as a flat <adapter> XML (no safetensors/JSON) -- the M2 win.
    tensors = {f"lora.{i}.{k}": getattr(lo, k).numpy()
               for i, lo in enumerate(loras) for k in ("A", "B")}
    meta = dict(subject=name, base="qwen3-0.6b-q8", r=r, alpha=2 * r, last_k=LORA_LAST_K,
                targets=list(loramod.TARGETS), epochs=epochs,
                loss_start=round(losses[0], 4), loss_end=round(losses[-1], 4),
                eval_metric="identity-consistency", eval_before=f"{b_hit}/{b_tot}",
                eval_after=f"{a_hit}/{a_tot}", eval_gate="off")
    out = f"state/adapters/{name.lower()}.adapter.xml"
    artifact_save.write_adapter(tensors, meta, out)
    print(f"[saved] {out}")
    print(f"\nRESULT: identity-consistency {b_hit}/{b_tot} -> {a_hit}/{a_tot} "
          f"({'LEARNED' if a_hit > b_hit else 'no gain'})")


if __name__ == "__main__":
    main()
