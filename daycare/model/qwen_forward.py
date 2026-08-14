"""The cache-free, trainable Qwen3 forward -- shared by the trainer and lesson mode.

Upstream tinygrad's `Transformer` is built for inference: each block is wrapped in
`@function(precompile=True)` around a stateful KV-cache store, which is not a path
gradients flow through. So we reuse the *loaded weights* (attn_q/k/v/output, norms,
ffn_*, qk-norm, rope) but run a plain causal attention with no KV cache. That makes
it differentiable, which is what both callers need:

  - nursery/consolidate.py  -- batch SFT ("sleep")
  - substrate/tinygrad_train.py -- lesson mode, where the fast-weight lobe learns
                                   per turn (research/fast-weight-lobe-scope.md)

Two non-obvious rules live here, both learned the hard way (see
research/name-learning-scope.md operational notes):

  * MAXLEN padding -- eager tinygrad recompiles per unique shape, so every forward
    is right-padded to a constant length and the 28-layer graph compiles ONCE.
    Right-padding is safe: causal masking means real positions never attend to it.
  * grad_last_k -- realize the frozen lower blocks (no gradient needed there) to
    chunk the graph and dodge the mega-graph scheduler stall, but leave the top-K
    un-realized so gradients still reach their adapters. Realizing a grad-bearing
    block severs the graph (grad -> None).
"""
from __future__ import annotations

import os

from ..nursery.trainer_env import use_train_tinygrad

use_train_tinygrad()  # train tinygrad on sys.path BEFORE importing it

from tinygrad import Tensor  # noqa: E402
from tinygrad.llm.cli import SimpleTokenizer  # noqa: E402
from tinygrad.llm.model import Transformer, apply_rope, precompute_freqs_cis  # noqa: E402

GGUF = os.environ.get("DAYCARE_MODEL_GGUF", "/home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf")
MAXLEN = int(os.environ.get("DAYCARE_MAXLEN", "64"))


def load(gguf_path: str = GGUF, max_context: int = 2048):
    """-> (model, tokenizer, freqs). The fp, differentiable side of the house."""
    model, kv = Transformer.from_gguf(gguf_path, max_context)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    cfg = model.blk[0].config
    freqs = precompute_freqs_cis(cfg.rope_dim, cfg.max_context or max_context, cfg.rope_theta)
    return model, tok, freqs


def block_forward(block, x: Tensor, freqs: Tensor) -> Tensor:
    cfg = block.config
    B, T, _ = x.shape
    h = block.attn_norm(x)
    q, k, v = block.attn_q(h), block.attn_k(h), block.attn_v(h)
    if cfg.qk_norm and cfg.qk_norm != cfg.head_dim:
        q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    q = q.reshape(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    if cfg.qk_norm == cfg.head_dim:
        q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    q = apply_rope(q[..., : cfg.rope_dim], freqs[:T]).cat(q[..., cfg.rope_dim :], dim=-1)
    k = apply_rope(k[..., : cfg.rope_dim], freqs[:T]).cat(k[..., cfg.rope_dim :], dim=-1)
    mask = Tensor.full((1, 1, T, T), float("-inf"), dtype=x.dtype, device=x.device).triu(1)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    x = x + block.attn_output(attn)
    hf = block.ffn_norm(x)
    return x + block.ffn_down(block.ffn_gate(hf).silu() * block.ffn_up(hf))


def logits_all(model, freqs, ids: list[int], grad_last_k: int = 0) -> Tensor:
    """Raw logits at ALL positions, padded to MAXLEN.

    grad_last_k: how many top blocks to leave un-realized so gradients reach their
    adapters. 0 (default) realizes every block -- pure inference, cheapest graph.
    """
    assert len(ids) <= MAXLEN, f"sequence {len(ids)} exceeds MAXLEN {MAXLEN}"
    padded = ids + [0] * (MAXLEN - len(ids))
    x = model.token_embd(Tensor([padded])).float()
    n = len(model.blk)
    for i, block in enumerate(model.blk):
        x = block_forward(block, x, freqs)
        if i < n - grad_last_k:
            x = x.realize()
    return model.output(model.output_norm(x))          # (1, MAXLEN, vocab)


def prompt_ids(tok, prompt: str) -> list[int]:
    return (
        tok.prefix()
        + tok.role("user")
        + tok.encode(prompt + "\n/no_think")
        + tok.end_turn()
        + tok.role("assistant")
    )


def build(tok, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    return prompt_ids(tok, prompt), tok.encode(completion) + tok.end_turn()


IGNORE = -1


def completion_loss(model, freqs, p_ids: list[int], c_ids: list[int], grad_last_k: int = 0) -> Tensor:
    """Cross-entropy over the completion tokens only -- at a FIXED shape.

    We mask rather than slice. Slicing `logits[p-1:len(seq)-1]` makes the loss (and
    therefore the whole backward graph) a different shape for every distinct
    completion length, and tinygrad re-schedules per shape -- which is minutes each.
    An identity-only curriculum hid this because every completion was ~the same
    length; a mixed curriculum with ballast has ~12 distinct lengths and it stalls.

    So: score all MAXLEN-1 positions and set every non-completion target to
    IGNORE. Same graph for every example -> scheduled and compiled ONCE.
    """
    seq = p_ids + c_ids
    logits = logits_all(model, freqs, seq, grad_last_k)[0]      # (MAXLEN, vocab)
    # position i predicts token i+1, so the completion tokens seq[p:] are predicted
    # from positions p-1 .. len(seq)-2.
    targets = [IGNORE] * (MAXLEN - 1)
    for i, tok_id in enumerate(c_ids):
        targets[len(p_ids) - 1 + i] = tok_id
    # reduction="sum" then divide by the real token count: tinygrad's "mean" divides
    # by ALL MAXLEN-1 positions, not just the non-ignored ones, which would scale the
    # gradient by (completion_len / MAXLEN-1) -- silently under-weighting short
    # completions. The divisor is a Tensor, not a python float, so its value lives in
    # a buffer and the graph shape stays identical (a baked-in constant would
    # re-trigger the per-length re-schedule this whole change exists to remove).
    total = logits[: MAXLEN - 1].sparse_categorical_crossentropy(
        Tensor(targets), ignore_index=IGNORE, reduction="sum")
    # reshape(()) -> scalar: backward requires a scalar, and total/Tensor([n]) is (1,).
    return (total / Tensor([float(len(c_ids))])).reshape(())


def greedy(model, tok, freqs, prompt: str, max_new: int = 24, grad_last_k: int = 0) -> str:
    ids = prompt_ids(tok, prompt)
    start = len(ids)
    for _ in range(max_new):
        if len(ids) >= MAXLEN:
            break
        nxt = int(logits_all(model, freqs, ids, grad_last_k)[0, len(ids) - 1].argmax().item())
        if tok.is_end(nxt):
            break
        ids.append(nxt)
    return tok.decode(ids[start:])
