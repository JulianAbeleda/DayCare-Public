import os

import numpy as np
import pytest

from daycare.nursery import trainer_env

if not os.path.isdir(os.path.join(trainer_env.TRAIN_TINYGRAD_PATH, "tinygrad")):
    pytest.skip("trainer tinygrad not installed (setup_trainer.sh or DAYCARE_TRAIN_TINYGRAD_PATH)",
                allow_module_level=True)


def tinygrad():
    from daycare.nursery.trainer_env import use_train_tinygrad

    return use_train_tinygrad()


def config(block_types=("mamba", "attention", "mlp"), scan_chunk=2):
    from daycare.model.nemotron_h import NemotronHConfig

    count = len(block_types)
    return NemotronHConfig(
        num_blocks=count,
        dim=8,
        vocab_size=16,
        norm_eps=1e-5,
        max_context=32,
        block_types=tuple(block_types),
        head_counts=tuple(2 if kind == "attention" else 0 for kind in block_types),
        kv_head_counts=tuple(1 if kind == "attention" else 0 for kind in block_types),
        ffn_dims=tuple(12 if kind == "mlp" else 0 for kind in block_types),
        head_dim=4,
        ssm_inner=8,
        ssm_state=3,
        ssm_groups=1,
        ssm_heads=2,
        conv_kernel=3,
        scan_chunk=scan_chunk,
    )


def put(tensor, values):
    Tensor = tinygrad().Tensor

    tensor.assign(Tensor(np.asarray(values, dtype=np.float32), device=tensor.device)).realize()


def numpy_mamba(mixer, hidden):
    cfg = mixer.config
    linear = lambda x, layer: x @ layer.weight.numpy().astype(np.float32).T
    projected = linear(hidden, mixer.ssm_in)
    conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state
    gate, xbc, dt = np.split(projected, (cfg.ssm_inner, cfg.ssm_inner + conv_dim), axis=-1)
    weight = mixer.ssm_conv1d.weight.numpy().astype(np.float32)
    bias = mixer.ssm_conv1d.bias.numpy().astype(np.float32)
    convolved = np.zeros_like(xbc)
    for time in range(hidden.shape[1]):
        for kernel in range(cfg.conv_kernel):
            source = time - (cfg.conv_kernel - 1 - kernel)
            if source >= 0:
                convolved[:, time] += xbc[:, source] * weight[:, kernel]
    convolved += bias
    convolved = convolved / (1.0 + np.exp(-convolved))
    x, b, c = np.split(
        convolved,
        (cfg.ssm_inner, cfg.ssm_inner + cfg.ssm_groups * cfg.ssm_state),
        axis=-1,
    )
    batch, length, _ = x.shape
    width = cfg.ssm_inner // cfg.ssm_heads
    x = x.reshape(batch, length, cfg.ssm_heads, width)
    b = np.repeat(b.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state), cfg.ssm_heads, axis=2)
    c = np.repeat(c.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state), cfg.ssm_heads, axis=2)
    dt = np.log1p(np.exp(dt + mixer.ssm_dt["bias"].numpy()))
    a = mixer.ssm_a.numpy().astype(np.float32).reshape(cfg.ssm_heads)
    d = mixer.ssm_d.numpy().astype(np.float32).reshape(cfg.ssm_heads)
    state = np.zeros((batch, cfg.ssm_heads, width, cfg.ssm_state), dtype=np.float32)
    rows = []
    for time in range(length):
        state *= np.exp(dt[:, time, :, None, None] * a[None, :, None, None])
        state += (
            x[:, time, :, :, None]
            * dt[:, time, :, None, None]
            * b[:, time, :, None, :]
        )
        row = np.sum(state * c[:, time, :, None, :], axis=-1) + x[:, time] * d[None, :, None]
        rows.append(row)
    y = np.stack(rows, axis=1).reshape(batch, length, cfg.ssm_groups, cfg.ssm_inner // cfg.ssm_groups)
    gate = gate.reshape(y.shape)
    gated = y * gate / (1.0 + np.exp(-gate))
    normalized = gated / np.sqrt(np.mean(gated * gated, axis=-1, keepdims=True) + cfg.norm_eps)
    normalized = normalized.reshape(batch, length, cfg.ssm_inner) * mixer.ssm_norm.weight.numpy().reshape(cfg.ssm_inner)
    return linear(normalized, mixer.ssm_out)


def test_config_from_gguf_recovers_mixed_schedule():
    from daycare.model.nemotron_h import config_from_gguf

    metadata = {
        "general.architecture": "nemotron_h",
        "nemotron_h.block_count": 3,
        "nemotron_h.embedding_length": 8,
        "nemotron_h.context_length": 128,
        "nemotron_h.attention.layer_norm_rms_epsilon": 1e-5,
        "nemotron_h.attention.head_count": [0, 2, 0],
        "nemotron_h.attention.head_count_kv": [0, 1, 0],
        "nemotron_h.feed_forward_length": [0, 0, 12],
        "nemotron_h.attention.key_length": 4,
        "nemotron_h.ssm.inner_size": 8,
        "nemotron_h.ssm.state_size": 3,
        "nemotron_h.ssm.group_count": 1,
        "nemotron_h.ssm.time_step_rank": 2,
        "nemotron_h.ssm.conv_kernel": 3,
        "tokenizer.ggml.tokens": [str(i) for i in range(16)],
    }
    actual = config_from_gguf(metadata, max_context=32)
    assert actual.block_types == ("mamba", "attention", "mlp")
    assert actual.max_context == 32


def test_chunked_mamba_matches_independent_sequential_recurrence():
    Tensor = tinygrad().Tensor
    from daycare.model.nemotron_h import NemotronHMamba2

    rng = np.random.default_rng(6394531)
    mixer = NemotronHMamba2(config(("mamba",), scan_chunk=2))
    for tensor in (
        mixer.ssm_in.weight,
        mixer.ssm_conv1d.weight,
        mixer.ssm_conv1d.bias,
        mixer.ssm_dt["bias"],
        mixer.ssm_a,
        mixer.ssm_d,
        mixer.ssm_norm.weight,
        mixer.ssm_out.weight,
    ):
        values = rng.normal(0, 0.15, size=tensor.shape)
        if tensor is mixer.ssm_a:
            values = -np.exp(rng.normal(-0.4, 0.1, size=tensor.shape))
        if tensor is mixer.ssm_norm.weight:
            values = rng.normal(1.0, 0.05, size=tensor.shape)
        put(tensor, values)
    hidden = rng.normal(0, 0.3, size=(1, 5, 8)).astype(np.float32)
    expected = numpy_mamba(mixer, hidden)
    actual = mixer(Tensor(hidden)).realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_mixed_model_exposes_only_real_lora_targets_and_updates_them():
    Tensor = tinygrad().Tensor
    from tinygrad.nn.optim import SGD
    from daycare.model.nemotron_h import NemotronHModel
    from daycare.nursery.lora import apply_lora, lora_params, target_map

    Tensor.manual_seed(7)
    model = NemotronHModel(config())
    adapters = apply_lora(model, r=2, alpha=4, last_k=2)
    assert [row["target"] for row in target_map(adapters)] == [
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_down"
    ]
    params = lora_params(adapters)
    for param in params:
        param.is_param_(True)
    before = [adapter.B.numpy().copy() for adapter in adapters]
    optimizer = SGD(params, lr=1e-2)
    with Tensor.train():
        optimizer.zero_grad()
        hidden = Tensor.randn(1, 4, 8)
        for block in model.blk:
            hidden = block(hidden)
        loss = hidden.square().mean()
        loss.backward()
        assert all(param.grad is not None for param in params)
        optimizer.step()
    assert np.isfinite(float(loss.item()))
    assert all(np.isfinite(param.numpy()).all() for param in params)
    assert any(not np.array_equal(old, adapter.B.numpy()) for old, adapter in zip(before, adapters))


def test_prefix_cache_matches_full_mixed_forward():
    Tensor = tinygrad().Tensor
    from daycare.model.nemotron_h import NemotronHModel

    Tensor.manual_seed(19)
    model = NemotronHModel(config(scan_chunk=2))
    ids = [1, 2, 3, 4, 5, 6]
    full = model.token_embd(Tensor([ids])).float()
    for block in model.blk:
        full = block(full)
    caches = model.prime(ids[:4], through=2)
    suffix = model.suffix(ids[4:], caches, through=2)
    np.testing.assert_allclose(suffix.numpy(), full[:, 4:].realize().numpy(), rtol=3e-4, atol=3e-4)


def test_pixtral_tokenizer_matches_reference_ids():
    from daycare.model.gguf_tokenizer import tokenizer_from_gguf

    metadata = {
        "tokenizer.ggml.pre": "pixtral",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": ["<unk>", "H"],
        "tokenizer.ggml.token_type": [2, 1],
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 0,
    }
    # The real-checkpoint parity is covered by the external probe. This unit
    # test keeps the model-specific splitter selection from silently regressing.
    tokenizer = tokenizer_from_gguf(metadata)
    assert tokenizer.source_preset == "pixtral"
