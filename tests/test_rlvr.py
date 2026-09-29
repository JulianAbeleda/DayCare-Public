import os

import pytest

from daycare.nursery import trainer_env

if not trainer_env.trainer_available():
    pytest.skip("trainer tinygrad not installed (setup_trainer.sh or DAYCARE_TRAIN_TINYGRAD_PATH)",
                allow_module_level=True)

import numpy as np
from daycare.nursery.rlvr import leave_one_out, policy_loss, sample_actions
from tinygrad import Tensor


def test_rloo_baseline_and_no_signal():
    np.testing.assert_allclose(leave_one_out([1, 0, 0, 1]), [2/3, -2/3, -2/3, 2/3])
    np.testing.assert_array_equal(leave_one_out([1, 1, 1]), [0, 0, 0])
    with pytest.raises(ValueError): leave_one_out([1])


def test_reward_moves_policy_toward_success_not_failure():
    logits = Tensor([0., 0.]).is_param_(True)
    with Tensor.train():
        loss = policy_loss(logits, [0, 1], [1., 0.])
        loss.backward()
    gradient = logits.grad.numpy()
    assert gradient[0] < 0 < gradient[1]
    # Zero reward variance gives no policy gradient, rather than imitation.
    zeros = Tensor([0., 0.]).is_param_(True)
    with Tensor.train(): policy_loss(zeros, [0, 1], [0., 0.]).backward()
    np.testing.assert_array_equal(zeros.grad.numpy(), [0., 0.])


def test_sampling_and_exact_expected_gradient():
    actions, probabilities = sample_actions([0., np.log(3)], np.random.default_rng(1), 4000)
    np.testing.assert_allclose(probabilities, [.25, .75])
    assert abs((actions == 1).mean() - .75) < .03
    # Enumerate every two-sample group: expected RLOO gradient equals -d E[r].
    expected = np.zeros(2)
    for i in range(2):
        for j in range(2):
            logits = Tensor([0., float(np.log(3))]).is_param_(True)
            with Tensor.train(): policy_loss(logits, [i, j], [float(i == 0), float(j == 0)]).backward()
            expected += probabilities[i]*probabilities[j]*logits.grad.numpy()
    np.testing.assert_allclose(expected, [-.1875, .1875], atol=1e-6)


def test_chunked_sequence_loss_accumulates_same_gradient():
    from daycare.nursery.rlvr import token_policy_loss
    full=Tensor([-.2,-.3,-.5]).is_param_(True)
    with Tensor.train():token_policy_loss(full,2.,4).backward()
    chunks=Tensor([-.2,-.3,-.5]).is_param_(True)
    with Tensor.train():
        token_policy_loss(chunks[:1],2.,4).backward()
        token_policy_loss(chunks[1:],2.,4).backward()
    np.testing.assert_allclose(full.grad.numpy(),chunks.grad.numpy())
    np.testing.assert_allclose(full.grad.numpy(),[-.5,-.5,-.5])


def test_native_token_reward_updates_lora_and_preserves_base():
    from tinygrad.nn import Linear
    from tinygrad.nn.optim import Adam
    from daycare.nursery.lora import LoRALinear
    from daycare.nursery.rlvr import token_policy_loss,step_policy
    Tensor.manual_seed(31)
    base=Linear(4,2,bias=False)
    base.weight.is_param_(False)
    original=base.weight.numpy().copy()
    layer=LoRALinear(base,r=2,alpha=4)
    for p in layer.params:p.is_param_(True)
    optimizer=Adam(layer.params,lr=.01)
    hidden=Tensor([[1.,-.5,.3,2.]])
    before=layer(hidden).log_softmax().numpy().copy()
    optimizer.zero_grad()
    for token,advantage in zip([0,1],leave_one_out([1.,0.])):
        with Tensor.train():
            token_policy_loss(layer(hidden).log_softmax()[0,token],advantage,2).backward()
            for p in layer.params:p.grad.realize()
    # Caller is in inference mode; the update owner enables train mode locally.
    assert not Tensor.training
    assert step_policy(optimizer,layer.params)>0
    assert not Tensor.training
    after=layer(hidden).log_softmax().numpy()
    assert after[0,0]>before[0,0] and after[0,1]<before[0,1]
    assert np.any(layer.B.numpy()!=0)
    np.testing.assert_array_equal(base.weight.numpy(),original)
