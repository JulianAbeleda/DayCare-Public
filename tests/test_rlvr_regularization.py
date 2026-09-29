import os

import pytest

from daycare.nursery import trainer_env

if not os.path.isdir(os.path.join(trainer_env.TRAIN_TINYGRAD_PATH, "tinygrad")):
    pytest.skip("trainer tinygrad not installed (setup_trainer.sh or DAYCARE_TRAIN_TINYGRAD_PATH)",
                allow_module_level=True)

from types import SimpleNamespace
import numpy as np
from daycare.nursery.rlvr import regularized_token_loss
from daycare.nursery.rlvr_update import frozen_lora_tail, update_batch
from daycare.nursery.provenance import file_sha256
from daycare.nursery.lora import LoRALinear
from tinygrad import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam


def test_exact_kl_entropy_and_reference_has_no_gradient():
    p = np.array([[.8,.2],[.4,.6]], dtype=np.float32)
    q = np.array([[.5,.5],[.3,.7]], dtype=np.float32)
    logits = Tensor(np.log(p)).is_param_(True)
    ref = Tensor(np.log(q)).is_param_(True)
    with Tensor.train():
        loss, (pg, kl, entropy) = regularized_token_loss(logits.log_softmax(), Tensor([0,1]),
            .7, 8, ref, 2, kl_beta=.1, entropy_beta=.2)
        loss.backward()
    np.testing.assert_allclose(kl.item(), (p*(np.log(p)-np.log(q))).sum()/2, atol=1e-6)
    np.testing.assert_allclose(entropy.item(), -(p*np.log(p)).sum()/2, atol=1e-6)
    # tinygrad may materialize an explicit zero gradient for a detached leaf.
    if ref.grad is not None:
        np.testing.assert_array_equal(ref.grad.numpy(), np.zeros_like(q))
    assert np.isfinite(logits.grad.numpy()).all()
    log_ratio = np.log(p)-np.log(q)
    expected = .7/8*(p-np.eye(2,dtype=np.float32))
    expected += .1/2*p*(log_ratio-(p*log_ratio).sum(axis=1,keepdims=True))
    expected += .2/2*p*(np.log(p)-(p*np.log(p)).sum(axis=1,keepdims=True))
    np.testing.assert_allclose(logits.grad.numpy(), expected, atol=1e-6)


def test_chunking_preserves_regularized_gradient():
    values = np.array([[1.,2.],[3.,1.],[.5,.5]], dtype=np.float32)
    grads=[]
    for sizes in ([3], [1,2]):
        logits=Tensor(values).is_param_(True)
        offset=0
        for size in sizes:
            with Tensor.train():
                loss,_=regularized_token_loss(logits[offset:offset+size].log_softmax(),
                    Tensor([0,1,0][offset:offset+size]), .4, 8,
                    Tensor(np.full((size,2),np.log(.5),dtype=np.float32)), 3,
                    kl_beta=.1,entropy_beta=.2)
                loss.backward()
            offset+=size
        grads.append(logits.grad.numpy())
    np.testing.assert_allclose(*grads, atol=1e-6)


def test_reference_adapter_is_immutable():
    base=Linear(2,2,bias=False)
    layer=LoRALinear(base,r=1,alpha=2)
    reference=frozen_lora_tail(SimpleNamespace(ffn_up=layer))
    before=reference.ffn_up.B.numpy().copy()
    layer.B.assign(Tensor.ones(*layer.B.shape)).realize()
    np.testing.assert_array_equal(before,reference.ffn_up.B.numpy())
    assert not np.array_equal(layer.B.numpy(),before)


def test_actual_update_owner_checks_replay_before_mutation(tmp_path):
    weight=Tensor([[.8,-.2],[-.3,.7]]).is_param_(True)
    class Sampler:
        def logprobs(self, h, temperature):
            return (h@weight/temperature).log_softmax()
    sampler=Sampler()
    hidden=np.eye(2,dtype=np.float32)
    tokens=np.array([0,1],dtype=np.int32)
    logp=sampler.logprobs(Tensor(hidden).unsqueeze(0),1.).numpy()[0]
    path=tmp_path/'features.npz'
    np.savez(path,hidden=hidden,tokens=tokens,old_logp=logp[[0,1],tokens])
    record=dict(feature=str(path),feature_sha256=file_sha256(path),episode=0)
    groups=[dict(records=[record],advantages=np.array([1.,-1.]))]
    cfg=dict(temperature=1.,logprob_tolerance=.005,kl_beta=0.,entropy_beta=.001,max_grad_norm=0)
    opt=Adam([weight],lr=.01)
    before=weight.numpy().copy()
    result=update_batch(sampler,None,opt,[weight],groups,cfg)
    assert result['stepped'] and result['entropy']>0
    assert not np.array_equal(before,weight.numpy())
    after=weight.numpy().copy()
    with pytest.raises(ValueError,match='logprob mismatch'):
        update_batch(sampler,None,opt,[weight],groups,cfg)
    np.testing.assert_array_equal(after,weight.numpy())


def test_matched_aggregation_divides_kl_and_entropy_by_trajectories():
    p = np.array([[.8,.2],[.4,.6]], dtype=np.float32)
    q = np.array([[.5,.5],[.3,.7]], dtype=np.float32)
    for policy, denominator in (('legacy_token_mean', 2), ('matched', 8)):
        _, (_, kl, entropy) = regularized_token_loss(Tensor(np.log(p)), Tensor([0,1]), .7, 8, Tensor(np.log(q)), 2,
                                                     kl_beta=.1, entropy_beta=.2, kl_aggregation=policy)
        np.testing.assert_allclose(kl.item(), (p*(np.log(p)-np.log(q))).sum()/denominator, atol=1e-6)
        np.testing.assert_allclose(entropy.item(), -(p*np.log(p)).sum()/denominator, atol=1e-6)
    with pytest.raises(ValueError, match='unknown KL aggregation'):
        regularized_token_loss(Tensor(np.log(p)), Tensor([0,1]), .7, 8, Tensor(np.log(q)), 2, kl_aggregation='x')


def test_kl_gradient_share_matches_the_policy_gradient_under_matched():
    """4 trajectories x 128 tokens, |A| = .5, beta 1e-3, reference O(1) away: the KL term's gradient norm over the
    PG's is ~beta/|A| (order 1e-3) under `matched` and T/N = 128x smaller under the legacy token mean (audit D1)."""
    rng = np.random.default_rng(0)
    logits, ref = rng.standard_normal((2, 512, 16)).astype(np.float32)
    tokens = rng.integers(0, 16, 512)
    advantages = np.repeat([.5, -.5, .5, -.5], 128)
    share = {}
    for policy in ('legacy_token_mean', 'matched'):
        grads = []
        for pg_on, kl_beta in ((1, 0.), (0, 1e-3)):
            x = Tensor(logits).is_param_(True)
            with Tensor.train():
                lp = x.log_softmax()
                _, (pg, kl, _) = regularized_token_loss(lp, Tensor(tokens), 1., 4, Tensor(ref).log_softmax(), 512,
                                                        kl_beta=1e-3, kl_aggregation=policy)
                pg = -(lp.gather(1, Tensor(tokens).reshape(-1, 1)).reshape(-1)*Tensor(advantages.astype(np.float32))).sum()/4
                (pg if pg_on else kl_beta*kl).backward()
            grads.append(np.linalg.norm(x.grad.numpy()))
        share[policy] = grads[1]/grads[0]
    np.testing.assert_allclose(share['matched']/share['legacy_token_mean'], 128, rtol=1e-3)
    assert 1e-4 < share['matched'] < 1e-2 and share['legacy_token_mean'] < 1e-4
