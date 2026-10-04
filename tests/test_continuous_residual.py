import numpy as np
import pytest
import torch
from marl.continuous_residual import ContinuousMARL, residual_command, continuous_anchor
from scripts.run_continuous_marl import make_episode, update, protocol, act
from scripts.run_measured_marl import observation
from scripts.run_tpg_learning import make_episode as original_episode
from tests.test_measured_marl import state


def test_command_bound_identity_blocked_and_small_nonzero_correction():
    rng=np.random.default_rng(58);nominal=rng.normal(size=(1000,3))
    nominal/=np.maximum(np.linalg.norm(nominal,axis=-1,keepdims=True),1.)
    raw=rng.normal(size=(1000,3))*10;enabled=rng.random(1000)>.3
    output=residual_command(nominal,raw,enabled,.2)
    assert np.max(np.linalg.norm(output-nominal,axis=-1))<=.2+1e-12
    assert np.max(np.linalg.norm(output,axis=-1))<=1.+1e-12
    np.testing.assert_array_equal(output[~enabled],nominal[~enabled])
    np.testing.assert_array_equal(residual_command(nominal,np.zeros_like(raw),enabled,.2),nominal)
    assert np.linalg.norm(residual_command(np.zeros((1,3)),np.full((1,3),1e-4),[True],.2))>0


@pytest.mark.parametrize('variant',['r_mappo','mappo_anchor','graph_ppo','graph_anchor'])
def test_continuous_shapes_causality_and_parameter_updates(variant):
    torch.set_num_threads(1);net=ContinuousMARL(variant,16);s=state(8)
    s['nodes'][...,20:22]=0.
    dist,_=net('low',s)
    assert dist.mean.shape==(8,3,3)
    torch.testing.assert_close(dist.mean,torch.zeros_like(dist.mean))
    torch.testing.assert_close(continuous_anchor(dist,s),torch.zeros(8,3))
    p=protocol();p.update(epochs=1,minibatch_size=8)
    for level in ('high','low'):
        with torch.no_grad():d,value=net(level,s);a=d.sample();lp=d.log_prob(a)
        rows=[dict(**{k:v[i].numpy() for k,v in s.items()},action=a[i].numpy(),logp=lp[i].numpy(),
            value=value[i].numpy(),reward=float(i%3),duration=1,done=i==7) for i in range(8)]
        before={k:v.clone() for k,v in net.state_dict().items()}
        stats=update(net,torch.optim.Adam(net.parameters_for(level),lr=.001),rows,level,np.zeros_like(value[-1]),p)
        assert np.isfinite(stats['loss'])
        assert any(not torch.equal(v,before[k]) for k,v in net.state_dict().items() if k.startswith(level+'_actor'))
    changed={k:v.clone() for k,v in s.items()};changed['history'][:,1]+=20.
    torch.testing.assert_close(net('low',s)[0].mean[:,0],net('low',changed)[0].mean[:,0])


def test_blocked_rollout_trains_critic_without_spurious_actor_gradient():
    torch.set_num_threads(1);net=ContinuousMARL('graph_anchor',16);s=state(8)
    s['valid'][...,0]=False
    with torch.no_grad():d,v=net('low',s);a=d.sample();lp=d.log_prob(a)
    records=[dict(**{k:x[i].numpy() for k,x in s.items()},action=a[i].numpy(),logp=lp[i].numpy(),
        value=v[i].numpy(),reward=-1.,duration=1,done=i==7) for i in range(8)]
    before={k:x.clone() for k,x in net.state_dict().items()}
    p=protocol();p.update(epochs=1,minibatch_size=8)
    result=update(net,torch.optim.Adam(net.parameters_for('low'),lr=.001),records,'low',np.zeros(3),p)
    assert result['active_samples']==0 and np.isfinite(result['loss'])
    assert all(torch.equal(x,before[k]) for k,x in net.state_dict().items() if k.startswith('low_actor'))
    assert any(not torch.equal(x,before[k]) for k,x in net.state_dict().items() if k.startswith('low_critic'))


def test_zero_mean_full_control_path_matches_rule():
    a=original_episode(2070000999,duration=4.);b=make_episode(2070000999,duration=4.)
    net=ContinuousMARL('graph_anchor',16)
    try:
        while not a.done:
            assert a.needs_decision==b.needs_decision
            if a.needs_decision:
                a.choose_priority(a.conventional_priority());b.choose_priority(b.conventional_priority())
            raw,_,_=act(net,'low',observation(b),greedy=True)
            np.testing.assert_array_equal(raw,np.zeros((3,3)))
            a.step_control(a.conventional_low());b.step_control(raw)
        assert a.result('rule')['final_state_hash']==b.result('continuous')['final_state_hash']
        assert b.residual_sum==0.
    finally:a.close();b.close()


def test_anchor_pairs_have_bit_identical_initial_parameters():
    for a,b in (('r_mappo','mappo_anchor'),('graph_ppo','graph_anchor')):
        torch.manual_seed(58);na=ContinuousMARL(a,16)
        torch.manual_seed(58);nb=ContinuousMARL(b,16)
        for key,value in na.state_dict().items():torch.testing.assert_close(value,nb.state_dict()[key],rtol=0,atol=0)
