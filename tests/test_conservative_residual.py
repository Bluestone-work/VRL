import numpy as np
import torch
import pytest
from marl.conservative_residual import (ResidualMARL, bounded_candidates, reference_distribution,
                                        anchor_penalty, measured_anchor_weight)
from tests.test_measured_marl import state
from scripts.run_residual_marl import make_episode, old_make_episode, update, protocol
from scripts.run_measured_marl import observation, tensors


def test_bound_preserves_nominal_blocked_controls_and_permissions():
    rng=np.random.default_rng(80)
    commands=rng.normal(size=(3,9,3));commands/=np.maximum(np.linalg.norm(commands,axis=-1,keepdims=True),1.)
    features=rng.normal(size=(3,9,9)).astype(np.float32)
    valid=np.ones((3,9),bool);valid[1,:7]=False
    output, result=bounded_candidates(commands,features,valid,.2)
    np.testing.assert_array_equal(output[:,0],commands[:,0])
    np.testing.assert_array_equal(output[1],commands[1])
    np.testing.assert_array_equal(result[1],features[1])
    assert np.max(np.linalg.norm(output[[0,2]]-commands[[0,2],0,None],axis=-1))<=.2+1e-12
    assert np.max(np.linalg.norm(output,axis=-1))<=1.+1e-12
    np.testing.assert_array_equal(result[:,:,6:],features[:,:,6:])


def test_reference_mask_and_measured_risk_finite():
    s=state();s['candidates'][...,8]=0.;s['candidates'][:,:,0,8]=1.
    s['nodes'][...,20:22]=0.
    ref=reference_distribution(s,np.log(8.))
    torch.testing.assert_close(ref.probs[:,:,0],torch.full((2,3),.5))
    torch.testing.assert_close(anchor_penalty(ref,s,np.log(8.)),torch.zeros(2,3))
    s['valid'][:,1,:]=False;s['valid'][:,1,7]=True
    ref=reference_distribution(s,np.log(8.))
    assert torch.all(ref.probs[:,1,7]==1.)
    s['nodes'][:,0,21]=.3;s['nodes'][:,2,20]=1.
    torch.testing.assert_close(measured_anchor_weight(s),torch.tensor([[3.,1.,3.],[3.,1.,3.]]))
    assert torch.isfinite(anchor_penalty(ref,s,np.log(8.))).all()


@pytest.mark.parametrize('a,b',[('r_mappo','mappo_anchor'),('graph_ppo','graph_anchor')])
def test_factorial_initialization_exact_and_updates_learn(a,b):
    torch.set_num_threads(1)
    torch.manual_seed(9);n1=ResidualMARL(a,16)
    torch.manual_seed(9);n2=ResidualMARL(b,16)
    for k,v in n1.state_dict().items():torch.testing.assert_close(v,n2.state_dict()[k],rtol=0,atol=0)
    s=state(8);s['nodes'][...,20:22]=0.
    p=protocol();p.update(epochs=1,minibatch_size=8)
    for level in ('high','low'):
        with torch.no_grad():d,value=n2(level,s);action=d.sample();logp=d.log_prob(action)
        records=[dict(**{k:v[i].numpy() for k,v in s.items()},action=action[i].numpy(),logp=logp[i].numpy(),
            value=value[i].numpy(),reward=float(i%3),duration=1,done=i==7) for i in range(8)]
        before={k:v.clone() for k,v in n2.state_dict().items()}
        stats=update(n2,torch.optim.Adam(n2.parameters_for(level),lr=.001),records,level,np.zeros_like(value[-1].numpy()),p)
        assert np.isfinite(stats['loss'])
        assert any(not torch.equal(v,before[k]) for k,v in n2.state_dict().items() if k.startswith(level+'_actor'))
        other='high' if level=='low' else 'low'
        assert all(torch.equal(v,before[k]) for k,v in n2.state_dict().items() if k.startswith(other+'_actor'))


def test_new_rule_controls_and_observed_history_equal_old_rule():
    a=old_make_episode(2040000999,duration=4.)
    b=make_episode(2040000999,duration=4.)
    try:
        while not a.done:
            assert a.needs_decision==b.needs_decision
            if a.needs_decision:
                a.choose_priority(a.conventional_priority());b.choose_priority(b.conventional_priority())
            for k in ('history','nodes','pairs','active','valid'):
                np.testing.assert_array_equal(observation(a)[k],observation(b)[k])
            aa,bb=a.conventional_low(),b.conventional_low()
            np.testing.assert_array_equal(a.low_commands[np.arange(3),aa],b.low_commands[np.arange(3),bb])
            a.step_control(aa);b.step_control(bb)
        assert a.result('tpg')['final_state_hash']==b.result('tpg')['final_state_hash']
    finally:a.close();b.close()


def test_no_peer_history_actor_leakage_for_new_variants():
    s=state();changed={k:v.clone() for k,v in s.items()};changed['history'][:,1]+=10.
    net=ResidualMARL('graph_anchor',16)
    torch.nn.init.normal_(net.low_actor.score[-1].weight,std=.1)
    torch.testing.assert_close(net('low',s)[0].logits[:,0],net('low',changed)[0].logits[:,0])
