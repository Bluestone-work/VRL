"""Admission tests for causal information, ordering ability and PPO semantics."""
import ast
from itertools import permutations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from marl.measured_marl import MeasuredMARL, bids_to_priority, temporal_gae
from marl.tpg_learning import HISTORY, LOW_DIM, NODE_DIM, PAIR_DIM, N_LOW, CANDIDATE_DIM
from scripts.run_measured_marl import PROTOCOL, observation, tensors, update


def state(batch=2):
    torch.manual_seed(61)
    return dict(history=torch.randn(batch,3,HISTORY,LOW_DIM),
        nodes=torch.randn(batch,3,NODE_DIM), pairs=torch.zeros(batch,3,3,PAIR_DIM),
        candidates=torch.randn(batch,3,N_LOW,CANDIDATE_DIM),
        valid=torch.ones(batch,3,N_LOW,dtype=torch.bool), active=torch.ones(batch,3,dtype=torch.bool))


def test_bids_can_express_all_feasible_permutations_and_do_not_preempt():
    nodes=np.zeros((3,NODE_DIM)); pairs=np.zeros((3,3,PAIR_DIM))
    orders=list(permutations(range(3)))
    for index,order in enumerate(orders):
        bids=np.argsort(order)
        assert bids_to_priority(bids,nodes,pairs)==index
    pairs[2,0,4]=1; pairs[0,2,4]=-1
    for order in orders:
        result=orders[bids_to_priority(np.argsort(order),nodes,pairs)]
        assert result.index(2)<result.index(0)


def test_equal_bids_have_measured_cost_tie_breaking():
    nodes=np.zeros((3,NODE_DIM)); nodes[:,15]=1; nodes[:,14]=[.3,.1,.2]
    index=bids_to_priority(np.zeros(3),nodes,np.zeros((3,3,PAIR_DIM)))
    assert list(permutations(range(3)))[index]==(1,2,0)


def test_mappo_ippo_same_actor_initialization_but_different_value_information():
    torch.manual_seed(11); a=MeasuredMARL('r_mappo',16)
    torch.manual_seed(11); b=MeasuredMARL('r_ippo',16)
    for name in ('high_actor','low_actor'):
        for k,v in getattr(a,name).state_dict().items():
            torch.testing.assert_close(v,getattr(b,name).state_dict()[k])
    original=state(); modified={k:v.clone() for k,v in original.items()}
    modified['history'][:,1]+=4
    for level in ('high','low'):
        # IPPO critic is local-history + same public current team packet.
        torch.testing.assert_close(b(level,original)[1][:,0],b(level,modified)[1][:,0])
        assert not torch.allclose(a(level,original)[1][:,0],a(level,modified)[1][:,0])


@pytest.mark.parametrize('variant',['r_mappo','r_ippo','vctpg_ac','vctpg_no_ac'])
def test_policy_shapes_masks_and_actor_history_access(variant):
    net=MeasuredMARL(variant,16); inputs=state()
    inputs['valid'][:,0,:]=False; inputs['valid'][:,0,7]=True
    dh,vh=net('high',inputs); dl,vl=net('low',inputs)
    assert dl.logits.shape==(2,3,9) and vl.shape==(2,3)
    assert torch.all(dl.probs[:,0,7]==1)
    assert vh.shape==((2,) if net.joint_high else (2,3))
    # Nonzero heads make this a real information-access check, not zero-head identity.
    torch.nn.init.normal_(net.low_actor.score[-1].weight,std=.1)
    torch.nn.init.normal_(net.high_actor.score[-1].weight,std=.1)
    modified={k:v.clone() for k,v in inputs.items()}; modified['history'][:,1]+=2
    torch.testing.assert_close(net('low',inputs)[0].logits[:,2],net('low',modified)[0].logits[:,2])
    if not net.joint_high:
        torch.testing.assert_close(net('high',inputs)[0].logits[:,2],net('high',modified)[0].logits[:,2])


def test_gae_bootstraps_cutoff_but_not_terminal_or_episode_reset():
    records=[dict(value=np.array([2.,3.]),reward=1.,done=False,duration=2,next_value=np.array([4.,5.])),
             dict(value=np.array([4.,5.]),reward=2.,done=True,duration=1,next_value=np.array([99.,99.]))]
    adv,ret=temporal_gae(records,np.array([99.,99.]),.9,.8)
    np.testing.assert_allclose(adv[1],[-2.,-3.])
    np.testing.assert_allclose(adv[0],1+.9**2*np.array([4.,5.])-np.array([2.,3.])+(.9*.8)**2*np.array([-2.,-3.]),rtol=1e-6)
    np.testing.assert_allclose(ret[1],[2.,2.])


@pytest.mark.parametrize('variant',['r_mappo','r_ippo','vctpg_ac','vctpg_no_ac'])
def test_per_agent_ppo_updates_both_levels_without_changing_other_actor(variant):
    torch.set_num_threads(1); net=MeasuredMARL(variant,16)
    p=json.loads(PROTOCOL.read_text()); p.update(epochs=1,minibatch_size=8)
    inputs=state(8)
    inputs['nodes'][...,23]=1
    for level in ('high','low'):
        records=[]
        with torch.no_grad():dist,value=net(level,inputs); action=dist.sample(); logp=dist.log_prob(action)
        for i in range(8):
            r={k:v[i].numpy() for k,v in inputs.items()}
            r.update(action=action[i].numpy(),logp=logp[i].numpy(),value=value[i].numpy(),reward=float(i%3),
                done=i==7,duration=1,commands=np.zeros((3,3),np.float32),
                measured_velocity=np.ones((3,3),np.float32),measurement_valid=np.ones(3,bool))
            records.append(r)
        before={k:v.detach().clone() for k,v in net.state_dict().items()}
        optimizer=torch.optim.Adam(net.parameters_for(level),lr=.001)
        stat=update(net,optimizer,records,level,np.zeros_like(value[-1].numpy()),p)
        assert np.isfinite(stat['loss'])
        assert any(not torch.equal(before[k],v) for k,v in net.state_dict().items() if k.startswith(level+'_actor.'))
        other='low' if level=='high' else 'high'
        assert all(torch.equal(before[k],v) for k,v in net.state_dict().items() if k.startswith(other+'_actor.'))


def test_action_conditioned_feature_is_used_only_in_proposed_variant():
    inputs=state()
    for variant in ('vctpg_ac','vctpg_no_ac'):
        net=MeasuredMARL(variant,16)
        torch.nn.init.normal_(net.low_actor.score[-1].weight,std=.1)
        before=net('low',inputs)[0].logits.clone()
        with torch.no_grad():net.low_actor.dynamics[-1].bias.add_(2.)
        after=net('low',inputs)[0].logits
        if variant=='vctpg_ac':assert not torch.allclose(before,after)
        else:torch.testing.assert_close(before,after)


def test_actor_critic_code_has_no_truth_container_access():
    tree=ast.parse(Path('marl/measured_marl.py').read_text())
    forbidden={'env','transport','solution','masses','station_ids','positions_mm','flow_model','info','reward'}
    # Reward access in GAE is dictionary indexing, not policy environment access.
    assert not [n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in forbidden]


def test_control_observation_ignores_forbidden_truth_containers():
    from types import SimpleNamespace
    packet=state(1)
    ep=SimpleNamespace(low_history=packet['history'][0].numpy(),nodes=packet['nodes'][0].numpy(),
        pairs=packet['pairs'][0].numpy(),low_candidates=packet['candidates'][0].numpy(),
        low_valid=packet['valid'][0].numpy(),scene=SimpleNamespace(active=np.ones(3,bool)))
    class Trap:
        def __getattr__(self,key):raise AssertionError('Navigation accessed truth: '+key)
    ep.env=Trap(); ep.info=Trap()
    first=observation(ep)
    ep.env=SimpleNamespace(positions_mm=np.ones((3,3))*1e6,masses=np.ones(4)*99)
    second=observation(ep)
    for k in first:np.testing.assert_array_equal(first[k],second[k])
    net=MeasuredMARL('r_mappo',16)
    torch.testing.assert_close(net('low',tensors(first))[0].logits,net('low',tensors(second))[0].logits)


def test_new_and_old_confirmation_pools_remain_guarded():
    from scripts.run_option_learning import check_development_seed
    for seed in (1930000000,1930000039,1980000000,1980000039):
        with pytest.raises(ValueError):check_development_seed(seed)
