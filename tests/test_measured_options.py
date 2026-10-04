"""Options keep detector identities, causal information and physical costs distinct."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from marl.multicluster import ClusterObservation
from marl.measured_options import (option_observation, latch_targets, target_packet,
    joint_measured_projection, measured_assignment, priority_options,
    OptionActorCritic, smdp_gae)
from scripts.option_learning_episode import OptionEpisode, PROTOCOL
from scripts.run_option_learning import check_development_seed
from scripts.tracked_learning_episode import TrackedLearningEpisode


def packet(n=3):
    nav = np.zeros((n,111),np.float32)
    for i in range(n):
        for k in range(4):
            nav[i,11+6*k:17+6*k] = [1.,.1*(k+1),1.,0.,0.,1.]
    rel = np.zeros((n,n,3))
    visible = np.zeros((n,n),bool)
    for i in range(n):
        for j in range(n):
            if i!=j:
                rel[i,j,0] = 3.*(j-i); visible[i,j]=True
    return ClusterObservation(nav,np.tile(np.arange(4),(n,1)),rel,np.zeros_like(rel),visible,np.ones(n,bool))


def settings():
    return json.loads(PROTOCOL.read_text())['supervisor']


def test_target_identity_survives_slot_permutation_without_input_mutation():
    p=packet(); targets=latch_targets(p,np.array([2,3,4]))
    original=p.navigation.copy()
    q=target_packet(p,targets)
    assert q.clot_ids[:,0].tolist()==[1,2,3]
    np.testing.assert_array_equal(p.navigation,original)
    reordered=ClusterObservation(q.navigation,
        q.clot_ids,q.peer_relative_mm,q.peer_relative_velocity_mm_s,q.peer_visible,q.active)
    again=target_packet(reordered,targets)
    assert again.clot_ids[:,0].tolist()==[1,2,3]


def test_invisible_peers_never_enter_actor_or_supervisor():
    p=packet(); p.peer_visible[:]=False
    a,_=option_observation(p,np.zeros(3,int))
    u,check=joint_measured_projection(np.zeros((3,3)),p,settings())
    p.peer_relative_mm[:]=1e8;p.peer_relative_velocity_mm_s[:]=-1e8
    b,_=option_observation(p,np.zeros(3,int))
    v,other=joint_measured_projection(np.zeros((3,3)),p,settings())
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(u,v)
    assert check==other


def test_joint_projection_considers_both_proposed_commands():
    p=packet(2); p.peer_relative_mm[0,1,0]=2.2;p.peer_relative_mm[1,0,0]=-2.2
    desired=np.array([[1.,0,0],[-1.,0,0]])
    result,report=joint_measured_projection(desired,p,settings())
    assert report['residual']<1e-5
    assert (result[1]-result[0])[0] >= .13-1e-6
    assert np.linalg.norm(result,axis=1).max()<=1+1e-8


def test_infeasible_constraints_are_reported_not_relabelled_safe():
    p=packet(2);p.peer_relative_mm[0,1,0]=.1;p.peer_relative_mm[1,0,0]=-.1
    _,report=joint_measured_projection(np.zeros((2,3)),p,settings())
    assert report['residual']>0
    assert 'safe' not in report


def test_missing_tracks_are_not_ground_truth_recovered():
    p=packet(); p.active[:]=False
    f,valid=option_observation(p,np.zeros(3,int))
    assert valid[:,0].all() and not valid[:,1:].any()
    result,report=joint_measured_projection(np.ones((3,3)),p,settings())
    assert np.all(result==0) and report['missing_tracks']==3


def test_untrained_argmax_is_memory_for_n1_and_n3():
    for n in (1,3):
        f,mask=option_observation(packet(n),np.zeros(n,int))
        net=OptionActorCritic()
        dist,value=net(torch.from_numpy(f)[None],torch.from_numpy(mask)[None])
        assert torch.all(dist.logits.argmax(-1)==0)
        assert torch.isfinite(value).all()


def test_balanced_assignment_uses_distinct_measured_ids():
    p=packet()
    choice=measured_assignment(p)
    assert len(set(latch_targets(p,choice)))==3
    assert np.all((choice>=1)&(choice<=4))


def test_smdp_discount_uses_elapsed_control_steps():
    adv,ret=smdp_gae([1.,2.],[.5,1.],[False,False],[2,3],4.,.9,.8)
    last=2.+.9**3*4.-1.
    first=1.+.9**2*1.-.5+.9**2*.8**2*last
    np.testing.assert_allclose(adv,[first,last],rtol=1e-6)
    np.testing.assert_allclose(ret,adv+[.5,1.])


def test_smdp_terminal_does_not_bootstrap_next_episode():
    adv,_=smdp_gae([1.,2.],[0.,100.],[True,True],[10,1],999.,.99,.95)
    np.testing.assert_allclose(adv,[1.,-98.])


@pytest.mark.parametrize('seed',[1730000000,1630000000,1530000000,1430000000])
def test_confirmation_pools_rejected(seed):
    with pytest.raises(ValueError,match='Confirmation'):
        check_development_seed(seed)


def test_legacy_option_zero_reproduces_frozen_memory_trajectory():
    a=TrackedLearningEpisode(1740000100,duration=1.)
    b=OptionEpisode(1740000100,duration=1.,supervisor='legacy')
    try:
        while not a.done:
            choice=a.library.conventional_choice(a.packet,a.valid,a.details,memory=True)
            a.step(choice)
        while not b.done:
            b.step_option(np.zeros(3,int))
        np.testing.assert_array_equal(a.env.positions_mm,b.env.positions_mm)
        np.testing.assert_array_equal(a.env.masses,b.env.masses)
        assert a.result('memory')['final_state_hash']==b.result('memory')['final_state_hash']
    finally:
        a.close();b.close()


def test_flat_and_hierarchical_memory_share_exact_physics():
    a=OptionEpisode(1740000101,duration=1.,option_steps=1)
    b=OptionEpisode(1740000101,duration=1.,option_steps=10)
    try:
        for ep in (a,b):
            while not ep.done:ep.step_option(np.zeros(3,int))
        assert a.result('memory')['final_state_hash']==b.result('memory')['final_state_hash']
        assert a.steps==b.steps==10
        assert a.macro_steps==10 and b.macro_steps==1
    finally:
        a.close();b.close()


def test_observations_do_not_depend_on_instrumentation_labels():
    ep=OptionEpisode(1740000102,duration=.3)
    try:
        for name in ('_route_directions','_assigned_targets','_target_distances','_solve_assignment'):
            setattr(ep.env,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError('privileged query')))
        # Only the measurement/actor preparation is audited here; native physics
        # may legitimately use routes internally when the environment steps.
        ep.prepare()
        choice=ep.conventional_options('balanced')
        assert choice.shape==(3,)
        assert np.isfinite(ep.high_features).all()
    finally:ep.close()
