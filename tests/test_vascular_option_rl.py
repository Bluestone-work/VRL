"""Admission checks for information boundaries, pairing and PPO returns."""
import numpy as np
import torch
from marl.vascular_option_rl import OptionPolicy, OBS_DIM, HISTORY
from scripts.vascular_option_study import Episode, advantages, DEV_BASE


def test_central_critic_does_not_make_actor_read_other_agents():
    torch.manual_seed(1); p=OptionPolicy(memory=True,central=True)
    with torch.no_grad():
        p.actor.weight.normal_()
        x=torch.randn(1,3,HISTORY,OBS_DIM); m=torch.ones(1,3)
        a,v=p(x,m); changed=x.clone(); changed[:,1:]+=10
        b,_=p(changed,m)
        torch.testing.assert_close(a.logits[:,0],b.logits[:,0])
        perm=[2,0,1]; c,w=p(x[:,perm],m[:,perm])
        torch.testing.assert_close(a.logits[:,perm],c.logits)
        torch.testing.assert_close(v[:,perm],w)


def test_gae_resets_at_episode_boundary():
    r=np.array([1.,2.,100.])[:,None,None]; v=np.zeros_like(r)
    done=np.array([0.,1.,0.])[:,None]
    a,_=advantages(r,v,done,np.array([[3.]]),gamma=1.,lam=1.)
    np.testing.assert_allclose(a[:,0,0],[3,2,103])


def test_pairing_and_no_online_truth_proxy():
    a=Episode('mca_m1_lvo',3,DEV_BASE,horizon=1.)
    b=Episode('mca_m1_lvo',3,DEV_BASE,horizon=1.)
    try:
        assert a.initial_state_hash==b.initial_state_hash
        assert a.manifest['scenario_hash']==b.manifest['scenario_hash']
        assert not hasattr(a.ctl.env,'positions_mm')
        assert not hasattr(a.ctl.env,'masses')
        assert not hasattr(a.ctl.env,'edges')
        x,m=a.observation(); y,k=b.observation()
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(a.candidates[:,0],b.candidates[:,0])
        while not a.done:
            a.step(np.zeros(3,int)); b.step(np.zeros(3,int))
        assert a.metrics()==b.metrics()
    finally:
        a.env.close(); b.env.close()
