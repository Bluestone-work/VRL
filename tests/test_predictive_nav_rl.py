import numpy as np
import torch
from marl.predictive_nav_rl import NavigationActor,MAX_CANDIDATES,CAND_DIM,advantages


def observations(batch=4):
    h=torch.randn(batch,24,11);hm=torch.zeros(batch,24,dtype=torch.bool);ctx=torch.randn(batch,19)
    c=torch.randn(batch,MAX_CANDIDATES,CAND_DIM);c[:,:,-1]=0;c[:,0,-1]=1
    c[:,:,-2]=0
    valid=torch.zeros(batch,MAX_CANDIDATES,dtype=torch.bool);valid[:,:4]=True
    return h,hm,ctx,c,valid


def test_reward_gradient_changes_command_probabilities_and_invalid_actions_stay_masked():
    torch.manual_seed(94);torch.set_num_threads(1);net=NavigationActor(proposal_bias=2.)
    obs=observations();d,_=net.dist(*obs);initial=d.probs.detach().clone()
    optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    # Positive environment advantage for candidate 1, negative for the prior.
    for _ in range(5):
        d,_=net.dist(*obs);loss=-d.log_prob(torch.ones(4,dtype=torch.long)).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    d,_=net.dist(*obs)
    assert (d.probs[:,1]>initial[:,1]).all()
    assert torch.equal(d.probs[:,4:],torch.zeros_like(d.probs[:,4:]))
    assert (d.logits.argmax(-1)==1).all()


def test_actor_uses_measured_history_for_adaptation():
    torch.manual_seed(95);net=NavigationActor()
    # After PPO starts changing the zero-initialized head, history can affect rank.
    torch.nn.init.normal_(net.actor[-1].weight,std=.1)
    obs=list(observations());d,_=net.dist(*obs)
    obs[0]=obs[0]+1.;other,_=net.dist(*obs)
    assert not torch.allclose(d.probs,other.probs)


def test_gae_cuts_terminal_streams_and_bootstraps_truncations():
    rows=[dict(stream=(0,1,0),reward=1.,value=2.,terminal=False),
          dict(stream=(0,1,0),reward=3.,value=4.,terminal=True),
          dict(stream=(0,2,0),reward=5.,value=6.,terminal=False,truncated=True,terminal_value=10.)]
    adv,ret=advantages(rows,{(0,1,0):999.,(0,2,0):999.},gamma=.9,lam=1.)
    np.testing.assert_allclose(ret,[3.7,3.,14.],atol=1e-5)


def test_rl_off_reproduces_parent_and_policy_receives_no_dynamic_truth():
    from scripts.benchmark_lysis import LysisEpisode
    from marl.dyn_predictor import Ensemble
    from marl.dyn_select import PredictiveSelector
    from marl.predictive_nav_rl import RLNavigator,PREDICTORS,PARENT_WEIGHTS
    ep=LysisEpisode(1,'mca_m1_lvo',2139000000,flow_inlet_mm_s=.05)
    try:
        est=ep.observe();tgt=ep.plan_targets_now(est);rule=ep.ctl.act(tgt,est);hold=ep.hold(est)
        parent=PredictiveSelector(ep,'learned',Ensemble(PREDICTORS),.0445,PARENT_WEIGHTS)
        off=RLNavigator(ep,rl_off=True)
        np.testing.assert_allclose(parent(ep,est,tgt,rule,hold),off(ep,est,tgt,rule,hold),atol=1e-12)
        actor=RLNavigator(ep,stochastic=True)
        # Static clot map was captured at construction. Plant truth is irrelevant
        # to these features; the sensor estimate was already obtained.
        old_pos=ep.env.positions_mm.copy();old_mass=ep.env.masses.copy()
        ep.env.positions_mm[:]=np.nan;ep.env.masses[:]=np.nan
        out=actor(ep,est,tgt,rule,hold)
        assert np.isfinite(out).all()
        assert np.linalg.norm(out,axis=-1).max()<=1.+1e-6
        assert actor.records and actor.records[0]['learn']
        assert actor.summary()['rl_decisions']>0
        ep.env.positions_mm[:]=old_pos;ep.env.masses[:]=old_mass
    finally:ep.close()
