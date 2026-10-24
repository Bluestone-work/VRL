import numpy as np
import torch
from marl.image_flow_sac import ImageActor,ImageCritic,ControlEpisode,ImageFlowController,CTX,observation_tensors


def inputs():
    return (torch.randn(4,24,11),torch.zeros(4,24,dtype=torch.bool),
            torch.zeros(4,CTX),torch.randn(4,4,32,32))


def test_actor_image_and_history_affect_continuous_command_after_reward_update():
    torch.manual_seed(95);torch.set_num_threads(1);actor=ImageActor();obs=list(inputs())
    optimizer=torch.optim.Adam(actor.parameters(),lr=.002)
    # A reward-derived action preference must change the final command, without
    # any heuristic selecting a different command after the actor.
    for _ in range(4):
        action,_=actor.sample(*obs,deterministic=True)
        loss=(action-torch.tensor([.2,-.1,.1])).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    command,_=actor.sample(*obs,deterministic=True)
    altered=list(obs);altered[3]=altered[3]+.5
    other,_=actor.sample(*altered,deterministic=True)
    assert not torch.allclose(command,other)
    altered=list(obs);altered[0]=altered[0]+1.
    other,_=actor.sample(*altered,deterministic=True)
    assert not torch.allclose(command,other)
    assert command.abs().max()<1


def test_double_critic_and_squashed_action_gradients_are_finite():
    torch.manual_seed(96);actor=ImageActor();critic=ImageCritic();obs=inputs()
    action,logp=actor.sample(*obs);q1,q2=critic(*obs,action)
    loss=(.002*logp-torch.minimum(q1,q2)).mean();loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.parameters())
    assert not torch.allclose(q1,q2)


def test_station_keeping_command_survives_tpg_hold_and_original_hold_still_zeros():
    ep=ControlEpisode(1,'mca_m1_lvo',2149000000,flow_inlet_mm_s=.1)
    try:
        est=ep.observe();target=ep.plan_targets_now(est);ep.ctl.act(target,est)
        command=np.array([[-.15,0.,0.]])
        ep.station_keeping=False;ep.step(est,command,np.array([True]))
        np.testing.assert_allclose(ep.prev_local,0.)
        est=ep.observe();ep.station_keeping=True;ep.step(est,command,np.array([True]))
        np.testing.assert_allclose(ep.prev_local,command)
        assert ep.authority['hold_nonzero_steps']==1 and ep.hold_s>0
    finally:ep.close()


def test_policy_observation_does_not_read_current_plant_truth():
    ep=ControlEpisode(1,'mca_m1_lvo',2149000001,flow_inlet_mm_s=.05)
    try:
        est=ep.observe();target=ep.plan_targets_now(est);rule=ep.ctl.act(target,est);hold=ep.hold(est)
        actor=ImageActor();ctl=ImageFlowController(ep,actor=actor)
        ep.env.positions_mm[:]=np.nan;ep.env.masses[:]=np.nan
        obs=ctl.observe(ep,est,target,rule,hold)
        assert all(np.isfinite(r['ctx']).all() and np.isfinite(r['image']).all() for r in obs)
        with torch.no_grad():a,_=actor.sample(*observation_tensors(obs),deterministic=True)
        out=ctl.execute(ep,est,a.numpy())
        assert np.isfinite(out).all() and np.linalg.norm(out,axis=1).max()<=1.+1e-6
    finally:ep.close()


def test_guard_does_not_apply_the_reference_inward_bias_twice():
    ep=ControlEpisode(1,'mca_m1_lvo',2149000002,flow_inlet_mm_s=.05)
    try:
        est=ep.observe();target=ep.plan_targets_now(est);rule=ep.ctl.act(target,est);hold=ep.hold(est)
        ctl=ImageFlowController(ep);ctl.observe(ep,est,target,rule,hold)
        expected=ctl.base.copy()
        actual=ctl.execute(ep,est,expected)
        np.testing.assert_allclose(actual,expected,atol=3e-7)
    finally:ep.close()


def test_hold_projection_compensates_measured_axial_drift():
    ep=ControlEpisode(1,'mca_m1_lvo',2149000003,flow_inlet_mm_s=.1)
    try:
        est=ep.observe();target=ep.plan_targets_now(est);rule=ep.ctl.act(target,est)
        ctl=ImageFlowController(ep);ctl.observe(ep,est,target,rule,np.array([True]))
        ctl.drift[0]=.1*ctl.F[0][0];ctl.gain[0]=1.
        command=ctl.execute(ep,est,np.zeros((1,3)))
        predicted_velocity=ctl.F[0].T@command[0]*ctl.speed+ctl.drift[0]
        assert abs(predicted_velocity@ctl.F[0][0])<1e-8
        assert np.linalg.norm(command[0])>.05
    finally:ep.close()
