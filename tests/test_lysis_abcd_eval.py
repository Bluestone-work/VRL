import numpy as np
import torch
from marl.lysis_abcd import FlowPolicy
from scripts.train_lysis_nav import History
from scripts.train_lysis_nav import weighted_prior_loss


def test_reset_windows_and_batch_independence():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    p=FlowPolicy(history=True,arch='transformer',layers=1,window=8,dim=39).eval()
    hist=History(2,8)
    seq,mask=hist.push(np.ones((2,39),np.float32))
    seq=torch.tensor(seq);mask=torch.tensor(mask)
    with torch.no_grad():
        a=p.backbone(seq,mask)
        changed=seq.clone();changed[mask]=1234
        assert torch.allclose(a,p.backbone(changed,mask))
        assert torch.allclose(a[:1],p.backbone(seq[:1],mask[:1]),atol=1e-6)
        reset=History(2,8)
        assert reset.mask.all() and not reset.seq.any()


def test_auxiliary_reaches_context_and_checkpoint_roundtrip():
    torch.manual_seed(2)
    p=FlowPolicy(history=True,arch='transformer',layers=1,window=8,dim=39)
    seq=torch.randn(2,8,39);mask=torch.zeros(2,8,dtype=torch.bool)
    pred=p.predict_motion(seq,mask,torch.randn(2,3))
    pred.square().mean().backward()
    assert p.context.weight_ih_l0.grad.abs().sum()>0
    q=FlowPolicy(history=True,arch='transformer',layers=1,window=8,dim=39)
    q.load_state_dict(p.state_dict())
    assert torch.equal(p.backbone(seq,mask),q.backbone(seq,mask))


def test_previous_command_uses_execution_frame():
    # world -> local F; deployment must invert the OLD frame, as training does.
    old=np.array([[[0.,1,0],[-1,0,0],[0,0,1]]])
    world=np.array([[.3,.4,.5]])
    local=np.einsum('nij,nj->ni',old,world)
    assert np.allclose(np.einsum('nji,nj->ni',old,local),world)


def test_history_previous_command_contract_is_local_frame():
    # The policy token stores the executed local command, while the plant
    # receives its world transform. The two representations must not be mixed.
    local = np.array([[.2, -.4, .7]])
    F = np.array([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])
    world = np.einsum('nji,nj->ni', F, local)
    assert not np.allclose(local, world)
    assert np.allclose(np.einsum('nij,nj->ni', F, world), local)


def test_prior_regularizer_is_sample_weighted_not_broadcast_matrix():
    mean=torch.tensor([[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]])
    prior=torch.zeros_like(mean); w=torch.ones(3)
    # squared errors are 1, 4, 9; correct mean is 14/3.
    assert torch.allclose(weighted_prior_loss(mean,prior,w),torch.tensor(14/3))
    masked=weighted_prior_loss(mean,prior,torch.tensor([1.,0.,1.]))
    assert torch.allclose(masked,torch.tensor(5.))


def test_gae_preserves_wait_span_and_separate_truncated_episode():
    from scripts.train_lysis_local import gae
    first=(0,1,0); second=(0,2,0)
    # Includes a waiting transition between decision and reward. At truncation
    # value 8 bootstraps; the next episode's reward 100 must never leak back.
    buf=dict(stream=[first,first,first,second],rew=[0.,0.,4.,100.],
             val=[0.,0.,0.,0.],done=[False,False,False,True])
    adv,ret=gae(buf,{first:8.},gamma=.5,lam=1.)
    assert np.allclose(ret,[2.,4.,8.,100.])
    buf['done'][2]=True
    _,ret=gae(buf,{first:8.},gamma=.5,lam=1.)
    assert np.allclose(ret,[1.,2.,4.,100.])


def test_lateral_intervention_preserves_speed_and_legacy_default():
    from types import SimpleNamespace
    from scripts.train_lysis_nav import command
    frames = np.eye(3)[None]
    ctl = SimpleNamespace(pts=np.array([[1.,0.,0.]]), carrot=np.array([[1.,0.,0.]]),
                          frames=lambda est: frames)
    ep = SimpleNamespace(n=1, ctl=ctl, sensor=SimpleNamespace(healthy=np.array([1.])),
                         env=SimpleNamespace(config=SimpleNamespace(robot_radius_mm=.1),
                         tree=SimpleNamespace(normals=np.array([[0.,1.,0.]]),
                                              binormals=np.array([[0.,0.,1.]]))))
    est=SimpleNamespace(pos=np.zeros((1,3))); action=np.array([[.6,.8,-.5]])
    default=command(ep,est,action,np.array([True]),np.array([0.]))
    explicit=command(ep,est,action,np.array([True]),np.array([0.]),1.,1.)
    reduced=command(ep,est,action,np.array([True]),np.array([0.]),1.,.25)
    assert np.array_equal(default,explicit)
    assert np.allclose(np.linalg.norm(default,axis=1),np.linalg.norm(reduced,axis=1))
    assert np.allclose(np.linalg.norm(reduced,axis=1),.6)
    assert np.allclose(reduced[0,1:]/reduced[0,0],.25*default[0,1:]/default[0,0])
    assert not command(ep,est,action,np.array([False]),np.array([0.]),1.,.25).any()


def test_lateral_gate_is_per_robot_and_leaves_approach_unchanged():
    from types import SimpleNamespace
    from scripts.train_lysis_nav import lateral_scale_for_targets
    ep=SimpleNamespace(n=3,env=SimpleNamespace(clot_positions_mm=np.zeros((1,3))))
    est=SimpleNamespace(pos=np.array([[.1,0.,0.],[2.,0.,0.],[0.,0.,0.]]))
    cfg=dict(lateral_residual_scale=.25,lateral_near_radius=.6)
    assert np.array_equal(lateral_scale_for_targets(ep,est,[0,0,-1],cfg),[.25,1.,1.])
    assert np.array_equal(lateral_scale_for_targets(ep,est,[0,0,-1],{}),[1.,1.,1.])


def test_observable_history_uses_final_command_and_acquisition_age():
    from scripts.benchmark_lysis import LysisEpisode
    from scripts.train_lysis_nav import token, make_policy
    from marl.deployable_sensing import DeployableConfig
    ep=LysisEpisode(1,'mca_m1_lvo',2600000021,sense_cfg=DeployableConfig(latency_steps=2))
    try:
        est=ep.observe();targets=ep.plan_targets_now(est);ep.ctl.act(targets,est)
        # A held robot must record zero actually-sent command, not the proposed command.
        ep.step(est,np.ones((1,3)),np.ones(1,dtype=bool))
        assert not ep.sent_world.any()
        est=ep.observe();targets=ep.plan_targets_now(est);ep.ctl.act(targets,est)
        t=token(ep,est,targets,ep.hold(est),ep.sent_world,observable_history=True)
        assert t.shape==(1,40) and not t[:,23:26].any()
        assert np.allclose(t[:,-1],ep.env.elapsed_s-est.frame_time_s)
        cfg=dict(arch='transformer',layers=1,window=8,matrix_arm='nav_tf_v3',observable_history=True)
        p=make_policy(cfg);seq=torch.tensor(np.repeat(t[:,None,:],8,axis=1))
        d,v=p.dist(seq,torch.zeros((1,8),dtype=torch.bool))
        assert d.mean.shape==(1,3) and torch.isfinite(v).all()
    finally:
        ep.close()
