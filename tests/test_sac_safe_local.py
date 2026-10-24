from types import SimpleNamespace
import numpy as np
from marl.sac_safe_local import SafeLocalSAC
from marl.image_flow_sac import ControlEpisode,ImageActor


def test_speed_adaptation_preserves_the_parent_direction():
    ctl=SafeLocalSAC.__new__(SafeLocalSAC)
    ctl.actor=SimpleNamespace(action_mode='residual')
    ctl.references=np.array([[.3,.4,0.]])
    ctl.cfg={'action_scales':[.2,.02,.02]}
    command=ctl.decode(np.array([[.5,0.,0.]]))
    np.testing.assert_allclose(np.cross(command,ctl.references),0.,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(command,axis=1),.6)


def test_finished_and_tpg_held_agents_keep_original_execution():
    ep=ControlEpisode(2,'mca_m1_lvo',2168000000,flow_inlet_mm_s=.05)
    try:
        est=ep.observe();target=np.array([-1,-1]);rule=ep.ctl.act(target,est);hold=np.array([False,True])
        ctl=SafeLocalSAC(ep,actor=ImageActor());ctl.observe(ep,est,target,rule,hold)
        expected=ctl.base.copy();expected[hold]=0.
        output=ctl.execute(ep,est,ctl.references+np.array([[.2,.02,.02],[.2,.02,.02]]))
        np.testing.assert_array_equal(output,expected)
        assert not ep.station_keeping
    finally:ep.close()


def test_zero_actor_recovers_original_curved_vessel_regression_episode():
    ep=ControlEpisode(3,'ica_siphon',2400610030,flow_inlet_mm_s=.05)
    try:
        ctl=SafeLocalSAC(ep,actor=ImageActor())
        while True:
            est=ep.observe();target=ep.plan_targets_now(est);rule=ep.ctl.act(target,est);hold=ep.hold(est)
            done,_=ep.step(est,ctl(ep,est,target,rule,hold),hold)
            if done:break
        row=ep.row('safe_initial')
        assert row['strict_success'] and row['removal']==1. and row['wall_contact_s']==0.
        np.testing.assert_allclose(row['elapsed_s'],51.1,rtol=0,atol=1e-9)
        np.testing.assert_allclose(row['tpg_hold_s'],13.9,rtol=0,atol=1e-9)
    finally:ep.close()
