"""Observation boundary, paired scenes, and independently measured safety."""
from dataclasses import replace
import numpy as np
import pytest

from marl.multicluster import MultiClusterConfig, MultiClusterController, ClusterObservation, balanced_cluster_ids
from marl.partial_obs import CLOT0
from marl.multicluster_observation import ClusterSensorAdapter
from scripts.multicluster_protocol import paired_environment, SpacingTracker, attach_spacing_monitor
from scripts.evaluate_multicluster import DEFAULT_CONFIG, run_episode
from environments.mca_physical_env import DynamicsConfig


def packet(n=2, gap=1.8):
    nav = np.zeros((n, 111), np.float32)
    nav[:, CLOT0:CLOT0+6] = [1., .1, 1., 0., 0., 1.]
    rel = np.zeros((n, n, 3))
    visible = np.zeros((n, n), bool)
    if n > 1:
        rel[0, 1, 0], rel[1, 0, 0] = gap, -gap
        visible[0, 1] = visible[1, 0] = True
    return ClusterObservation(nav, np.tile([3,-1,-1,-1], (n,1)), rel, np.zeros_like(rel), visible, np.ones(n,bool))


@pytest.mark.parametrize('kwargs', [dict(clusters=True), dict(clusters=0), dict(wait_horizon_s=-1),
    dict(min_spacing_mm=float('nan')), dict(observation_noise=-.1), dict(position_noise_mm=-1),
    dict(peer_sensing_radius_mm=1.), dict(method='single_sequential',clusters=2)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MultiClusterConfig(**kwargs)


def test_partition_counts():
    np.testing.assert_array_equal(balanced_cluster_ids(5,2),[0,0,0,1,1])
    with pytest.raises(ValueError): balanced_cluster_ids(1,2)


def test_policy_reads_observations_only_and_is_deterministic():
    c = MultiClusterController(MultiClusterConfig())
    assert not hasattr(c, 'env')
    o = packet()
    a, _ = c.act(o)
    c.reset(42)
    b, _ = c.act(o)
    np.testing.assert_array_equal(a,b)
    assert np.all(np.linalg.norm(a,axis=1)<=1+1e-12)
    # Bodies initially too close get separating commands, not blind stopping.
    assert (a[1]-a[0])[0] > 0


def test_no_invisible_teammate_leakage():
    o = packet()
    o.peer_visible[:] = False
    a, _ = MultiClusterController(MultiClusterConfig()).act(o)
    o.peer_relative_mm[:] = 1e9
    o.peer_relative_velocity_mm_s[:] = -1e9
    b, _ = MultiClusterController(MultiClusterConfig()).act(o)
    np.testing.assert_array_equal(a,b)


def test_sequential_target_identity_survives_slot_reordering():
    o = packet(1)
    c = MultiClusterController(MultiClusterConfig(method='single_sequential', clusters=1))
    c.act(o)
    assert c.current_target[0] == 3
    o.clot_ids[0,:2] = [2,3]
    o.navigation[0,CLOT0+6:CLOT0+12] = [1.,.1,-1.,0.,0.,1.]
    a, _ = c.act(o)
    assert c.current_target[0] == 3 and a[0,0] < 0
    o.navigation[0,CLOT0+6] = 0
    c.act(o)
    assert c.current_target[0] == 2


def test_measured_spacing_catches_interior_crossing():
    before=np.array([[0.,0,0],[2.,0,0]])
    after=np.array([[2.,0,0],[0.,0,0]])
    t=SpacingTracker(before,np.ones(2,bool),1.)
    t.begin_step();t.update(before,after,np.ones(2,bool),np.ones(2),1.);t.end_step()
    assert t.minimum == pytest.approx(0.)
    assert t.pair_violation_s == pytest.approx(.5)
    assert t.control_violation_steps == 1
    # A historical violation never causes a later separated step to be counted.
    t.begin_step();t.update(before,before,np.ones(2,bool),np.ones(2),1.);t.end_step()
    assert t.control_violation_steps == 1


def test_spacing_respects_exits_and_no_pairs():
    before=np.array([[0.,0,0],[2.,0,0]])
    after=np.array([[0.,0,0],[1.5,0,0]])
    t=SpacingTracker(before,np.ones(2,bool),1.)
    t.begin_step();t.update(before,after,np.ones(2,bool),np.array([1.,.25]),1.);t.end_step()
    assert t.minimum == pytest.approx(1.5)
    assert t.summary()['spacing_compliant']
    assert SpacingTracker(before[:1],np.ones(1,bool),1).summary()['minimum_spacing_mm'] is None


def base_cfg():
    return replace(DynamicsConfig.from_json(DEFAULT_CONFIG),episode_duration_s=.5)


def test_scene_identity_and_catalytic_budget_across_arms():
    manifests=[]
    for n,budget in [(1,'fixed_total'),(2,'fixed_total'),(3,'fixed_total'),(2,'per_cluster')]:
        e,m=paired_environment(base_cfg(),n,1302000000,budget=budget)
        manifests.append(m)
        assert e.config.action_prior == 'none'
        assert e.config.command_speed == 'bounded'
        assert np.array_equal(e.positions_mm[:n],np.asarray(m['shared_scene']['start_pool_mm'])[:n])
        assert m['aggregate_lysis_capacity_mass_per_s'] == pytest.approx(.36 if budget=='fixed_total' else .72)
        e.close()
    assert len({m['scenario_hash'] for m in manifests}) == 1


def test_adapter_masks_out_of_range_peers_and_records_executed_command():
    e,_=paired_environment(base_cfg(),2,1302000000)
    sensor=ClusterSensorAdapter(e,MultiClusterConfig(observation_noise=0.,position_noise_mm=0.))
    o=sensor.observe()
    assert np.all(o.peer_relative_mm[~o.peer_visible]==0)
    executed=np.array([[.25,0,0],[-.1,0,0]])
    sensor.execute(executed)
    np.testing.assert_allclose(sensor.observe().navigation[:,:3],executed,atol=1e-7)
    e.close()


def test_sensor_does_not_read_routes_or_assignment(monkeypatch):
    e,_=paired_environment(base_cfg(),2,1302000000)
    sensor=ClusterSensorAdapter(e,MultiClusterConfig())
    def forbidden(*a,**k): raise AssertionError('privileged route accessor')
    for name in ('_route_directions','_assigned_targets','_target_distances','_solve_assignment','_observation','_apply_action_prior'):
        monkeypatch.setattr(e,name,forbidden)
    e.routes=e._route_next_hop=None
    o=sensor.observe()
    a,_=MultiClusterController(MultiClusterConfig()).act(o)
    assert np.isfinite(a).all()
    e.close()


@pytest.mark.parametrize('method,n',[('single_sequential',1),('single_route',1),('multi_parallel',2),('multi_unshielded',3)])
def test_episode_populates_safety_and_provenance(method,n):
    row=run_episode(base_cfg(),MultiClusterConfig(method=method,clusters=n),1302000000)
    assert row['status']=='completed' and not row['sealed_test_used']
    assert row['actual_config']['action_prior']=='none'
    assert row['elapsed_s']==pytest.approx(.5)
    assert row['local_observation']==(method!='single_route')
    assert not row['controller']['safety_certificate']
    assert row['reset_info']['scenario_hash']
    if row['cluster_safe_success']:
        assert row['safe_collision_free'] and row['spacing_compliant']


def test_metric_tap_does_not_change_physics():
    a,_=paired_environment(base_cfg(),2,1302000000)
    b,_=paired_environment(base_cfg(),2,1302000000)
    t=SpacingTracker(b.positions_mm[:2],b.active[:2],2.)
    attach_spacing_monitor(b,t)
    for _ in range(4):
        command=np.array([[.2,0,0],[-.1,0,0]])
        a.step(command);b.step(command)
        np.testing.assert_array_equal(a.positions_mm,b.positions_mm)
        np.testing.assert_array_equal(a.masses,b.masses)
    a.close();b.close()


def test_crashes_preserve_denominators_and_block_performance_claim():
    from scripts.evaluate_multicluster_isolated import summarize_attempts
    from scripts.safe_metrics import episode_metrics,WallTracker
    metrics=episode_metrics(dict(success=True,collision_free_success=True,remaining_mass=0.,elapsed_s=1.),WallTracker(1),1.)
    r=summarize_attempts([dict(status='completed',cluster_safe_success=True,**metrics),dict(status='process_exit',returncode=-11)])
    assert r['requested_episodes']==2 and r['completed_episodes']==1 and r['failed_episodes']==1
    assert r['cluster_safe_success_missing_bounds']==[.5,1.]
    assert not r['performance_comparison_admissible']
    assert summarize_attempts([])['cluster_safe_success_missing_bounds'] is None


def test_repeat_episode_matches_final_state_and_safety():
    cfg=MultiClusterConfig(clusters=2)
    a=run_episode(base_cfg(),cfg,1302000000)
    b=run_episode(base_cfg(),cfg,1302000000)
    assert a['final_state_hash']==b['final_state_hash']
    assert a['spacing_violation_pair_s']==b['spacing_violation_pair_s']


def test_spacing_filter_does_not_relabel_prediction_as_realized_safety():
    policy=MultiClusterController(MultiClusterConfig())
    _,summary=policy.act(packet(gap=1.5))
    assert 'minimum_spacing_mm' not in summary
    assert 'spacing_violation_steps' not in summary
    assert summary['predicted_conflict_agent_steps'] > 0


def test_step_guard_is_not_an_episode_completion_or_timeout():
    row=run_episode(base_cfg(),MultiClusterConfig(),1302000000,max_steps=1)
    assert row['development_guard_hit']
    assert row['termination_reason']=='development_step_guard'
    assert not row['cluster_safe_success'] and not row['timeout']


def test_reserved_layout_is_rejected_before_environment_reset():
    with pytest.raises(ValueError,match='registered evaluation seeds'):
        paired_environment(base_cfg(),2,990000000)
