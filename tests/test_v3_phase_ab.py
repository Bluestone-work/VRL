"""Phase A/B checks: deployment parking, N-independent base tasks, metrics, exact search."""
import itertools
import math

import numpy as np
import pytest

from marl.v3.configuration import Planner, ordered_tour_sets, set_partitions
from marl.v3.metrics import FourMetricRecorder
from marl.v3.scene import BaseTask, SceneBuilder
from marl.v3.vessel_graph import VesselGraph

TASK = BaseTask('mca_m1_lvo', 3100000001, 3, 'dispersed', 'pulsatile')


@pytest.fixture(scope='module')
def scene():
    sb = SceneBuilder(TASK)
    yield sb
    sb.close()


def test_partition_counts():
    stirling = {(4, 1): 1, (4, 2): 7, (4, 3): 6, (4, 4): 1, (5, 2): 15, (5, 3): 25}
    for (m, k), v in stirling.items():
        assert sum(1 for _ in set_partitions(range(m), k)) == v
    lah = {(4, 1): 24, (4, 2): 36, (4, 3): 12, (5, 4): 20}
    for (m, k), v in lah.items():
        assert sum(1 for _ in ordered_tour_sets(m, k)) == v


def test_base_task_identical_across_cluster_number(scene):
    from scripts.benchmark_lysis import LysisEpisode
    envs = []
    for n in (1, 2, 3):
        f = scene.env_factory([scene.root_station] * n)
        ep = LysisEpisode(n, TASK.anatomy, TASK.seed, horizon=300., sensing='image', tpg=False, navigation=True,
                          flow_inlet_mm_s=.05, dynamic_obstacle_count=TASK.obstacles, env_factory=f)
        envs.append(ep)
    for ep in envs[1:]:
        assert np.array_equal(ep.env.transport.points, envs[0].env.transport.points)
        assert np.array_equal(ep.env.clot_stations, envs[0].env.clot_stations)
        assert np.array_equal(ep.env.positions_mm[ep.n:], envs[0].env.positions_mm[1:])
        assert ep.manifest['base_task_hash'] == envs[0].manifest['base_task_hash']
        assert ep.env.flow_model.driving_pressure == envs[0].env.flow_model.driving_pressure
    for ep in envs:
        ep.close()


def test_parked_cluster_does_not_move_and_is_not_lost(scene):
    from scripts.benchmark_lysis import LysisEpisode
    f = scene.env_factory([scene.root_station, scene.root_station])
    ep = LysisEpisode(2, TASK.anatomy, TASK.seed, horizon=300., sensing='image', tpg=False, navigation=True,
                      flow_inlet_mm_s=.1, dynamic_obstacle_count=TASK.obstacles, env_factory=f)
    env = ep.env
    assert env.parked.all() and not env.active[:2].any()
    env.release(0)
    p1 = env.positions_mm[1].copy()
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.zeros((2, 3)))
        assert not term
    assert np.array_equal(env.positions_mm[1], p1)
    assert info['lost_robots'] == 0 and info['parked_robots'] == 1
    env.release(1)
    assert env.active[1] and not env.parked[1]
    ep.close()


def test_metrics_definitions():
    rec = FourMetricRecorder(2, 2, d_safe=2., body_radius=.08, deadline=10., tol=.3, target_owner=[0, 1])
    z = np.zeros(2)
    for k in range(10):
        pos = np.array([[0., 0, 0], [1.5 if k < 3 else 5., 0, 0]])
        rec.update(k * .1 + .1, .1, positions=pos, active=[True, True], wall_s=z, obstacle_contact_s=z,
                   obstacle_events=[0, 0], lost_now=[False, False], clearance=[1., 1.], hold=[False, False],
                   shield_changed=[k == 0, False], wall_rewritten=[False, False], cmd_world=np.ones((2, 3)),
                   actual_velocity=np.ones((2, 3)), delivered_mask=z)
    rec.mark_arrival(0, 4.); rec.mark_arrival(1, 6.)
    s = rec.summary()
    assert s['success'] and s['safe_success'] and s['conflict']
    assert math.isclose(s['spacing_violation_pair_s'], .3, abs_tol=1e-9) and s['conflict_events'] == 1
    assert s['T_last_deadline_scored_s'] == 6. and s['shield_intervention_steps'] == 1
    rec2 = FourMetricRecorder(1, 1, d_safe=2., body_radius=.08, deadline=10., tol=.3, target_owner=[0])
    rec2.update(.1, .1, positions=np.zeros((1, 3)), active=[True], wall_s=[.05], obstacle_contact_s=[0.],
                obstacle_events=[0], lost_now=[False], clearance=[.1], hold=[False], shield_changed=[False],
                wall_rewritten=[False], cmd_world=np.ones((1, 3)), actual_velocity=np.ones((1, 3)), delivered_mask=[0.])
    rec2.mark_arrival(0, 3.)
    s2 = rec2.summary()
    assert s2['success'] and not s2['safe_success'] and s2['low_contact_success']   # contact is not "safe"
    rec3 = FourMetricRecorder(1, 2, d_safe=2., body_radius=.08, deadline=10., tol=.3, target_owner=[0, 0])
    rec3.mark_arrival(0, 3.)
    s3 = rec3.summary()
    assert not s3['success'] and s3['T_last_deadline_scored_s'] == 10. and s3['target_coverage'] == .5


def test_exact_search_matches_brute_force(scene):
    g = VesselGraph(scene.probe)
    pl = Planner(g, scene.targets, scene.root_station)
    best, log = pl.exact(2)
    assert log['certificate'] == 'exact_within_model'
    brute = np.inf
    for tours in ordered_tour_sets(pl.m, 2):
        for nodes in itertools.product(*[pl.candidates_for(t[0]) for t in tours]):
            for order in itertools.permutations(range(2)):
                brute = min(brute, pl.evaluate(tours, list(nodes), list(order))['J'])
    assert math.isclose(best[0], brute, rel_tol=1e-9)


def test_local_search_never_beats_exact(scene):
    g = VesselGraph(scene.probe)
    for k in (1, 2, 3):
        pe = Planner(g, scene.targets, scene.root_station); pl = Planner(g, scene.targets, scene.root_station)
        e, _ = pe.exact(k); l, _ = pl.local(k)
        assert l[0] >= e[0] - 1e-9


def test_long_catheter_transit_is_not_a_deadlock(scene):
    g = VesselGraph(scene.probe)
    pl = Planner(g, scene.targets, scene.root_station)
    deep = max(pl.candidates_for(0), key=lambda s: g.depth[s])
    pl.v_c = float(g.dist[pl.root_station, deep]) / 100.   # catheter transit of 100 s (> 60 s stall window)
    tours = [tuple(range(pl.m))]
    assert pl.catheter_time(pl.root_station, deep) > 60.
    sim = pl.simulate(tours, [deep], [0], [0])
    assert sim['release_times'][0] is not None and not sim['deadlock']
    assert abs(sim['release_times'][0] - pl.catheter_time(pl.root_station, deep)) <= pl.dt + 1e-9
