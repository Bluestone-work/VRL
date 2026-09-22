"""Connectivity-aware allocation (EXP_0008): capacity + overlap behaviour."""
import numpy as np
import pytest

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vector_env import VectorVascularEnv
from environments.vessel_geometry import resolve_pool
from marl.connectivity_allocator import (
    ALLOCATION_MODES,
    allocate,
    connectivity_assignment,
    flow_spread_assignment,
    nearest_assignment,
)


SCENARIOS = resolve_pool("anatomical")[:5]


def _env(scenario, robots=8, clots=3, **kw):
    return Vascular3DMARLEnv(
        scenario=scenario, scenario_pool=[scenario], randomize_scenario=False,
        num_robots=robots, num_clots=clots, horizon=60, robot_radius=0.0011,
        seed=42, **kw
    )


def _route_overlap(env, assignments):
    """Mean shared-edge fraction across assigned routes (same accounting the
    allocator reports): for each route, the edges used by MORE than one
    assigned route, divided by route length; averaged over robots."""
    edge_counts = {}
    total = 0.0
    for robot, target in enumerate(assignments):
        if target < 0:
            continue
        _distance, hop = env._route(int(target))
        cur = int(env.robot_stations[robot])
        path = [cur]
        for _ in range(4096):
            nxt = int(hop[cur])
            if nxt == cur:
                break
            path.append(nxt)
            cur = nxt
            if cur == int(env.clot_stations[target]):
                break
        for s in path:
            edge_counts[s] = edge_counts.get(s, 0) + 1
        shared = sum(edge_counts[s] - 1 for s in path)
        total += shared / max(len(path), 1)
    return total / max(int((assignments >= 0).sum()), 1)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_capacity_respected_when_alternative_exists(scenario):
    """No clot exceeds the lysis saturation while an unsaturated one exists."""
    env = _env(scenario)
    env.reset(seed=1000)
    out = connectivity_assignment(env)
    assert out.target_load.size > 1
    if out.distinct_targets > 1:
        assert out.target_load.max() <= 4, out.target_load


def test_single_clot_capacity_is_total():
    """With one live clot every robot is assigned (there is nowhere else)."""
    env = _env("pulmonary_saddle", robots=8, clots=1, randomize_clots=False)
    env.reset(seed=1000)
    out = connectivity_assignment(env)
    assert out.target_load.size == 1
    assert int(out.target_load[0]) == 8
    assert np.all(out.assignments >= 0)


def test_connectivity_beats_nearest_on_overlap():
    """Corridor overlap: connectivity-aware <= nearest on every scenario."""
    for scenario in SCENARIOS:
        env = _env(scenario)
        env.reset(seed=1000)
        near = allocate(env, "nearest")
        conn = allocate(env, "connectivity_aware")
        assert _route_overlap(env, conn.assignments) <= \
            _route_overlap(env, near.assignments) + 1e-9


def test_mean_overlap_connectivity_leq_flow_spread():
    """On average across scenarios/episodes connectivity matches or beats
    the flow_spread baseline (nearest is dominated far more strongly)."""
    conn_total, flow_total, near_total = 0.0, 0.0, 0.0
    episodes = 0
    for scenario in SCENARIOS:
        env = _env(scenario)
        for ep in range(10):
            env.reset(seed=3000 + ep)
            near = allocate(env, "nearest")
            flow = allocate(env, "flow_spread")
            conn = allocate(env, "connectivity_aware")
            near_total += _route_overlap(env, near.assignments)
            flow_total += _route_overlap(env, flow.assignments)
            conn_total += _route_overlap(env, conn.assignments)
            episodes += 1
    assert conn_total / episodes <= near_total / episodes
    # flow_spread is a strong baseline; connectivity should not be materially
    # worse (allow 2% slack for scenarios where distance dominates).
    assert conn_total <= flow_total * 1.02


def test_switching_penalty_stabilises_assignments():
    env = _env("pulmonary_saddle", robots=6)
    env.reset(seed=1000)
    a1 = connectivity_assignment(env)
    a2 = connectivity_assignment(env, previous_assignments=a1.assignments)
    stable = int((a1.assignments == a2.assignments).sum())
    assert stable >= 5  # at most one robot re-tasks on a static scene


def test_allocate_dispatch_and_mode_validation():
    env = _env(SCENARIOS[0])
    env.reset(seed=7)
    for mode in ALLOCATION_MODES:
        out = allocate(env, mode)
        assert out.assignments.shape == (8,)
        assert np.all(out.assignments >= -1)
    with pytest.raises(ValueError, match="unknown allocation mode"):
        allocate(env, "bogus")


def test_vector_env_assignment_override_and_reset():
    env = VectorVascularEnv(
        n_envs=2, scenario="pulmonary_saddle", num_robots=5, num_clots=3,
        horizon=50, seed=11, robot_radius=0.0011, randomize_scenario=False,
    )
    env.reset_all()
    env.set_task_assignments(np.zeros((2, 5), np.int32))
    assert np.all(env._assign() == 0)
    # dead clot guard: assignment to a cleared clot becomes -1
    env.clot_masses[:, 0] = 0.0
    assert np.all(env._assign() == -1)
    env.clear_task_assignments()
    # nearest rule restored
    assert np.all(env._assign() >= 0)
    # shape guard
    with pytest.raises(ValueError):
        env.set_task_assignments(np.zeros((3, 5), np.int32))
