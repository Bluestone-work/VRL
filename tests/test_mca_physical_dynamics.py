"""Analytical transport, convergence, boundary and physical Gym regressions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from environments.mca_physiology import PressureDrivenTreeFlow
from environments.mca_physical_dynamics import PhysicalTubeTransport
from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from environments.vessel_tree_generator import VesselSegment, segments_to_vessel_tree


def pipe(length=20., radius=1.):
    seg = VesselSegment(start=np.zeros(3), end=np.array([length, 0., 0.]),
                        radius_prox=radius, radius_dist=radius, parent_idx=None,
                        generation=0, n_stations=int(length)+1)
    tree = segments_to_vessel_tree([seg], min_radius=0, fit_unit_cube=False)
    tree.physical_mm_per_unit = 1.
    return tree


def setup_transport(*, fraction=.1, inlet=.6*np.pi, length=20., **kwargs):
    flow = PressureDrivenTreeFlow(pipe(length), 1., inlet_flow_ml_min=inlet)
    return PhysicalTubeTransport(flow, spatial_fraction=fraction, **kwargs), flow.solve()


def move(transport, solution, positions, command=None, dt=.05, active=None):
    positions = np.asarray(positions, float).reshape(-1, 3)
    n = len(positions)
    command = np.zeros_like(positions) if command is None else np.asarray(command, float).reshape(n, 3)
    return transport.advance(positions, transport.nearest_edges(positions), np.full(n, .08),
                             command, np.ones(n, bool) if active is None else active, solution, dt)


def test_uniform_pipe_exact_advection_and_path_in_mm():
    transport, solution = setup_transport()
    result = move(transport, solution, [[2., 0., 0.], [2., .4, 0.]])
    np.testing.assert_allclose(result.positions_mm[:, 0], [3., 2.+.84], atol=1e-10)
    np.testing.assert_allclose(result.path_mm, [1., .84], atol=1e-10)
    assert result.substeps > 1
    assert not result.wall_contact_s.any()


def test_control_interval_is_world_hold_not_residual_controller():
    transport, solution = setup_transport()
    result = move(transport, solution, [[2., 0., 0.]], [[-1., 0., 0.]])
    assert result.positions_mm[0, 0] == pytest.approx(2 + 19*.05)


def junction_transport(points, links, *, radii=None, fraction=.1):
    points = np.asarray(points, float)
    graph = [[] for _ in points]
    for start, end in links:
        length = float(np.linalg.norm(points[start] - points[end]))
        graph[start].append((end, length))
        graph[end].append((start, length))
    tree = SimpleNamespace(points=points, n_stations=len(points), inlet_station=0,
                           radii=np.ones(len(points)) if radii is None else np.asarray(radii, float),
                           station_graph=graph)
    flow = PressureDrivenTreeFlow(tree, 1., inlet_flow_ml_min=1e-12)
    return PhysicalTubeTransport(flow, spatial_fraction=fraction), flow.solve()


@pytest.mark.parametrize('sign', [-1, 1])
def test_internal_pipe_stations_are_not_caps(sign):
    transport, solution = setup_transport(inlet=1e-12)
    start = np.array([[10.25, .3, 0.]])
    result = move(transport, solution, start, [[sign*5., 0., 0.]], dt=.8)
    np.testing.assert_allclose(result.positions_mm, start + [[sign*4., 0., 0.]], atol=1e-10)
    assert not result.wall_contact_s.any() and not result.blocked_s.any()
    assert result.path_mm[0] == pytest.approx(4.)


@pytest.mark.parametrize('side', [-1, 1])
def test_junction_uses_forward_incident_branch_and_preserves_world_action(side):
    transport, solution = junction_transport(
        [[0, 0, 0], [1, 0, 0], [2, .5, 0], [2, -.5, 0]], [(0, 1), (1, 2), (1, 3)])
    result = move(transport, solution, [[.9, 0, 0]], [[1., side*.25, 0]], dt=.3)
    np.testing.assert_allclose(result.positions_mm, [[1.2, side*.075, 0]], atol=1e-10)
    assert transport.ends[result.edge[0], 1] == (2 if side == 1 else 3)
    # Reversing an action must allow an upstream transition to the parent.
    returned = transport.advance(result.positions_mm, result.edge, np.array([.08]),
                                 np.array([[-1., 0., 0.]]), result.active, solution, .4)
    np.testing.assert_allclose(returned.positions_mm, [[.8, side*.075, 0]], atol=1e-10)
    assert returned.edge[0] == 0


def test_coincident_junction_checks_child_aperture_and_records_blocking():
    transport, solution = junction_transport(
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]],
        [(0, 1), (1, 2), (2, 3)], radii=[1., 1., .04, .04])
    result = move(transport, solution, [[.9, 0, 0]], [[1., 0., 0.]], dt=.4)
    assert result.edge[0] == 0 and result.positions_mm[0, 0] <= 1 + 1e-10
    assert result.blocked_s[0] > 0 and result.active[0]
    # A smaller passive body can traverse the same aperture, with no radius floor.
    result = transport.advance(np.array([[.9, 0., 0.]]), np.array([0]), np.array([.02]),
                               np.array([[1., 0., 0.]]), np.array([True]), solution, .4)
    assert result.edge[0] == 1
    assert result.positions_mm[0, 0] == pytest.approx(1.3)


def test_inward_motion_in_new_tube_cap_does_not_switch_back_to_parent():
    transport, solution = junction_transport(
        [[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, -2, 0]], [(0, 1), (1, 2), (1, 3)])
    # This valid cap state occurs when an offset centre enters a bent tube.
    # Even while t remains negative it is entering, not returning upstream.
    result = transport.advance(np.array([[1., -.01, 0.]]), np.array([1]), np.array([.08]),
                               np.array([[0., 1., 0.]]), np.array([True]), solution, .005)
    assert result.edge[0] == 1
    np.testing.assert_allclose(result.positions_mm, [[1., -.005, 0.]], atol=1e-10)


def test_outlet_crossing_time_and_no_respawn():
    transport, solution = setup_transport()
    result = move(transport, solution, [[19.8, 0., 0.]], dt=.05)
    assert not result.active[0]
    assert result.exit_time_s[0] == pytest.approx(.01, abs=1e-12)
    assert result.path_mm[0] == pytest.approx(.2)
    assert result.positions_mm[0, 0] == 20
    again = move(transport, solution, result.positions_mm, active=result.active)
    np.testing.assert_array_equal(again.positions_mm, result.positions_mm)
    assert again.path_mm[0] == 0
    assert again.substeps == 0


def test_inlet_is_open_boundary_too():
    transport, solution = setup_transport(inlet=1e-12)
    result = move(transport, solution, [[.02, 0., 0.]], [[-1., 0., 0.]])
    assert not result.active[0]
    assert result.exit_node[0] == transport.model.root
    assert result.exit_time_s[0] == pytest.approx(.02)


@pytest.mark.parametrize('fraction', [.1, .05, .025, .0125])
@pytest.mark.parametrize('outlet', [False, True])
def test_grazing_wall_exit_is_absorbed_at_open_aperture(fraction, outlet):
    transport, solution = setup_transport(inlet=1e-12, fraction=fraction)
    # The unconstrained path meets the exit plane outside radius .92. The
    # wall-constrained path is valid and must leave rather than fill an end cap.
    x, vx = (19.99, 1.) if outlet else (.01, -1.)
    result = move(transport, solution, [[x, .919, 0.]], [[vx, 1., 0.]], dt=.05)
    assert not result.active[0]
    assert result.exit_node[0] == (20 if outlet else 0)
    assert result.positions_mm[0, 0] == pytest.approx(20 if outlet else 0)
    assert result.positions_mm[0, 1] <= .92 + 1e-10
    assert result.exit_time_s[0] == pytest.approx(.01, abs=.0001)


def test_high_velocity_is_resolved_not_clipped():
    transport, solution = setup_transport(inlet=600*np.pi)
    result = move(transport, solution, [[2., 0., 0.]], dt=.002)
    assert not result.active[0]
    assert result.exit_time_s[0] == pytest.approx(18/20000, rel=1e-9)


def test_narrowing_blocks_finite_body_without_lumen_floor():
    transport, solution = setup_transport(inlet=1e-10)
    radii = solution['radius_mm'].copy()
    radii[10] = .01
    solution = transport.model.solve(radii)
    result = move(transport, solution, [[9., 0., 0.]], [[10., 0., 0.]], dt=.2)
    assert result.positions_mm[0, 0] < 10
    assert result.blocked_s[0] > 0
    assert result.active[0]
    assert solution['radius_mm'][10] == .01


def test_complete_occlusion_cannot_be_crossed_by_projection():
    transport, solution = setup_transport(inlet=.6*np.pi)
    radii = solution['radius_mm'].copy()
    radii[10] = 0
    solution = transport.model.solve(radii)
    result = move(transport, solution, [[9.5, 0., 0.]], [[10., 0., 0.]], dt=.2)
    assert result.positions_mm[0, 0] < 10
    assert result.blocked_s[0] > 0


def test_passive_bodies_share_flow_without_robot_propulsion():
    transport, solution = setup_transport()
    result = move(transport, solution, [[2., 0., 0.], [2., 0., 0.]],
                  [[1., 0., 0.], [0., 0., 0.]])
    assert result.positions_mm[0, 0] - result.positions_mm[1, 0] == pytest.approx(.05)


def test_wall_contact_uses_seconds_and_does_not_teleport_forward():
    transport, solution = setup_transport(inlet=1e-10)
    result = move(transport, solution, [[2., .90, 0.]], [[0., 1., 0.]], dt=.1)
    assert result.positions_mm[0, 1] == pytest.approx(.92)
    assert .07 < result.wall_contact_s[0] <= .1
    assert result.path_mm[0] == pytest.approx(.02, abs=1e-8)


def test_midpoint_spatial_refinement_converges_to_analytic_sheared_flow():
    # y(t)=.2+t; x'=20*(1-y^2). Integral is known, independent of implementation.
    t = .2
    exact = 2 + 20*(t - (.2**2*t + .2*t*t + t**3/3))
    errors = []
    for fraction in (.2, .1, .05):
        transport, solution = setup_transport(fraction=fraction)
        result = move(transport, solution, [[2., .2, 0.]], [[0., 1., 0.]], dt=t)
        errors.append(abs(result.positions_mm[0, 0] - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .004
    assert errors[0] / errors[1] > 3 and errors[1] / errors[2] > 3


def test_substep_budget_error_does_not_mutate_inputs():
    transport, solution = setup_transport(max_substeps=1)
    positions = np.array([[2., 0., 0.]])
    with pytest.raises(RuntimeError, match='NOT capped'):
        move(transport, solution, positions)
    np.testing.assert_array_equal(positions, [[2., 0., 0.]])


def env_config(**kwargs):
    base = DynamicsConfig(num_robots=1, particle_count=0, inlet_flow_ml_min=1e-10,
                          distal_resistance_ratio=0, initial_radius_fraction=1,
                          lysis_mass_per_s=.36, control_dt_s=.05, episode_duration_s=.2)
    return replace(base, **kwargs)


def test_lysis_rate_is_per_second_across_control_periods():
    masses = []
    for period in (.05, .1):
        env = MCAPhysicalEnv(env_config(control_dt_s=period), tree=pipe(), clot_stations=[5])
        env.reset(seed=42, options={'robot_positions_mm': np.array([[5., 0., 0.]])})
        while True:
            obs, reward, term, trunc, info = env.step(np.zeros((1, 3)))
            if term or trunc:
                break
        assert env.observation_space.contains(obs)
        assert not term and trunc
        masses.append(info['remaining_mass'])
    np.testing.assert_allclose(masses, [1-.36*.2]*2, atol=1e-10)


def test_all_lost_is_failure_not_success_and_requires_reset():
    env = MCAPhysicalEnv(env_config(inlet_flow_ml_min=.6*np.pi), tree=pipe(), clot_stations=[5])
    env.reset(seed=1, options={'robot_positions_mm': np.array([[19.8, 0., 0.]])})
    obs, _, term, trunc, info = env.step(np.zeros((1, 3)))
    assert term and not trunc and not info['success']
    assert info['termination_reason'] == 'all_robots_exited'
    assert info['remaining_mass'] == 1
    assert obs['agent_mask'].sum() == 0
    assert not obs['nodes'].any()
    assert info['robot_exit_time_s'][0] == pytest.approx(.01)
    with pytest.raises(RuntimeError, match='reset'):
        env.step(np.zeros((1, 3)))


def test_lost_robot_cannot_act_or_lyse_in_later_steps():
    env = MCAPhysicalEnv(env_config(num_robots=2, inlet_flow_ml_min=.6*np.pi),
                         tree=pipe(), clot_stations=[20])
    env.reset(seed=1, options={'robot_positions_mm': np.array([[19.8, 0., 0.], [2., 0., 0.]])})
    env.step(np.zeros((2, 3)))
    mass_after_exit = env.masses.copy()
    frozen = env.positions_mm[0].copy()
    env.step(np.array([[-1., 0., 0.], [0., 0., 0.]]))
    np.testing.assert_array_equal(env.positions_mm[0], frozen)
    np.testing.assert_array_equal(env.masses, mass_after_exit)


def test_success_requires_zero_mass_and_wins_over_horizon():
    env = MCAPhysicalEnv(env_config(lysis_mass_per_s=100), tree=pipe(), clot_stations=[5])
    env.reset(seed=1, options={'robot_positions_mm': np.array([[5., 0., 0.]])})
    _, _, term, trunc, info = env.step(np.zeros((1, 3)))
    assert term and not trunc and info['success']
    assert info['remaining_mass'] == 0


def test_environment_integration_failure_is_transactional():
    env = MCAPhysicalEnv(env_config(inlet_flow_ml_min=.6*np.pi, max_substeps_per_control=1),
                         tree=pipe(), clot_stations=[2])
    env.reset(seed=1, options={'robot_positions_mm': np.array([[2., 0., 0.]])})
    before, masses = env.positions_mm.copy(), env.masses.copy()
    with pytest.raises(RuntimeError, match='budget'):
        env.step(np.zeros((1, 3)))
    np.testing.assert_array_equal(env.positions_mm, before)
    np.testing.assert_array_equal(env.masses, masses)
    assert env.elapsed_s == 0


def test_reset_is_deterministic_and_clears_paths_and_masks():
    env = MCAPhysicalEnv(env_config(particle_count=2), tree=pipe(), clot_stations=[5])
    first, _ = env.reset(seed=42)
    start = env.positions_mm.copy()
    env.step(np.ones((1, 3)))
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(env.positions_mm, start)
    for k in first:
        np.testing.assert_array_equal(first[k], second[k])
    assert not env.path_mm.any() and env.active.all() and env.elapsed_s == 0


def test_partial_final_period_reaches_exact_physical_horizon():
    env = MCAPhysicalEnv(env_config(control_dt_s=.1, episode_duration_s=.25),
                         tree=pipe(), clot_stations=[10])
    env.reset(seed=1)
    for _ in range(3):
        _, _, term, trunc, info = env.step(np.zeros((1, 3)))
    assert trunc and not term
    assert info['elapsed_s'] == .25
    assert info['step_duration_s'] == pytest.approx(.05)


def test_invalid_actions_cannot_poison_state():
    env = MCAPhysicalEnv(env_config(), tree=pipe(), clot_stations=[10])
    env.reset(seed=1)
    for action in (np.full((1, 3), np.nan), np.zeros((3,))):
        with pytest.raises(ValueError):
            env.step(action)
    assert env.elapsed_s == 0


@pytest.mark.parametrize('field,value', [('robot_speed_mm_s', np.nan),
    ('control_dt_s', 0), ('num_robots', 0), ('particle_count', -1),
    ('initial_radius_fraction', 1.1), ('lysis_mass_per_s', -1),
    ('spatial_fraction', .5), ('assumption_provenance', '')])
def test_physical_configuration_rejects_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        replace(DynamicsConfig(), **{field: value})


def test_policy_bridge_runs_real_update_and_rejects_legacy_schema():
    import torch
    from marl.mca_physical_policy import (
        make_physical_agent, physical_policy_action, store_physical_transition,
    )
    torch.set_num_threads(1)
    env = MCAPhysicalEnv(env_config(num_robots=2), tree=pipe(), clot_stations=[5])
    obs, _ = env.reset(seed=42)
    agent = make_physical_agent(env, seed=42)
    for _ in range(4):
        raw, lp, value, execution, ctx, state = physical_policy_action(agent, env, obs)
        returned, _, term, trunc, info = env.step(execution)
        store_physical_transition(agent, obs, raw, lp, value, ctx, state, env,
                                   returned, info, term, trunc)
        obs = returned
    metrics = agent.update(n_epochs=1, batch_size=4)
    assert metrics and all(np.isfinite(v) for v in metrics.values())
    del agent.meta['observation_schema']
    with pytest.raises(RuntimeError, match='schema'):
        physical_policy_action(agent, env, obs)


def test_bridge_bootstraps_time_limit_but_not_robot_loss():
    from marl.mca_physical_policy import (
        make_physical_agent, physical_policy_action, store_physical_transition,
    )
    env = MCAPhysicalEnv(env_config(episode_duration_s=.05), tree=pipe(), clot_stations=[5])
    obs, _ = env.reset(seed=42)
    agent = make_physical_agent(env, seed=42)
    raw, lp, value, execution, ctx, state = physical_policy_action(agent, env, obs)
    returned, _, term, trunc, info = env.step(execution)
    assert trunc and not term
    store_physical_transition(agent, obs, raw, lp, value, ctx, state, env, returned, info, term, trunc)
    data = agent.buffer.get()
    assert data['dones'][0, 0] == 1 and data['terminals'][0, 0] == 0
