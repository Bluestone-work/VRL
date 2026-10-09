from types import SimpleNamespace

import numpy as np
import pytest

from marl.hierarchical_navigation import (
    EventReplanner, NavigationConfig, SemanticAvoidance, WallRecoveryExecutor, relaxed_success, trajectory_cost, tube_clearance,
)


def straight_map(radius=1.):
    return SimpleNamespace(a=np.array([[-10., 0., 0.]]), ab=np.array([[20., 0., 0.]]),
                           l2=np.array([400.]), r0=np.array([radius]), r1=np.array([radius]), neigh=None)


def test_hard_union_clearance_is_not_inflated_by_smoothing():
    points = np.array([[[0., .9, 0.], [1., 0., 0.]]])
    np.testing.assert_allclose(tube_clearance(points, straight_map(), .1, 0), [[0., .9]], atol=1e-12)


def test_local_model_prefers_progress_when_safe_and_avoids_obstacle():
    cfg = NavigationConfig(switching_weight=0.)
    commands = np.array([[1., 0., 0.], [.6, .8, 0.], [0., 0., 0.]])
    direction, position, drift = np.array([1., 0., 0.]), np.zeros(3), np.zeros(3)
    costs, _, _ = trajectory_cost(commands, direction, position, drift, 1., [], straight_map(), .08, 0, cfg)
    assert costs.argmin() == 0
    costs, _, _ = trajectory_cost(commands, direction, position, drift, 1.,
                                  [(np.array([.5, 0., 0.]), np.zeros(3), .2)], straight_map(), .08, 0, cfg)
    assert costs.argmin() == 1


def test_wall_risk_discourages_outward_motion():
    commands = np.array([[.6, .8, 0.], [1., 0., 0.], [.6, -.8, 0.]])
    costs, wall, _ = trajectory_cost(commands, np.array([1., 0., 0.]), np.array([0., .85, 0.]),
                                     np.zeros(3), 1., [], straight_map(), .08, 0, NavigationConfig())
    assert wall[0] < 0
    assert costs.argmin() != 0


def test_stall_replanning_has_cooldown_and_excludes_intentional_hold():
    replanner = EventReplanner(1)
    calls = []
    controller = SimpleNamespace(_plan=lambda *args: calls.append(args))
    estimate = SimpleNamespace(pos=np.zeros((1, 3)), active=np.array([True]))
    for step in range(81):
        replanner.update(step*.1, estimate, [0], np.array([[1., 0., 0.]]), [False], controller)
    assert len(calls) == 3
    assert all(event['reason'] == 'commanded_stall' for event in replanner.events)
    for step in range(81, 121):
        replanner.update(step*.1, estimate, [0], np.array([[1., 0., 0.]]), [True], controller)
    assert len(calls) == 3


def test_net_zero_but_real_excursion_is_not_stall():
    replanner = EventReplanner(1)
    calls = []
    controller = SimpleNamespace(_plan=lambda *args: calls.append(args))
    estimate = SimpleNamespace(pos=np.zeros((1, 3)), active=np.array([True]))
    for step in range(21):
        estimate.pos[0, 0] = .3*np.sin(step*np.pi/20)
        replanner.update(step*.1, estimate, [0], np.array([[1., 0., 0.]]), [False], controller)
    assert not calls


def test_relaxed_metric_does_not_drop_loss_static_collision_or_peer_safety():
    row = dict(task_success=True, lost=0, obstacle_events_static=0, wall_contact_s=4.,
               obstacle_events_dynamic=5, robot_pair_contact_s=0., spacing={'spacing_compliant': True})
    assert relaxed_success(row)
    for field, value in [('lost', 1), ('obstacle_events_static', 1), ('wall_contact_s', 5.),
                         ('robot_pair_contact_s', .1), ('spacing', {'spacing_compliant': False})]:
        assert not relaxed_success(dict(row, **{field: value}))


def test_configuration_rejects_invalid_prediction_timing():
    with pytest.raises(ValueError):
        NavigationConfig(horizon_s=0.)


def test_guard_preserves_safe_reference_and_holds_without_online_truth():
    sensor = SimpleNamespace(a=np.array([[-10., 0., 0.]]), ab=np.array([[20., 0., 0.]]),
                             healthy=np.ones(2), ends=np.array([[0, 1]]), env=SimpleNamespace())
    executor = WallRecoveryExecutor(sensor, 1, .08)
    estimate = SimpleNamespace(pos=np.zeros((1, 3)), edge=np.array([0]), active=np.array([True]))
    controller = SimpleNamespace(frames=lambda estimate: np.eye(3)[None], nominal=np.array([[1., 0., 0.]]))
    reference = np.array([[1., 0., 0.]])
    np.testing.assert_array_equal(executor.act(0., estimate, controller, reference, [False]), reference)
    np.testing.assert_array_equal(executor.act(0., estimate, controller, reference, [True]), np.zeros((1, 3)))
    estimate.pos[0, 1] = .9
    corrected = executor.act(0., estimate, controller, np.array([[.6, .8, 0.]]), [False])
    assert corrected[0, 1] < .8
    assert np.linalg.norm(corrected) <= 1.


def test_lateral_guard_does_not_cancel_global_route_heading():
    sensor = SimpleNamespace(a=np.array([[-10., 0., 0.]]), ab=np.array([[20., 0., 0.]]),
                             healthy=np.ones(2), ends=np.array([[0, 1]]), env=SimpleNamespace())
    executor = WallRecoveryExecutor(sensor, 1, .08)
    estimate = SimpleNamespace(pos=np.array([[0., .95, 0.]]), edge=np.array([0]), active=np.array([True]))
    controller = SimpleNamespace(frames=lambda estimate: np.eye(3)[None], nominal=np.array([[.6, .8, 0.]]))
    reference = controller.nominal.copy()
    np.testing.assert_array_equal(executor.act(0., estimate, controller, reference, [False]), reference)


def test_semantic_avoidance_ignores_obstacle_behind_and_commits_side():
    controller = SimpleNamespace(frames=lambda estimate: np.eye(3)[None], nominal=np.array([[1., 0., 0.]]))
    estimate = SimpleNamespace(active=np.array([True]), obstacles=[[(np.array([-.3, .1, 0.]), np.zeros(3), .1)]])
    policy = SemanticAvoidance(1, .08)
    np.testing.assert_array_equal(policy.act(0., estimate, controller), controller.nominal)
    estimate.obstacles = [[(np.array([.4, .1, 0.]), np.zeros(3), .1)]]
    first = policy.act(.1, estimate, controller)
    assert first[0, 1] < 0
    estimate.obstacles = [[(np.array([.4, -.1, 0.]), np.zeros(3), .1)]]
    np.testing.assert_allclose(policy.act(.2, estimate, controller), first)
