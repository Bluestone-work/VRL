from types import SimpleNamespace

import numpy as np

from marl.progress_navigation import AllObstacleNavigator, ProgressConfig, RouteProgressMonitor, remaining_route_mm, route_subgoal


def state():
    controller = SimpleNamespace(route=[np.array([0, 1, 2])], prog=np.array([0]),
                                 pts=np.array([[0., 0., 0.], [5., 0., 0.], [10., 0., 0.]]), body=.08,
                                 env=SimpleNamespace(clot_positions_mm=np.array([[10., 0., 0.]])))
    estimate = SimpleNamespace(pos=np.zeros((1, 3)), active=np.array([True]),
                               obstacles=[[(np.array([1., .2, 0.]), np.zeros(3), .5)]])
    return controller, estimate


def test_connected_route_remaining_and_subgoal():
    controller, estimate = state()
    assert remaining_route_mm(controller, 0, estimate.pos[0]) == 10.
    np.testing.assert_array_equal(route_subgoal(controller, 0, 6.), [10., 0., 0.])


def test_spatial_oscillation_triggers_route_frontier_stall():
    controller, estimate = state()
    monitor = RouteProgressMonitor(1)
    for step in range(81):
        estimate.pos[0, 1] = .4*np.sin(step*.5)
        monitor.update(step*.1, estimate, [0], [False], controller)
    assert len(monitor.events) == 1
    assert monitor.events[0]['reason'] == 'route_frontier_stall'


def test_real_progress_prevents_loop_trigger_even_if_route_station_changes():
    controller, estimate = state()
    monitor = RouteProgressMonitor(1)
    for step in range(91):
        estimate.pos[0, 0] = step*.1
        controller.prog[0] = int(estimate.pos[0, 0] >= 2.5)
        monitor.update(step*.1, estimate, [0], [False], controller)
    assert not monitor.events


def test_goal_lysis_and_coordinated_hold_are_not_route_deadlock():
    controller, estimate = state()
    monitor = RouteProgressMonitor(1)
    estimate.pos[0, 0] = 9.5
    for step in range(121):
        monitor.update(step*.1, estimate, [0], [False], controller)
    assert not monitor.events
    estimate.pos[0, 0] = 0.
    for step in range(121, 241):
        monitor.update(step*.1, estimate, [0], [True], controller)
    assert not monitor.events


def test_busy_plan_and_small_moving_obstacles_do_not_trigger_large_recovery():
    controller, estimate = state()
    monitor = RouteProgressMonitor(1)
    for step in range(121):
        monitor.update(step*.1, estimate, [0], [False], controller, [True])
    assert not monitor.events
    estimate.obstacles[0][0] = (np.array([1., .2, 0.]), np.zeros(3), .1)
    for step in range(121, 241):
        monitor.update(step*.1, estimate, [0], [False], controller)
    assert not monitor.events


def test_all_obstacle_navigator_is_available_as_a_separate_ablation():
    assert callable(AllObstacleNavigator.act_all)
    assert callable(AllObstacleNavigator.act_all_reference_guard)


def test_clearance_gate_is_part_of_semantic_ablation_configuration():
    assert ProgressConfig(semantic_min_clearance_mm=.3).semantic_min_clearance_mm == .3
