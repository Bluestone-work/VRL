from types import SimpleNamespace

import numpy as np
import pytest

from marl.risk_navigation import (ProgressBoundRiskConfig, ProgressBoundRiskNavigator,
                                   RiskConfig, RiskHysteresisNavigator)


def setup():
    sensor = SimpleNamespace(
        a=np.array([[-10., 0., 0.]]), ab=np.array([[20., 0., 0.]]),
        length=np.array([20.]), healthy=np.array([1., 1.]), ends=np.array([[0, 1]]),
        env=SimpleNamespace(),
        map_coordinates=lambda estimate, robot: (
            np.zeros(3), 1., float(np.linalg.norm(estimate.pos[robot, 1:])))
    )
    controller = SimpleNamespace(
        frames=lambda estimate: np.eye(3)[None], nominal=np.array([[1., 0., 0.]]),
        env=SimpleNamespace(elapsed_s=0.))
    estimate = SimpleNamespace(
        pos=np.array([[0., .75, 0.]]), vel=np.array([[0., .3, 0.]]),
        edge=np.array([0]), active=np.array([True]),
        obstacles=[[]])
    return sensor, controller, estimate


def test_risk_config_requires_real_hysteresis():
    with pytest.raises(ValueError):
        RiskConfig(risk_exit_score=.6, risk_enter_score=.5)


def test_risk_enters_only_after_confirmation_and_exits_after_hold():
    sensor, controller, estimate = setup()
    cfg = RiskConfig(risk_filter_alpha=1., risk_enter_confirm_steps=2,
                     risk_exit_confirm_steps=2, risk_min_hold_s=1.,
                     risk_rearm_offset_fraction=.80)
    navigator = RiskHysteresisNavigator(sensor, 1, .08, cfg)
    controller.env.elapsed_s = 0.
    risk = navigator._score(0, estimate, controller, np.array([0., 1., 0.]))
    navigator._update_mode(0., 0, risk)
    assert not navigator.risk_active[0]
    controller.env.elapsed_s = .1
    risk = navigator._score(0, estimate, controller, np.array([0., 1., 0.]))
    navigator._update_mode(.1, 0, risk)
    assert navigator.risk_active[0]
    estimate.pos[0, 1] = .05
    estimate.vel[0] = 0.
    for time_s in (1.1, 1.2):
        controller.env.elapsed_s = time_s
        risk = navigator._score(0, estimate, controller, np.array([1., 0., 0.]))
        navigator._update_mode(time_s, 0, risk)
    assert not navigator.risk_active[0]


def test_risk_does_not_rearm_while_offset_stays_high():
    sensor, controller, estimate = setup()
    cfg = RiskConfig(risk_filter_alpha=1., risk_enter_confirm_steps=1,
                     risk_exit_confirm_steps=1, risk_min_hold_s=0.1,
                     risk_rearm_offset_fraction=.80)
    navigator = RiskHysteresisNavigator(sensor, 1, .08, cfg)
    controller.env.elapsed_s = 0.
    high = navigator._score(0, estimate, controller, np.array([0., 1., 0.]))
    navigator._update_mode(0., 0, high)
    assert navigator.risk_active[0]
    estimate.pos[0, 1] = .7
    estimate.vel[0] = 0.
    controller.env.elapsed_s = .2
    low = navigator._score(0, estimate, controller, np.array([1., 0., 0.]))
    navigator._update_mode(.2, 0, low)
    assert not navigator.risk_active[0]
    controller.env.elapsed_s = .3
    high = navigator._score(0, estimate, controller, np.array([0., 1., 0.]))
    navigator._update_mode(.3, 0, high)
    assert not navigator.risk_active[0]


def test_risk_guard_removes_outward_component_but_keeps_forward_component():
    sensor, controller, estimate = setup()
    cfg = RiskConfig(risk_filter_alpha=1.)
    navigator = RiskHysteresisNavigator(sensor, 1, .08, cfg)
    controller.env.elapsed_s = 0.
    risk = navigator._score(0, estimate, controller, np.array([.5, .8, 0.]))
    guarded = navigator._guard(0, estimate, controller, np.array([.5, .8, 0.]), risk)
    assert guarded[0] > 0.
    assert guarded[1] < .8


def test_forward_obstacle_keeps_semantic_command_priority():
    sensor, controller, estimate = setup()
    estimate.obstacles = [[(np.array([.3, 0., 0.]), np.zeros(3), .2)]]
    navigator = RiskHysteresisNavigator(sensor, 1, .08, RiskConfig())
    reference = np.array([[1., 0., 0.]])
    expected = navigator.executor.act(0., estimate, controller,
                                     navigator.semantic.act(0., estimate, controller), [False])
    local = navigator.act_risk(0., estimate, controller, reference, [False])
    np.testing.assert_allclose(local, expected, atol=1e-8)
    assert navigator.last_modes[0].startswith('semantic_guard')


def test_progress_bound_config_and_state_require_stall_for_entry():
    sensor, controller, estimate = setup()
    controller.route = [np.array([0, 1])]
    controller.prog = np.array([0])
    controller.pts = np.array([[0., 0., 0.], [10., 0., 0.]])
    controller.env.clot_positions_mm = np.array([[10., 0., 0.]])
    controller.body = .08
    cfg = ProgressBoundRiskConfig(risk_filter_alpha=1., risk_enter_confirm_steps=1,
                                  progress_stall_s=2., progress_commit_s=.5)
    navigator = ProgressBoundRiskNavigator(sensor, 1, .08, cfg)
    controller.env.elapsed_s = 0.
    progress = navigator._progress_state(0., 0, estimate, [0], controller, [False])
    assert not progress['stalled']
    assert not navigator.risk_active[0]
    controller.env.elapsed_s = 2.1
    progress = navigator._progress_state(2.1, 0, estimate, [0], controller, [False])
    assert progress['stalled']


def test_progress_bound_risk_releases_after_commitment_timeout():
    sensor, controller, estimate = setup()
    controller.route = [np.array([0, 1])]
    controller.prog = np.array([0])
    controller.pts = np.array([[0., 0., 0.], [10., 0., 0.]])
    controller.env.clot_positions_mm = np.array([[10., 0., 0.]])
    controller.body = .08
    cfg = ProgressBoundRiskConfig(risk_filter_alpha=1., risk_enter_confirm_steps=1,
                                  progress_stall_s=.1, progress_commit_s=.2,
                                  risk_enter_offset_fraction=.7, risk_rearm_offset_fraction=.6)
    navigator = ProgressBoundRiskNavigator(sensor, 1, .08, cfg)
    controller.env.elapsed_s = 0.
    navigator.act_progress_bound(0., estimate, controller, np.array([[0., 1., 0.]]), [False], [0])
    controller.env.elapsed_s = .2
    navigator.act_progress_bound(.2, estimate, controller, np.array([[0., 1., 0.]]), [False], [0])
    controller.env.elapsed_s = .5
    navigator.act_progress_bound(.5, estimate, controller, np.array([[0., 1., 0.]]), [False], [0])
    assert navigator.last_modes[0] in ('reference', 'semantic_guard')
