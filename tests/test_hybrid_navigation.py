from types import SimpleNamespace

import numpy as np
import pytest

from marl.hybrid_navigation import HybridConfig, HybridNavigator


def setup(radius=.1, relative=None):
    sensor = SimpleNamespace(a=np.array([[-10., 0., 0.]]), ab=np.array([[20., 0., 0.]]),
                             healthy=np.ones(2), ends=np.array([[0, 1]]), env=SimpleNamespace())
    controller = SimpleNamespace(frames=lambda estimate: np.eye(3)[None], nominal=np.array([[1., 0., 0.]]))
    estimate = SimpleNamespace(pos=np.zeros((1, 3)), edge=np.array([0]), active=np.array([True]),
                               obstacles=[[(np.array([.4, .1, 0.]) if relative is None else relative, np.zeros(3), radius)]])
    return HybridNavigator(sensor, 1, .08), controller, estimate


def test_small_detected_forward_obstacle_uses_committed_pass():
    navigator, controller, estimate = setup()
    local = navigator.act(0., estimate, controller, np.zeros((1, 3)), [False])
    assert local[0, 0] > 0
    assert local[0, 1] < 0
    assert navigator.last_modes == ['small_obstacle_pass']


def test_large_obstacle_preserves_original_reference_exactly():
    navigator, controller, estimate = setup(radius=.4)
    reference = np.array([[.5, .2, .1]])
    np.testing.assert_array_equal(navigator.act(0., estimate, controller, reference, [False]), reference)
    assert navigator.last_modes == ['reference']


def test_behind_obstacle_does_not_trigger_pass():
    navigator, controller, estimate = setup(relative=np.array([-.4, .1, 0.]))
    reference = np.array([[1., 0., 0.]])
    np.testing.assert_array_equal(navigator.act(0., estimate, controller, reference, [False]), reference)


def test_holds_dominate_recovery_and_passing():
    navigator, controller, estimate = setup()
    navigator.trigger(0., [0])
    np.testing.assert_array_equal(navigator.act(0., estimate, controller, np.ones((1, 3)), [True]), np.zeros((1, 3)))


def test_nonpositive_size_gate_rejected():
    with pytest.raises(ValueError):
        HybridConfig(small_obstacle_max_radius_mm=0.)
