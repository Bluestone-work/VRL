"""Unit tests for the method-1 / method-2 cluster protocol.

These tests use a tiny fake physical environment so the protocol state machine
can be checked without invoking the long-horizon numerical integrator.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import marl.multicluster as mc


class _Transport:
    @staticmethod
    def velocity_mm_s(positions, edges, solution):
        return np.zeros_like(positions, dtype=float)


class _FakeEnv:
    def __init__(self, n=2):
        self.num_robots = n
        self.positions_mm = np.zeros((n, 3), dtype=float)
        self.edges = np.zeros(n, dtype=np.int32)
        self.active = np.ones(n, dtype=bool)
        self.solution = object()
        self.transport = _Transport()
        self.config = SimpleNamespace(robot_speed_mm_s=1.0)


def test_balanced_partition_and_config_validation():
    np.testing.assert_array_equal(mc.balanced_cluster_ids(5, 2), [0, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        mc.balanced_cluster_ids(1, 2)
    with pytest.raises(ValueError):
        mc.MultiClusterConfig(clusters=True)
    with pytest.raises(ValueError):
        mc.MultiClusterConfig(wait_horizon_s=-1)


def test_spacing_shield_yields_one_cluster_and_counts_current_violation():
    env = _FakeEnv(2)
    c = object.__new__(mc.MultiClusterController)
    c.env, c.config = env, mc.MultiClusterConfig(clusters=2, min_spacing_mm=2, deadlock_window_steps=10)
    c.cluster_ids = np.array([0, 1], dtype=np.int32)
    c.wait_steps = np.zeros(2, dtype=np.int32)
    c.deadlock_events = c.yield_events = c.spacing_violations = 0
    c.min_spacing_seen_mm = np.inf
    c._right_of_way = 0
    env.positions_mm[:] = [[0, 0, 0], [1, 0, 0]]
    out = c._spacing_shield(np.ones((2, 3)), env.positions_mm.copy())
    assert np.all(out[1] == 0) and np.all(out[0] != 0)
    assert c.yield_events == 1 and c.spacing_violations == 1
    # A second call at the same gap is another violating step, not a replay of
    # the historical minimum beyond the current violation.
    c._spacing_shield(np.ones((2, 3)), env.positions_mm.copy())
    assert c.spacing_violations == 2


def test_deadlock_window_alternates_right_of_way():
    env = _FakeEnv(2)
    c = object.__new__(mc.MultiClusterController)
    c.env, c.config = env, mc.MultiClusterConfig(clusters=2, min_spacing_mm=2, deadlock_window_steps=2)
    c.cluster_ids = np.array([0, 1], dtype=np.int32)
    c.wait_steps = np.zeros(2, dtype=np.int32)
    c.deadlock_events = c.yield_events = c.spacing_violations = 0
    c.min_spacing_seen_mm = np.inf
    c._right_of_way = 0
    centres = np.array([[0., 0, 0], [1., 0, 0]])
    c._spacing_shield(np.ones((2, 3)), centres)
    out = c._spacing_shield(np.ones((2, 3)), centres)
    assert c.deadlock_events == 1
    assert c._right_of_way == 1 and np.all(out[0] == 0)
    assert np.all(out[1] != 0)


def test_parallel_controller_does_not_touch_route_or_assignment(monkeypatch):
    env = _FakeEnv(2)
    env.masses = np.ones(2)
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("privileged accessor"))
    env._route_directions = forbidden
    env._assigned_targets = forbidden
    c = mc.MultiClusterController(env, mc.MultiClusterConfig(clusters=2))
    c.observer = SimpleNamespace(observe=lambda: np.zeros((2, 111)), record_action=lambda action: None)
    monkeypatch.setattr(mc, "execute_local", lambda _env, action: action)
    action, info = c.act()
    assert action.shape == (2, 3)
    assert info["method"] == "multi_parallel"
