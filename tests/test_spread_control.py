import numpy as np

from marl.geometric_control import CONTROL_MODES, policy_action


def test_spread_mode_is_registered_and_bounded():
    assert "flow_spread" in CONTROL_MODES
    nodes = np.zeros((2, 36), np.float32)
    nodes[:, 6] = 0.1
    nodes[:, 19] = 1.0
    nodes[:, 20] = 1.0
    nodes[:, 24] = 1.0
    nodes[:, 29] = 0.5
    nodes[:, 35] = 0.1
    obs = {"nodes": nodes}

    class Tree:
        total_length = 1.0
        tangents = np.asarray([[1, 0, 0], [1, 0, 0]], np.float32)
        normals = np.asarray([[0, 1, 0], [0, 1, 0]], np.float32)
        binormals = np.asarray([[0, 0, 1], [0, 0, 1]], np.float32)

    class Env:
        num_robots = 2
        active_clots = 1
        clot_masses = np.asarray([1.0], np.float32)
        clot_positions = np.asarray([[0.2, 0.2, 0.2]], np.float32)
        robot_positions = np.asarray([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], np.float32)
        robot_stations = np.asarray([0, 1], np.int32)
        tree = Tree()

        def _route(self, clot):
            return np.asarray([0.1, 0.2], np.float32), None

    action = policy_action(np.zeros((2, 3), np.float32), obs, Env(), mode="flow_spread")
    assert np.isfinite(action).all()
    assert np.all(np.linalg.norm(action, axis=-1) <= 1.000001)
