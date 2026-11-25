from types import SimpleNamespace

import numpy as np

from marl.exit_mask import apply_exit_mask, compensate_exit_command, exit_mask
from tests.test_mca_physical_dynamics import setup_transport


def terminal_pipe():
    transport, solution = setup_transport(length=2.)
    # Provide only the surveyed map, nominal actuator and image estimate.
    # This episode deliberately has no true position, edge or flow solution.
    env = SimpleNamespace(transport=transport, config=SimpleNamespace(
        control_dt_s=.1, robot_speed_mm_s=3.))
    # Positive local Y maps to negative world X.
    frames = np.array([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])
    ctl = SimpleNamespace(body=.08, frames=lambda _: frames)
    ep = SimpleNamespace(env=env, sensor=SimpleNamespace(healthy=solution['radius_mm']), ctl=ctl)
    pos = np.array([[1.95, 0., 0.]])
    est = SimpleNamespace(pos=pos, edge=transport.nearest_edges(pos), active=np.ones(1, bool))
    return ep, est


def test_exit_prediction_uses_image_map_and_local_frame_with_measured_drift():
    ep, est = terminal_pipe()
    commands = np.array([[[0., 0., 0.], [0., 1., 0.], [0., -1., 0.]]])
    drift = np.array([[2., 0., 0.]])
    mask, count = apply_exit_mask(ep, est, commands, np.ones((1, 3)), margin=1., drift=drift)
    np.testing.assert_array_equal(mask, [[0., 1., 0.]])
    assert count == 2
    assert not exit_mask(ep, est, commands, mask, margin=1.)[0, 0]


def test_continuous_command_cancels_passive_flow_before_exit():
    ep, est = terminal_pipe()
    drift = np.array([[2., 0., 0.]])
    corrected = compensate_exit_command(ep, est, np.zeros((1, 3)), drift=drift)
    np.testing.assert_allclose(corrected, [[0., 2./3., 0.]])
    assert not exit_mask(ep, est, corrected[:, None], np.ones((1, 1)), margin=1., drift=drift).any()


def test_unavoidable_crossing_retains_latest_available_command_and_counts_actual_masks():
    ep, est = terminal_pipe()
    commands = np.array([[[0., 0., 0.], [0., .5, 0.], [0., 1., 0.]]])
    drift = np.array([[5., 0., 0.]])
    mask, count = apply_exit_mask(ep, est, commands, np.array([[1., 1., 0.]]), margin=1., drift=drift)
    np.testing.assert_array_equal(mask, [[0., 1., 0.]])
    assert count == 1
    np.testing.assert_allclose(compensate_exit_command(ep, est, commands[:, 0], drift=drift), [[0., 1., 0.]])
