import numpy as np

from marl.sac_local_envelope import EnvelopedSpeedSAC


def controller(references, distances):
    ctl = EnvelopedSpeedSAC.__new__(EnvelopedSpeedSAC)
    ctl.references = np.asarray(references, float)
    ctl.last_obs = [{'ctx': np.r_[np.zeros(41), d / 2., np.zeros(6)]} for d in distances]
    ctl.cfg = dict(transit_logit_gain=.08, contact_logit_gain=4.,
                   envelope_outer_mm=.35, envelope_inner_mm=.12, speed_logit_gain=4.)
    return ctl


def test_upstream_transit_retains_thrust_while_contact_can_brake():
    ctl = controller([[.6, .8, 0.], [.6, .8, 0.]], [2., .1])
    command = ctl.decode([[-1.], [-1.]])
    assert np.linalg.norm(command[0]) > .99
    assert np.linalg.norm(command[1]) < .35
    np.testing.assert_allclose(np.cross(command, ctl.references), 0., atol=1e-12)


def test_envelope_preserves_zero_actor_and_intentional_stationary_command():
    ctl = controller([[.3, .4, 0.], [0., 0., 0.]], [.2, .1])
    np.testing.assert_array_equal(ctl.decode([[0.], [0.]]), ctl.references)
    np.testing.assert_array_equal(ctl.decode([[1.], [1.]])[1], [0., 0., 0.])


def test_envelope_is_continuous_and_bounded_at_contact_transition():
    distances = np.linspace(.1, .4, 301)
    ctl = controller(np.tile([.3, .4, 0.], (len(distances), 1)), distances)
    speeds = np.linalg.norm(ctl.decode(np.full((len(distances), 1), -1.)), axis=1)
    assert (speeds >= 0.).all() and (speeds <= 1.).all()
    assert (np.diff(speeds) >= -1e-12).all()
    assert np.abs(np.diff(speeds)).max() < .005
