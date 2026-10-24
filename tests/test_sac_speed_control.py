import numpy as np
import torch

from marl.image_flow_sac import ControlEpisode
from marl.sac_speed_control import SpeedActor, SpeedCritic, SpeedOnlySAC


def test_scalar_sac_has_finite_actor_and_critic_gradients():
    torch.manual_seed(97)
    actor, critic = SpeedActor(), SpeedCritic()
    inputs = (
        torch.randn(4, 24, 11),
        torch.zeros(4, 24, dtype=torch.bool),
        torch.randn(4, 48),
        torch.randn(4, 4, 32, 32),
    )
    action, logp = actor.sample(*inputs)
    assert action.shape == (4, 1)
    q1, q2 = critic(*inputs, action)
    loss = (0.002 * logp - torch.minimum(q1, q2)).mean()
    loss.backward()
    for module in (actor, critic):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_speed_action_brakes_full_speed_and_keeps_reference_direction():
    ctl = SpeedOnlySAC.__new__(SpeedOnlySAC)
    ctl.references = np.array([[0.6, 0.8, 0.], [.3, .4, 0.], [0., 0., 0.]])
    ctl.cfg = {'speed_logit_gain': 4.}
    braking = ctl.decode(np.array([[-1.], [-1.], [-1.]]))
    assert np.linalg.norm(braking[0]) < .35
    np.testing.assert_allclose(np.cross(braking[:2], ctl.references[:2]), 0., atol=1e-12)
    for value in (-1., 0., 1.):
        command = ctl.decode(np.full((3, 1), value))
        assert np.isfinite(command).all()
        assert (np.linalg.norm(command, axis=1) <= 1. + 1e-12).all()
    np.testing.assert_array_equal(ctl.decode(np.zeros((3, 1))), ctl.references)


def test_zero_scalar_actor_recovers_original_long_wait_episode():
    ep = ControlEpisode(3, 'ica_siphon', 2400610030, flow_inlet_mm_s=.05)
    try:
        ctl = SpeedOnlySAC(ep, actor=SpeedActor())
        while True:
            est = ep.observe()
            target = ep.plan_targets_now(est)
            rule = ep.ctl.act(target, est)
            hold = ep.hold(est)
            done, _ = ep.step(est, ctl(ep, est, target, rule, hold), hold)
            if done:
                break
        row = ep.row('scalar_initial')
        assert row['strict_success'] and row['removal'] == 1. and row['wall_contact_s'] == 0.
        np.testing.assert_allclose(row['elapsed_s'], 51.1, rtol=0, atol=1e-9)
        np.testing.assert_allclose(row['tpg_hold_s'], 13.9, rtol=0, atol=1e-9)
    finally:
        ep.close()
