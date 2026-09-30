"""Entropy bonus must describe the current bounded policy, not old actions."""
import torch
from marl.mappo_policy import GaussianActor


def head_at(mean):
    head = GaussianActor(1, 3)
    with torch.no_grad():
        head.mean.weight.zero_()
        head.mean.bias.fill_(mean)
        head.log_std.fill_(-1.)
    return head


def test_entropy_gradient_restores_saturated_policy():
    torch.manual_seed(13)
    head = head_at(5.)
    _, entropy = head.evaluate_actions(torch.zeros(4096, 1), torch.zeros(4096, 3))
    (-entropy.mean()).backward()
    # Gradient descent on -H should bring the saturated mean back towards zero.
    assert torch.all(head.mean.bias.grad > 0)
    assert entropy.mean() < 0


def test_entropy_independent_of_rollout_action():
    head = head_at(2.)
    features = torch.zeros(128, 1)
    torch.manual_seed(13)
    _, first = head.evaluate_actions(features, torch.zeros(128, 3))
    torch.manual_seed(13)
    _, second = head.evaluate_actions(features, torch.full((128, 3), .8))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_entropy_extreme_saturation_has_finite_gradient():
    torch.manual_seed(13)
    head = head_at(10000.)
    _, entropy = head.evaluate_actions(torch.zeros(128, 1), torch.zeros(128, 3))
    assert torch.isfinite(entropy).all()
    (-entropy.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
