import torch

from marl.td3_local import TD3Actor, TD3Critic


def test_td3_shapes_and_zero_initialization():
    obs = (torch.zeros(3, 24, 11), torch.zeros(3, 24, dtype=torch.bool),
           torch.zeros(3, 48), torch.zeros(3, 4, 32, 32))
    actor = TD3Actor()
    critic = TD3Critic()
    action = actor(*obs)
    q1, q2 = critic(*obs, action)
    assert action.shape == (3, 3)
    assert q1.shape == q2.shape == (3,)
    assert torch.allclose(action, torch.zeros_like(action))
