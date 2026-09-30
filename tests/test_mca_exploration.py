import copy
import math

import numpy as np
import pytest
import torch

from marl.mappo_advanced import MAPPOAdvanced
from marl.mca_exploration import apply_exploration_schedule, exploration_cap


SCHEDULE = dict(kind='log_std_cap_v1', start_std=.7, end_std=.1, anneal_transitions=262144)


def test_logarithmic_schedule_endpoints_and_midpoint():
    assert exploration_cap(None, 0) is None
    assert exploration_cap(SCHEDULE, 0) == pytest.approx(.7)
    assert exploration_cap(SCHEDULE, 131072) == pytest.approx(math.sqrt(.07))
    assert exploration_cap(SCHEDULE, 262144) == pytest.approx(.1)
    assert exploration_cap(SCHEDULE, 500000) == pytest.approx(.1)


@pytest.mark.parametrize('change', [dict(start_std=float('nan')), dict(end_std=0),
    dict(end_std=1), dict(start_std=True), dict(anneal_transitions=0),
    dict(anneal_transitions=2.5), dict(anneal_transitions=True), dict(kind='unknown'), dict(extra=1)])
def test_invalid_schedule_fails_loudly(change):
    with pytest.raises(ValueError):
        exploration_cap(dict(SCHEDULE, **change), 0)


def test_projection_preserves_mean_rng_and_ppo_likelihood_then_survives_checkpoint(tmp_path):
    torch.manual_seed(31)
    agent = MAPPOAdvanced(n_agents=2, obs_dim=36, action_dim=3, state_dim=8,
        hidden_dim=16, num_layers=1, architecture='mlp', control_mode='local', device='cpu')
    original = copy.deepcopy(agent.actor.state_dict())
    rng = torch.get_rng_state().clone()
    assert apply_exploration_schedule(agent, None, 0) is None
    for key, value in original.items():
        torch.testing.assert_close(value, agent.actor.state_dict()[key], rtol=0, atol=0)
    apply_exploration_schedule(agent, SCHEDULE, 262144)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for key, value in original.items():
        if 'log_std' not in key:
            torch.testing.assert_close(value, agent.actor.state_dict()[key], rtol=0, atol=0)
    assert agent.actor.policy_head.log_std.exp().max().item() == pytest.approx(.1)
    obs = np.ones((2, 36), np.float32)
    ctx = dict(positions=np.zeros((2, 3), np.float32), velocities=np.zeros((2, 3), np.float32),
               adjacency=np.ones((2, 2), np.float32), agent_mask=np.ones(2, bool))
    action, lp, value = agent.act(obs, ctx, np.zeros(8, np.float32))
    recomputed, _ = agent.actor.evaluate_actions(torch.tensor(obs)[None], torch.tensor(action)[None],
                                                agent._ctx_to_torch(ctx, batched=False))
    np.testing.assert_allclose(lp, recomputed.detach().numpy()[0], rtol=1e-5, atol=1e-5)
    path = tmp_path/'checkpoint.pt';agent.save(path)
    fresh = copy.deepcopy(agent)
    with torch.no_grad():fresh.actor.policy_head.log_std.fill_(0)
    fresh.load(path)
    torch.testing.assert_close(agent.actor.policy_head.log_std, fresh.actor.policy_head.log_std)
    agent.buffer.store(obs, action, np.zeros(2), np.zeros(2), lp, value, ctx, np.zeros(8))
    with pytest.raises(RuntimeError, match='outstanding PPO rollout'):
        apply_exploration_schedule(agent, SCHEDULE, 262144)
