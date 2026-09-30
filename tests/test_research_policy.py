"""Policy-only regressions avoid mixing native torch and simulator loops."""
import numpy as np
import pytest
import torch
from marl.mappo_advanced import MAPPOAdvanced
from marl.policy_loader import load_policy


@pytest.mark.parametrize('architecture', ['gat', 'edge_bias_gat', 'adaptive_edge_gat', 'transformer', 'mlp'])
def test_v_critic_independent_and_first_ratio(architecture):
    agent = MAPPOAdvanced(3, 36, 3, hidden_dim=16, num_layers=1,
                          architecture=architecture, device='cpu')
    obs = torch.randn(2, 3, 36)
    ctx = {'adjacency': torch.ones(2, 3, 3), 'positions': torch.rand(2, 3, 3),
           'velocities': torch.zeros(2, 3, 3)}
    actions, lp, _ = agent.actor.get_actions(obs, ctx)
    lp2, _ = agent.actor.evaluate_actions(obs, actions, ctx)
    torch.testing.assert_close(torch.exp(lp2-lp), torch.ones_like(lp), atol=2e-5, rtol=2e-5)
    a = agent.critic(obs, actions, ctx)
    b = agent.critic(obs, torch.randn_like(actions), ctx)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not any('action_encoder' in k for k in agent.critic.state_dict())


def test_legacy_load_and_optimizer_mode_guard(tmp_path):
    legacy = MAPPOAdvanced(3, 36, 3, hidden_dim=16, num_layers=1,
                           critic_value_mode='q', dropout=.1, device='cpu')
    del legacy.meta['critic_value_mode']
    del legacy.meta['dropout']
    path = tmp_path/'legacy.pt'
    legacy.save(path)
    modern = MAPPOAdvanced(3, 36, 3, hidden_dim=16, num_layers=1, device='cpu')
    with pytest.raises(ValueError, match='critic_value_mode'):
        modern.load(path)
    compatible = MAPPOAdvanced(3, 36, 3, hidden_dim=16, num_layers=1,
                               critic_value_mode='q', dropout=.1, device='cpu')
    compatible.load(path)
    from types import SimpleNamespace
    env = SimpleNamespace(observation_space={'nodes': SimpleNamespace(shape=(3,36))})
    assert callable(load_policy(str(path), env))


def test_eval_restores_modes_on_failure(monkeypatch):
    from types import SimpleNamespace
    from scripts import train_vector_mappo as trainer
    agent = MAPPOAdvanced(3, 36, 3, hidden_dim=16, num_layers=1, device='cpu')
    agent.critic.eval()
    def fail(_):
        assert not agent.actor.training and not agent.critic.training
        raise RuntimeError('sentinel')
    monkeypatch.setattr(trainer, 'resolve_pool', fail)
    with pytest.raises(RuntimeError, match='sentinel'):
        trainer.evaluate_territories(agent, SimpleNamespace(scenario_pool='legacy'), 1)
    assert agent.actor.training and not agent.critic.training


def test_eval_episode_wall_totals(monkeypatch):
    from types import SimpleNamespace
    from scripts import train_vector_mappo as trainer
    class Env:
        def __init__(self, **kwargs): self.steps = 0
        def reset(self, **kwargs):
            self.steps = 0
            return {'nodes': np.zeros((3,36)), 'clot_state': np.zeros((1,6))}, {}
        def step(self, action):
            self.steps += 1
            return {'nodes': np.zeros((3,36)), 'clot_state': np.zeros((1,6))}, 0, self.steps == 3, False, {'success':True, 'removal_rate':1, 'wall_collisions':self.steps}
        def close(self): pass
    agent = SimpleNamespace(actor=torch.nn.Linear(1,1), critic=torch.nn.Linear(1,1),
                            act=lambda *a, **k:(None,None,None), env_action=lambda *a:None)
    monkeypatch.setattr(trainer, 'Vascular3DMARLEnv', Env)
    monkeypatch.setattr(trainer, 'resolve_pool', lambda _:['dummy'])
    monkeypatch.setattr(trainer, 'build_context', lambda *a:{})
    args = SimpleNamespace(scenario_pool='legacy', robots=3, clots=1, horizon=3,
                          robot_radius=.0011, obs_mode='geometric', reward_mode='milestone',
                          contact_mode='geodesic', coverage_bonus=.2, step_cost=0., approach_scale=.1,
                          reward_double_count='on', no_control_margin=False, eval_seed=4)
    metrics = trainer.evaluate_territories(agent, args, 1)['macro']
    assert metrics['wall_hits_total'] == 6
    assert metrics['wall_hits_per_step'] == 2
