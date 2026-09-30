"""An all-live mask must not change PPO, including partial minibatches."""
import copy

import numpy as np
import torch

from marl.mappo_advanced import MAPPOAdvanced


def test_all_live_mask_preserves_minibatch_losses_and_gradients():
    torch.manual_seed(20)
    base = MAPPOAdvanced(n_agents=3, obs_dim=36, action_dim=3, state_dim=12,
                         hidden_dim=32, num_layers=1, architecture="mlp",
                         control_mode="local", device="cpu", dropout=0,
                         lr_actor=0, lr_critic=0, max_grad_norm=1e9)
    agents = [copy.deepcopy(base), copy.deepcopy(base)]
    rng = np.random.default_rng(21)
    trajectory = []
    for _ in range(9):
        obs = rng.normal(size=(3, 36)).astype(np.float32)
        ctx = dict(adjacency=np.ones((3, 3), np.float32),
                   positions=rng.normal(size=(3, 3)).astype(np.float32),
                   velocities=np.zeros((3, 3), np.float32))
        state = rng.normal(size=12).astype(np.float32)
        action, lp, value = base.act(obs, ctx, state)
        trajectory.append((obs, ctx, state, action, lp, value,
                           rng.normal(size=3).astype(np.float32)))
    metrics, gradients = [], []
    for masked, agent in enumerate(agents):
        for t, (obs, ctx, state, action, lp, value, reward) in enumerate(trajectory):
            context = {k: v.copy() for k, v in ctx.items()}
            if masked:
                context["agent_mask"] = np.ones(3, bool)
            done = np.full(3, t == len(trajectory) - 1, np.float32)
            agent.buffer.store(obs, action, reward, done, lp, value, context, state,
                               next_obs=obs, next_ctx=context, next_state=state,
                               terminals=done)
        captured = []
        def capture():
            captured.append(torch.cat([p.grad.detach().flatten().clone()
                                       for p in agent.actor.parameters() if p.grad is not None]))
        # Keep parameters fixed while recording the actual backward results.
        agent.actor_optimizer.step = capture
        np.random.seed(22)
        torch.manual_seed(23)
        metrics.append(agent.update(n_epochs=1, batch_size=6))
        gradients.append(captured)
    assert len(gradients[0]) == len(gradients[1]) == 5  # 2+2+2+2+1 time samples
    for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"):
        np.testing.assert_allclose(metrics[0][key], metrics[1][key], rtol=2e-5, atol=1e-6,
                                   err_msg=f"all-live mask changed {key}")
    for reference, masked in zip(*gradients):
        torch.testing.assert_close(reference, masked, rtol=2e-5, atol=1e-6)
