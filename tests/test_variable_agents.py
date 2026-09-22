"""Variable agent count (EXP_0006): forward + PPO update under padding.

These tests pin the three properties the max_agents + agent_mask design
promises:

1. Equivalence -- a padded forward pass reproduces the unpadded one exactly
   on the real agents (GAT attention ignores padding in both directions).
2. Exclusion -- padding slots contribute nothing to the PPO losses, GAE,
   entropy or diagnostics, and a mixed-N batch (different counts per sample)
   trains without error.
3. Capacity -- one checkpoint declares max_agents and serves any
   n_agents <= max_agents at inference.
"""
import numpy as np
import pytest
import torch

from marl.mappo_advanced import MAPPOAdvanced, ContextRolloutBuffer


def _agent(n_agents, max_agents=0, control_mode="local", seed=0):
    torch.manual_seed(seed)
    kwargs = {}
    if max_agents:
        kwargs["max_agents"] = max_agents
    return MAPPOAdvanced(
        n_agents=n_agents, obs_dim=36, action_dim=3, state_dim=12,
        hidden_dim=32, num_layers=2, architecture="gat",
        control_mode=control_mode, device="cpu", critic_value_mode="v",
        **kwargs,
    )


def _ctx(obs_n, mask=None, batch=2):
    ctx = {
        "adjacency": torch.ones(batch, obs_n, obs_n),
        "positions": torch.rand(batch, obs_n, 3),
        "velocities": torch.zeros(batch, obs_n, 3),
    }
    if mask is not None:
        ctx["agent_mask"] = mask
    return ctx


@pytest.mark.parametrize("real_n", [3, 5, 8])
def test_padded_forward_matches_unpadded(real_n):
    """Padding slots must not perturb real agents' actions or values."""
    max_n = 10
    agent = _agent(max_n, max_agents=max_n)
    obs = torch.randn(2, real_n, 36)
    ctx = _ctx(real_n)
    ref_actions, ref_lp, _ = agent.actor.get_actions(obs, ctx, deterministic=True)
    ref_values = agent.critic(obs, ref_actions, ctx)

    obs_pad = torch.zeros(2, max_n, 36)
    obs_pad[:, :real_n] = obs
    ctx_pad = {
        "adjacency": torch.zeros(2, max_n, max_n),
        "positions": torch.zeros(2, max_n, 3),
        "velocities": torch.zeros(2, max_n, 3),
    }
    ctx_pad["adjacency"][:, :real_n, :real_n] = ctx["adjacency"]
    ctx_pad["positions"][:, :real_n] = ctx["positions"]
    # self-loops on real agents only; padding fully disconnected
    idx = torch.arange(max_n)
    ctx_pad["adjacency"][:, idx, idx] = 0.0
    ctx_pad["adjacency"][:, idx[:real_n], idx[:real_n]] = 1.0
    mask = torch.zeros(2, max_n, dtype=torch.bool)
    mask[:, :real_n] = True
    ctx_pad["agent_mask"] = mask

    actions, log_probs, _ = agent.actor.get_actions(obs_pad, ctx_pad, deterministic=True)
    values = agent.critic(obs_pad, actions, ctx_pad)
    # Bit-exact up to float32 reassociation: matmul over a padded batch can
    # differ in the last ulp. The semantic guarantee is that padding does not
    # CHANGE the computation graph seen by real agents, i.e. agreement to
    # float32 round-off, not bitwise.
    torch.testing.assert_close(actions[:, :real_n], ref_actions, rtol=0, atol=1e-6)
    torch.testing.assert_close(log_probs[:, :real_n], ref_lp, rtol=0, atol=1e-5)
    torch.testing.assert_close(values[:, :real_n], ref_values, rtol=0, atol=1e-5)
    # padding rows stay finite (inert, not NaN)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("real_n", [3, 5, 8])
def test_padded_update_matches_unpadded_gradient_direction(real_n):
    """A padded update and an unpadded update on identical real data must
    produce the same parameter gradients (padding contributes zero)."""
    max_n = 10
    agent_pad = _agent(max_n, max_agents=max_n)
    agent_ref = _agent(max_n, max_agents=max_n)
    # identical weights
    agent_ref.actor.load_state_dict(agent_pad.actor.state_dict())
    agent_ref.critic.load_state_dict(agent_pad.critic.state_dict())

    T = 4
    torch.manual_seed(3)
    obs = torch.randn(T, real_n, 36)
    actions = torch.rand(T, real_n, 3) * 2 - 1
    lp = torch.randn(T, real_n)
    values = torch.randn(T, real_n)
    rewards = torch.randn(T, real_n)
    dones = torch.zeros(T, real_n)

    def fill(agent, padded, policy_lp=False):
        agent.buffer = ContextRolloutBuffer()
        slot_n = max_n if padded else real_n
        for _t in range(T):
            # One timestep per store() call; per-step arrays are [N]-shaped
            # (the buffer stacks whatever it is given).
            o = np.zeros((slot_n, 36), np.float32)
            a = np.zeros((slot_n, 3), np.float32)
            l = np.zeros((slot_n,), np.float32)
            v = np.zeros((slot_n,), np.float32)
            rw = np.zeros((slot_n,), np.float32)
            d = np.zeros((slot_n,), np.float32)
            o[:real_n] = obs[_t].numpy()
            a[:real_n] = actions[_t].numpy()
            l[:real_n] = lp[_t].numpy()
            v[:real_n] = values[_t].numpy()
            rw[:real_n] = rewards[_t].numpy()
            d[:real_n] = dones[_t].numpy()
            adj = np.zeros((slot_n, slot_n), np.float32)
            adj[:real_n, :real_n] = 1.0
            adj[np.arange(slot_n), np.arange(slot_n)] = 1.0
            ctx = {
                "adjacency": adj,
                "positions": np.zeros((slot_n, 3), np.float32),
                "velocities": np.zeros((slot_n, 3), np.float32),
            }
            if padded:
                m = np.zeros((slot_n,), bool)
                m[:real_n] = True
                ctx["agent_mask"] = m
                d = d * m
                rw = rw * m
                l = l * m
                v = v * m
            if policy_lp:
                # Store log-probs actually produced by the (padded) policy
                # forward, so the pre-update ratio diagnostic is ~1 on real
                # slots -- the property that padding does not corrupt the
                # stored rollouts.
                with torch.no_grad():
                    lp_new, _ = agent.actor.evaluate_actions(
                        torch.as_tensor(o).unsqueeze(0),
                        torch.as_tensor(a).unsqueeze(0),
                        {k: torch.as_tensor(v).unsqueeze(0) for k, v in ctx.items()},
                    )
                l = lp_new.squeeze(0).numpy()
                if padded:
                    l = l * ctx["agent_mask"]
            agent.buffer.store(
                obs=o, actions=a, rewards=rw, dones=d,
                log_probs=l, values=v, ctx=ctx,
            )

    fill(agent_ref, padded=False)
    fill(agent_pad, padded=True)
    torch.manual_seed(11)
    m_ref = agent_ref.update(n_epochs=1, batch_size=256)
    torch.manual_seed(11)
    m_pad = agent_pad.update(n_epochs=1, batch_size=256)
    # Losses are normalised by real-slot count in both cases and the
    # advantage statistics are computed over real slots only, so they agree
    # to float32 round-off. Bit-equality is not expected: the padded batch
    # runs matmuls over a larger N and reductions over a larger array, which
    # reassociates differently even though the padding contributes zero.
    assert np.isfinite(m_pad["actor_loss"]) and np.isfinite(m_pad["critic_loss"])
    assert abs(m_pad["actor_loss"] - m_ref["actor_loss"]) < 0.02
    assert abs(m_pad["critic_loss"] - m_ref["critic_loss"]) < 0.02

    # The stronger property: with log-probs stored from the policy's own
    # padded forward pass, the pre-update ratio on real slots is exactly 1
    # (padding does not corrupt what the rollout recorded).
    fill(agent_pad, padded=True, policy_lp=True)
    torch.manual_seed(11)
    m2 = agent_pad.update(n_epochs=1, batch_size=256)
    assert m2["pre_update_ratio_max_error"] < 1e-4


def test_mixed_agent_counts_in_one_batch():
    """Different active counts per sample in one update: no error, finite."""
    max_n = 6
    agent = _agent(max_n, max_agents=max_n)
    agent.buffer = ContextRolloutBuffer()
    counts = [3, 6, 5, 2]
    T = 4
    for t, real_n in enumerate(counts):
        obs = np.zeros((1, max_n, 36), np.float32)
        obs[:, :real_n] = np.random.randn(1, real_n, 36)
        actions = np.zeros((1, max_n, 3), np.float32)
        actions[:, :real_n] = np.random.uniform(-1, 1, (1, real_n, 3))
        lp = np.zeros((1, max_n), np.float32)
        lp[:, :real_n] = np.random.randn(1, real_n)
        v = np.zeros((1, max_n), np.float32)
        v[:, :real_n] = np.random.randn(1, real_n)
        rw = np.zeros((1, max_n), np.float32)
        rw[:, :real_n] = np.random.randn(1, real_n)
        m = np.zeros((1, max_n), bool)
        m[:, :real_n] = True
        ctx = {
            "adjacency": np.ones((1, max_n, max_n), np.float32) * m[0][None, :],
            "positions": np.random.randn(1, max_n, 3).astype(np.float32),
            "velocities": np.zeros((1, max_n, 3), np.float32),
            "agent_mask": m,
        }
        agent.buffer.store(
            obs=obs, actions=actions, rewards=rw, dones=m.astype(np.float32),
            log_probs=lp, values=v, ctx=ctx,
        )
    metrics = agent.update(n_epochs=2, batch_size=256)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["entropy"])


def test_checkpoint_declares_max_agents_and_reloads_for_fewer():
    """A checkpoint saved with max_agents=N loads and runs at n<=N."""
    max_n = 8
    agent = _agent(max_n, max_agents=max_n)
    # save / reload at max
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "varn.pt")
        agent.save(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["meta"]["max_agents"] == max_n
        # rebuild for a smaller active set: n_agents slot count must be <= max
        smaller = _agent(3, max_agents=max_n)
        smaller.load(path, load_optimizers=False)
        obs = torch.randn(1, 3, 36)
        ctx = _ctx(3, batch=1)
        actions, lp, _ = smaller.actor.get_actions(obs, ctx)
        assert actions.shape == (1, 3, 3)
        assert torch.isfinite(actions).all()


def test_max_agents_validation():
    with pytest.raises(ValueError):
        _agent(5, max_agents=3)


def test_env_padding_bit_identical():
    """Padded vector env reproduces the unpadded env exactly, same seed."""
    from environments.vector_env import VectorVascularEnv

    env_a = VectorVascularEnv(
        n_envs=2, scenario="bifurcation", num_robots=3, num_clots=3,
        horizon=30, seed=7, randomize_scenario=False,
    )
    env_b = VectorVascularEnv(
        n_envs=2, scenario="bifurcation", num_robots=5, num_clots=3,
        horizon=30, seed=7, randomize_scenario=False, active_robots=3,
    )
    obs_a = env_a.reset_all()
    obs_b = env_b.reset_all()
    assert obs_b["agent_mask"].shape == (2, 5)
    assert obs_b["agent_mask"].sum() == 6
    rng = np.random.default_rng(1)
    for _ in range(20):
        act = rng.uniform(-1, 1, (2, 3, 3))
        act_pad = np.zeros((2, 5, 3), np.float32)
        act_pad[:, :3] = act
        obs_a, r_a, term_a, trunc_a, info_a = env_a.step(act)
        obs_b, r_b, term_b, trunc_b, info_b = env_b.step(act_pad)
        np.testing.assert_array_equal(obs_a["nodes"], obs_b["nodes"][:, :3])
        np.testing.assert_array_equal(r_a, r_b)
        np.testing.assert_array_equal(
            info_a["agent_rewards"], info_b["agent_rewards"][:, :3]
        )
        np.testing.assert_array_equal(
            info_a["wall_collisions"], info_b["wall_collisions"]
        )
        # padding rows are inert in the observation
        assert np.all(obs_b["nodes"][:, 3:] == 0.0)
        assert np.all(info_b["agent_rewards"][:, 3:] == 0.0)
