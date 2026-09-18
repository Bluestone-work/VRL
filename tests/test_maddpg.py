"""Tests for the MADDPG implementation.

The batched critic paths are optimisations of an explicit per-agent formulation,
so they are tested by equivalence against that formulation rather than by
asserting hand-computed numbers. `test_policy_gradient_matches_explicit_maddpg`
is the important one: it pins the O(N) actor objective to the textbook O(N^2)
construction, gradient by gradient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.maddpg_policy import (  # noqa: E402
    MADDPG,
    Actor,
    Critic,
    OUNoise,
    ReplayBuffer,
)


def _fill(buf: ReplayBuffer, n_agents: int, obs_dim: int, count: int,
          state_dim: int = 0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        buf.store(
            obs=rng.standard_normal((n_agents, obs_dim)).astype(np.float32),
            actions=rng.standard_normal((n_agents, 3)).astype(np.float32),
            rewards=rng.standard_normal(n_agents).astype(np.float32),
            next_obs=rng.standard_normal((n_agents, obs_dim)).astype(np.float32),
            done=False,
            state=None if state_dim == 0 else rng.standard_normal(state_dim).astype(np.float32),
            next_state=None if state_dim == 0 else rng.standard_normal(state_dim).astype(np.float32),
        )


# --------------------------------------------------------------------- critic


def test_forward_all_matches_per_agent_forward() -> None:
    """The batched Q must equal N separate single-agent evaluations."""
    torch.manual_seed(0)
    critic = Critic(4, 36, 3, 64, state_dim=24)
    obs = torch.randn(8, 4, 36)
    act = torch.randn(8, 4, 3)
    state = torch.randn(8, 24)

    batched = critic.forward_all(obs, act, state=state)
    assert batched.shape == (8, 4)
    for i in range(4):
        single = critic(obs, act, agent_index=i, state=state).squeeze(1)
        assert torch.allclose(batched[:, i], single, atol=1e-5)


def test_critic_is_permutation_invariant_in_peers() -> None:
    """Swapping two peers must not change Q for the evaluated agent.

    The original concatenating critic had to learn this symmetry from data; here
    it holds by construction, which is also what lets a policy transfer to a
    different swarm size.
    """
    torch.manual_seed(1)
    critic = Critic(4, 12, 3, 32)
    obs = torch.randn(5, 4, 12)
    act = torch.randn(5, 4, 3)

    q_before = critic(obs, act, agent_index=0)
    perm = [0, 3, 2, 1]  # permute peers, leave agent 0 in place
    q_after = critic(obs[:, perm], act[:, perm], agent_index=0)
    assert torch.allclose(q_before, q_after, atol=1e-6)


def test_critic_depends_on_the_evaluated_agent() -> None:
    """Q_i must NOT be symmetric in which agent is being evaluated."""
    torch.manual_seed(2)
    critic = Critic(3, 12, 3, 32)
    obs = torch.randn(6, 3, 12)
    act = torch.randn(6, 3, 3)
    q0 = critic(obs, act, agent_index=0)
    q1 = critic(obs, act, agent_index=1)
    assert not torch.allclose(q0, q1, atol=1e-4)


def test_critic_accepts_a_different_agent_count() -> None:
    """Pooling makes the critic width-independent, so one trained at N=3 can be
    evaluated at N=6 without a shape error."""
    torch.manual_seed(3)
    critic = Critic(3, 12, 3, 32)
    for n in (2, 3, 7):
        obs = torch.randn(4, n, 12)
        act = torch.randn(4, n, 3)
        q = critic.forward_all(obs, act)
        assert q.shape == (4, n)
        assert torch.isfinite(q).all()


def test_critic_requires_state_when_configured() -> None:
    critic = Critic(2, 8, 3, 16, state_dim=5)
    obs, act = torch.randn(2, 2, 8), torch.randn(2, 2, 3)
    with pytest.raises(ValueError):
        critic.forward_all(obs, act, state=None)


def test_policy_gradient_matches_explicit_maddpg() -> None:
    """The O(N) actor objective must reproduce the explicit O(N^2) MADDPG one.

    Reference: for each agent i, build the joint action where only agent i's entry
    carries gradient (peers detached), evaluate Q_i, average over i. The optimised
    path exploits the pooling structure to avoid materialising N joint tensors;
    this test checks that shortcut is exact, not approximate.
    """
    torch.manual_seed(4)
    b, n, obs_dim, state_dim = 16, 4, 36, 24
    critic = Critic(n, obs_dim, 3, 64, state_dim=state_dim)
    actor = Actor(obs_dim, 3, 64)
    obs = torch.randn(b, n, obs_dim)
    state = torch.randn(b, state_dim)

    fresh = actor(obs.reshape(b * n, -1)).reshape(b, n, -1)
    reference = 0.0
    for i in range(n):
        joint = torch.stack(
            [fresh[:, j] if j == i else fresh[:, j].detach() for j in range(n)],
            dim=1,
        )
        reference = reference + (-critic(obs, joint, agent_index=i, state=state).mean())
    reference = reference / n
    grad_ref = torch.autograd.grad(reference, list(actor.parameters()))

    fresh2 = actor(obs.reshape(b * n, -1)).reshape(b, n, -1)
    q = critic.forward_all_policy_grad(obs, fresh2, fresh2.detach(), state=state)
    optimised = -q.mean()
    grad_new = torch.autograd.grad(optimised, list(actor.parameters()))

    assert float(reference) == pytest.approx(float(optimised), abs=1e-6)
    for a, b_ in zip(grad_ref, grad_new):
        assert torch.allclose(a, b_, atol=1e-7), (a - b_).abs().max()


def test_policy_grad_flows_only_through_own_action() -> None:
    """Agent i's gradient must not flow through its peers' actions."""
    torch.manual_seed(5)
    n = 3
    critic = Critic(n, 8, 3, 32)
    obs = torch.randn(4, n, 8)
    fresh = torch.randn(4, n, 3, requires_grad=True)
    q = critic.forward_all_policy_grad(obs, fresh, fresh.detach())
    # Only agent 0's Q, so only agent 0's action should receive gradient.
    q[:, 0].sum().backward()
    assert fresh.grad is not None
    assert fresh.grad[:, 0].abs().sum() > 0
    assert fresh.grad[:, 1:].abs().sum() == 0


# ---------------------------------------------------------------------- buffer


def test_buffer_returns_per_agent_rewards() -> None:
    """Per-agent rewards are the point: a team scalar makes all critics identical."""
    buf = ReplayBuffer(capacity=100)
    _fill(buf, n_agents=3, obs_dim=12, count=40, state_dim=6)
    batch = buf.sample(16)
    assert batch['rewards'].shape == (16, 3)
    assert batch['state'].shape == (16, 6)
    assert batch['next_state'].shape == (16, 6)
    assert batch['done'].shape == (16, 1)


def test_buffer_omits_state_when_absent() -> None:
    buf = ReplayBuffer(capacity=50)
    _fill(buf, n_agents=2, obs_dim=8, count=20, state_dim=0)
    batch = buf.sample(8)
    assert 'state' not in batch


def test_buffer_respects_capacity() -> None:
    buf = ReplayBuffer(capacity=10)
    _fill(buf, n_agents=2, obs_dim=4, count=40)
    assert len(buf) == 10


# ----------------------------------------------------------------------- noise


def test_ou_noise_is_seeded_and_reproducible() -> None:
    """Noise must be reproducible from a seed, so a run can be replayed."""
    a = OUNoise(3, rng=np.random.default_rng(7))
    b = OUNoise(3, rng=np.random.default_rng(7))
    for _ in range(5):
        assert np.allclose(a.sample(), b.sample())


def test_ou_noise_sample_is_not_an_alias() -> None:
    """sample() must return a copy; handing out the internal state means the
    caller's stored action mutates on the next call."""
    noise = OUNoise(3, rng=np.random.default_rng(1))
    first = noise.sample()
    snapshot = first.copy()
    noise.sample()
    assert np.allclose(first, snapshot), "previous sample was mutated in place"


def test_ou_noise_reset_clears_state() -> None:
    noise = OUNoise(3, rng=np.random.default_rng(2))
    for _ in range(10):
        noise.sample()
    noise.reset()
    assert np.allclose(noise.state, 0.0)


# ------------------------------------------------------------------- training


@pytest.mark.parametrize("share", [True, False])
def test_update_runs_and_changes_parameters(share: bool) -> None:
    m = MADDPG(n_agents=3, obs_dim=20, action_dim=3, hidden_dim=32, batch_size=32,
               device='cpu', state_dim=12, share_parameters=share, seed=0)
    _fill(m.replay_buffer, 3, 20, 200, state_dim=12)
    before = [p.clone() for p in m.actors[0].parameters()]
    info = m.update()
    assert set(info) == {'actor_loss', 'critic_loss', 'mean_q'}
    assert all(np.isfinite(v) for v in info.values())
    after = list(m.actors[0].parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_update_is_a_noop_before_the_buffer_fills() -> None:
    m = MADDPG(n_agents=2, obs_dim=10, action_dim=3, hidden_dim=16, batch_size=64,
               device='cpu', seed=0)
    _fill(m.replay_buffer, 2, 10, 8)
    assert m.update() == {}


def test_shared_parameters_means_one_network() -> None:
    shared = MADDPG(n_agents=5, obs_dim=10, action_dim=3, hidden_dim=16,
                    device='cpu', share_parameters=True, seed=0)
    assert len(shared.actors) == 1
    assert all(shared.actor(i) is shared.actors[0] for i in range(5))

    separate = MADDPG(n_agents=5, obs_dim=10, action_dim=3, hidden_dim=16,
                      device='cpu', share_parameters=False, seed=0)
    assert len(separate.actors) == 5
    assert separate.actor(0) is not separate.actor(1)


def test_select_actions_shape_and_bounds() -> None:
    m = MADDPG(n_agents=4, obs_dim=15, action_dim=3, hidden_dim=16, device='cpu', seed=0)
    obs = np.random.default_rng(0).standard_normal((4, 15)).astype(np.float32)
    a = m.select_actions(obs, add_noise=True, noise_scale=1.0)
    assert a.shape == (4, 3)
    assert a.min() >= -1.0 and a.max() <= 1.0
    d = m.select_actions(obs, add_noise=False)
    assert np.allclose(d, m.select_actions(obs, add_noise=False)), "not deterministic"


def test_noise_scale_zero_is_deterministic() -> None:
    m = MADDPG(n_agents=3, obs_dim=10, action_dim=3, hidden_dim=16, device='cpu', seed=0)
    obs = np.zeros((3, 10), np.float32)
    a = m.select_actions(obs, add_noise=True, noise_scale=0.0)
    b = m.select_actions(obs, add_noise=False)
    assert np.allclose(a, b)


def test_soft_update_moves_targets_toward_online() -> None:
    m = MADDPG(n_agents=2, obs_dim=10, action_dim=3, hidden_dim=16, device='cpu',
               tau=0.5, seed=0)
    with torch.no_grad():
        for p in m.actors[0].parameters():
            p.add_(1.0)
    before = [p.clone() for p in m.target_actors[0].parameters()]
    m._soft_update()
    for b_, t, o in zip(before, m.target_actors[0].parameters(),
                        m.actors[0].parameters()):
        expected = 0.5 * o.data + 0.5 * b_
        assert torch.allclose(t.data, expected, atol=1e-6)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    m = MADDPG(n_agents=3, obs_dim=20, action_dim=3, hidden_dim=32, batch_size=32,
               device='cpu', state_dim=12, seed=0)
    _fill(m.replay_buffer, 3, 20, 100, state_dim=12)
    m.update()
    path = tmp_path / "p.pt"
    m.save(path)

    other = MADDPG(n_agents=3, obs_dim=20, action_dim=3, hidden_dim=32,
                   device='cpu', state_dim=12, seed=1)
    other.load(path)
    obs = np.random.default_rng(0).standard_normal((3, 20)).astype(np.float32)
    assert np.allclose(
        m.select_actions(obs, add_noise=False),
        other.select_actions(obs, add_noise=False),
        atol=1e-6,
    )


def test_load_tolerates_a_mismatched_agent_count(tmp_path: Path) -> None:
    """A shared homogeneous actor is valid at any swarm size."""
    small = MADDPG(n_agents=2, obs_dim=20, action_dim=3, hidden_dim=32,
                   device='cpu', seed=0)
    path = tmp_path / "small.pt"
    small.save(path)
    big = MADDPG(n_agents=9, obs_dim=20, action_dim=3, hidden_dim=32,
                 device='cpu', seed=1)
    big.load(path)  # must not raise
    obs = np.zeros((9, 20), np.float32)
    assert big.select_actions(obs, add_noise=False).shape == (9, 3)


def test_truncation_is_not_stored_as_terminal() -> None:
    """Documents the bootstrapping contract the trainer relies on.

    The trainer stores `done=terminated`, never `terminated or truncated`. A
    time-limit truncation is not a terminal state; recording it as one teaches the
    value function that the world ends at the horizon.
    """
    buf = ReplayBuffer(capacity=10)
    buf.store(
        obs=np.zeros((2, 5), np.float32), actions=np.zeros((2, 3), np.float32),
        rewards=np.zeros(2, np.float32), next_obs=np.zeros((2, 5), np.float32),
        done=False,  # truncated episode
    )
    assert buf.sample(1)['done'].item() == 0.0
