"""
Unit tests for GNN-MAPPO implementation.

Tests the core components to ensure they work correctly before training.
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.gat_policy import (
    GATLayer,
    GATEncoder,
    GATActor,
    GATCritic,
    build_adjacency_matrix,
)
from marl.mappo_policy import GaussianActor, MAPPO, RolloutBuffer
from marl.mappo_advanced import MAPPOAdvanced, ContextRolloutBuffer


def test_squashed_gaussian_actions_match_logged_probability():
    head = GaussianActor(feature_dim=8, action_dim=3)
    features = torch.randn(4, 5, 8)
    actions, sampled_log_prob, _ = head.get_actions(features)
    evaluated_log_prob, _ = head.evaluate_actions(features, actions)
    assert torch.all(actions > -1.0) and torch.all(actions < 1.0)
    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=1e-5)


def test_advanced_checkpoint_restores_training_state(tmp_path):
    agent = MAPPOAdvanced(
        n_agents=3, obs_dim=36, action_dim=3, state_dim=12,
        architecture="gat", device="cpu",
    )
    expected = next(agent.actor.parameters()).detach().clone()
    path = tmp_path / "resume.pt"
    agent.save(path, {"total_steps": 123, "episode_count": 7})
    for parameter in agent.actor.parameters():
        parameter.data.zero_()
    restored = agent.load(path)
    assert restored["total_steps"] == 123
    assert torch.allclose(next(agent.actor.parameters()), expected)


def test_context_buffer_keeps_post_transition_state():
    buffer = ContextRolloutBuffer()
    obs = np.zeros((3, 36), np.float32)
    ctx = {
        "adjacency": np.eye(3, dtype=np.float32),
        "positions": np.zeros((3, 3), np.float32),
        "velocities": np.zeros((3, 3), np.float32),
    }
    buffer.store(
        obs, np.zeros((3, 3), np.float32), np.zeros(3, np.float32),
        np.zeros(3, np.float32), np.zeros(3, np.float32),
        np.zeros(3, np.float32), ctx, np.zeros(12, np.float32),
        next_obs=np.ones_like(obs), next_ctx=ctx,
        next_state=np.ones(12, np.float32),
    )
    data = buffer.get()
    assert np.all(data["next_obs"] == 1.0)
    assert np.all(data["next_states"] == 1.0)


def test_advanced_mappo_vector_rollout_update():
    n_envs, n_agents, obs_dim, state_dim = 4, 3, 10, 12
    agent = MAPPOAdvanced(
        n_agents=n_agents, obs_dim=obs_dim, action_dim=3,
        state_dim=state_dim, architecture="gat", hidden_dim=32, device="cpu",
    )
    obs = np.random.randn(n_envs, n_agents, obs_dim).astype(np.float32)
    for _ in range(6):
        ctx = {
            "adjacency": np.ones((n_envs, n_agents, n_agents), np.float32),
            "positions": np.random.randn(n_envs, n_agents, 3).astype(np.float32),
            "velocities": np.random.randn(n_envs, n_agents, 3).astype(np.float32),
        }
        state = np.random.randn(n_envs, state_dim).astype(np.float32)
        actions, log_probs, values = agent.act_batch(obs, ctx, state)
        next_obs = np.random.randn(*obs.shape).astype(np.float32)
        agent.buffer.store(
            obs, actions, np.random.randn(n_envs, n_agents).astype(np.float32),
            np.zeros((n_envs, n_agents), np.float32), log_probs, values,
            ctx, state, next_obs=next_obs, next_ctx=ctx, next_state=state,
        )
        obs = next_obs
    metrics = agent.update(n_epochs=1, batch_size=24)
    assert all(np.isfinite(value) for value in metrics.values())


def test_gat_layer():
    """Test GAT layer forward pass."""
    print("Testing GATLayer...")

    batch_size = 4
    n_agents = 5
    in_dim = 32
    out_dim = 64
    num_heads = 4

    layer = GATLayer(in_dim, out_dim, num_heads)
    x = torch.randn(batch_size, n_agents, in_dim)

    # Test without adjacency matrix
    out = layer(x)
    assert out.shape == (batch_size, n_agents, out_dim), f"Expected {(batch_size, n_agents, out_dim)}, got {out.shape}"

    # Test with adjacency matrix
    adj = torch.ones(batch_size, n_agents, n_agents)
    out = layer(x, adj)
    assert out.shape == (batch_size, n_agents, out_dim)

    print("✓ GATLayer test passed")


def test_gat_encoder():
    """Test GAT encoder."""
    print("Testing GATEncoder...")

    batch_size = 4
    n_agents = 5
    obs_dim = 36
    hidden_dim = 128

    encoder = GATEncoder(obs_dim, hidden_dim, num_layers=2, num_heads=4)
    obs = torch.randn(batch_size, n_agents, obs_dim)

    features = encoder(obs)
    assert features.shape == (batch_size, n_agents, hidden_dim)

    print("✓ GATEncoder test passed")


def test_gat_actor():
    """Test GAT actor."""
    print("Testing GATActor...")

    batch_size = 4
    n_agents = 5
    obs_dim = 36
    action_dim = 3

    actor = GATActor(obs_dim, action_dim)
    obs = torch.randn(batch_size, n_agents, obs_dim)

    # Test forward pass
    actions = actor(obs)
    assert actions.shape == (batch_size, n_agents, action_dim)
    assert (actions >= -1).all() and (actions <= 1).all(), "Actions should be in [-1, 1]"

    # Test single agent
    obs_single = torch.randn(n_agents, obs_dim)
    actions_single = actor(obs_single)
    assert actions_single.shape == (n_agents, action_dim)

    print("✓ GATActor test passed")


def test_gat_critic():
    """Test GAT critic."""
    print("Testing GATCritic...")

    batch_size = 4
    n_agents = 5
    obs_dim = 36
    action_dim = 3
    state_dim = 15

    critic = GATCritic(obs_dim, action_dim, state_dim=state_dim)

    obs = torch.randn(batch_size, n_agents, obs_dim)
    actions = torch.randn(batch_size, n_agents, action_dim)
    state = torch.randn(batch_size, state_dim)

    values = critic(obs, actions, state=state)
    assert values.shape == (batch_size, n_agents, 1)

    print("✓ GATCritic test passed")


def test_adjacency_matrix():
    """Test adjacency matrix construction."""
    print("Testing build_adjacency_matrix...")

    batch_size = 2
    n_agents = 4

    # Create positions where some agents are close
    positions = torch.tensor([
        [[0.0, 0.0, 0.0],   # Agent 0
         [0.05, 0.0, 0.0],  # Agent 1 (close to 0)
         [0.5, 0.0, 0.0],   # Agent 2 (far)
         [0.55, 0.0, 0.0]], # Agent 3 (close to 2)
        [[0.0, 0.0, 0.0],
         [0.0, 0.05, 0.0],
         [0.0, 0.5, 0.0],
         [0.0, 0.55, 0.0]]
    ])

    adj = build_adjacency_matrix(positions, threshold=0.15, include_self=True)

    assert adj.shape == (batch_size, n_agents, n_agents)
    assert (adj >= 0).all() and (adj <= 1).all(), "Adjacency values should be binary"

    # Check self-loops
    for i in range(batch_size):
        assert adj[i].diag().sum() == n_agents, "All agents should have self-loops"

    # Check symmetry
    assert torch.allclose(adj, adj.transpose(1, 2)), "Adjacency matrix should be symmetric"

    # Check specific connections in first batch
    assert adj[0, 0, 1] == 1, "Agent 0 and 1 should be connected (distance 0.05 < 0.15)"
    assert adj[0, 0, 2] == 0, "Agent 0 and 2 should not be connected (distance 0.5 > 0.15)"

    print("✓ build_adjacency_matrix test passed")


def test_rollout_buffer():
    """Test rollout buffer."""
    print("Testing RolloutBuffer...")

    buffer = RolloutBuffer()
    n_agents = 3
    obs_dim = 36
    action_dim = 3

    # Store some transitions
    for t in range(5):
        buffer.store(
            obs=np.random.randn(n_agents, obs_dim),
            actions=np.random.randn(n_agents, action_dim),
            rewards=np.random.randn(n_agents),
            dones=np.zeros(n_agents),
            log_probs=np.random.randn(n_agents),
            values=np.random.randn(n_agents),
        )

    assert len(buffer) == 5

    # Get data
    data = buffer.get()
    assert data['obs'].shape == (5, n_agents, obs_dim)
    assert data['actions'].shape == (5, n_agents, action_dim)
    assert len(buffer) == 0  # Should be cleared after get()

    print("✓ RolloutBuffer test passed")


def test_mappo_initialization():
    """Test MAPPO initialization."""
    print("Testing MAPPO initialization...")

    n_agents = 3
    obs_dim = 36
    action_dim = 3
    state_dim = 15

    agent = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        use_gat=True,
        device='cpu',
    )

    assert agent.actor is not None
    assert agent.critic is not None
    assert agent.actor_optimizer is not None
    assert agent.critic_optimizer is not None

    print("✓ MAPPO initialization test passed")


def test_mappo_action_selection():
    """Test MAPPO action selection."""
    print("Testing MAPPO action selection...")

    n_agents = 3
    obs_dim = 36
    action_dim = 3

    agent = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_gat=True,
        device='cpu',
    )

    obs = np.random.randn(n_agents, obs_dim)
    positions = np.random.randn(n_agents, 3)

    # Test stochastic action selection
    actions, log_probs, values, adj = agent.select_action(obs, positions, deterministic=False)

    assert actions.shape == (n_agents, action_dim)
    assert log_probs.shape == (n_agents,)
    assert values.shape == (n_agents,)
    assert adj.shape == (n_agents, n_agents)

    # Test deterministic action selection
    actions_det, _, _, _ = agent.select_action(obs, positions, deterministic=True)
    assert actions_det.shape == (n_agents, action_dim)

    print("✓ MAPPO action selection test passed")


def test_mappo_update():
    """Test MAPPO update."""
    print("Testing MAPPO update...")

    n_agents = 3
    obs_dim = 36
    action_dim = 3

    agent = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_gat=True,
        device='cpu',
    )

    # Collect some fake rollout data
    for t in range(50):
        obs = np.random.randn(n_agents, obs_dim)
        positions = np.random.randn(n_agents, 3)

        actions, log_probs, values, adj = agent.select_action(obs, positions)
        rewards = np.random.randn(n_agents)
        dones = np.zeros(n_agents)

        agent.buffer.store(obs, actions, rewards, dones, log_probs, values, adj)

    # Perform update
    metrics = agent.update(n_epochs=2, batch_size=32)

    assert 'actor_loss' in metrics
    assert 'critic_loss' in metrics
    assert 'entropy' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

    print("✓ MAPPO update test passed")


def test_save_load():
    """Test model save and load."""
    print("Testing save/load...")

    import tempfile

    n_agents = 3
    obs_dim = 36
    action_dim = 3

    agent = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_gat=True,
        device='cpu',
    )

    # Get initial parameters
    initial_actor_param = next(agent.actor.parameters()).clone()

    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        save_path = f.name
        agent.save(save_path)

    # Modify parameters
    for param in agent.actor.parameters():
        param.data.fill_(0.0)

    # Load
    agent.load(save_path)

    # Check parameters are restored
    loaded_actor_param = next(agent.actor.parameters())
    assert torch.allclose(initial_actor_param, loaded_actor_param), "Parameters should be restored"

    # Cleanup
    import os
    os.remove(save_path)

    print("✓ Save/load test passed")


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("Running GNN-MAPPO Unit Tests")
    print("="*80)

    tests = [
        test_gat_layer,
        test_gat_encoder,
        test_gat_actor,
        test_gat_critic,
        test_adjacency_matrix,
        test_rollout_buffer,
        test_mappo_initialization,
        test_mappo_action_selection,
        test_mappo_update,
        test_save_load,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
