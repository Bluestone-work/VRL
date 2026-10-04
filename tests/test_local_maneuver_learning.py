from dataclasses import replace
import numpy as np
import pytest
import torch

from marl.multicluster import ClusterObservation, MultiClusterConfig
from marl.local_maneuver_learning import project_joint, ManeuverLibrary, LocalManeuverActorCritic, FEATURE_DIM
from scripts.run_local_learning import advantages


def packet(n=2):
    nav = np.zeros((n, 111), np.float32)
    nav[:, 6] = .9
    nav[:, 9] = 0.
    nav[:, 35] = 1.; nav[:, 44] = 0.
    nav[:, 11] = 1.; nav[:, 12] = 1.; nav[:, 13] = 1.; nav[:, 16] = 1.
    return ClusterObservation(nav, np.zeros((n, 4), np.int32),
        np.zeros((n, n, 3)), np.zeros((n, n, 3)), np.zeros((n, n), bool), np.ones(n, bool))


def test_joint_projection_rejects_outward_command_at_wall():
    p = packet()
    cfg = MultiClusterConfig(clusters=2)
    action, residual = project_joint(np.array([[1., 0, 0], [1., 0, 0]]), p, cfg)
    assert np.all(action[:, 0] <= -.079)
    assert residual.max() < 1e-6
    assert np.linalg.norm(action, axis=1).max() <= 1+1e-12


def test_conflicting_wall_and_peer_constraints_are_reported_not_certified():
    p = packet()
    p.peer_visible[0, 1] = True
    p.peer_relative_mm[0, 1, 0] = -1.
    action, residual = project_joint(np.zeros((2, 3)), p, MultiClusterConfig(clusters=2))
    assert np.isfinite(action).all()
    assert residual[0] > .01
    assert np.linalg.norm(action, axis=1).max() <= 1+1e-12


def test_batched_candidate_projection_matches_scalar_projection():
    p = packet()
    cfg = MultiClusterConfig(clusters=2)
    raw = np.random.default_rng(3).normal(size=(2, 10, 3))
    actions, residual = project_joint(raw, p, cfg)
    for k in range(10):
        a, r = project_joint(raw[:, k], p, cfg)
        assert np.allclose(actions[:, k], a)
        assert np.allclose(residual[:, k], r)


def test_observation_only_library_and_invalid_maneuver_mask():
    p = packet()
    lib = ManeuverLibrary(MultiClusterConfig(clusters=2))
    x, candidates, valid, details = lib.prepare(p)
    assert x.shape == (2, FEATURE_DIM)
    assert candidates.shape == (2, 10, 3)
    assert np.isfinite(x).all() and np.isfinite(candidates).all()
    assert not valid[:, 5:8].any()
    model = LocalManeuverActorCritic()
    dist = model.distribution(torch.from_numpy(x), torch.from_numpy(valid))
    assert torch.all(dist.probs[:, 5:8] == 0)
    assert torch.all(dist.logits.argmax(dim=-1) == 0)
    lib.commit(np.zeros(2, np.int64), details)
    assert not hasattr(lib, 'env')


def test_gae_does_not_bootstrap_across_terminal_reset():
    rewards = np.array([[1.], [100.]], np.float32)
    values = np.zeros_like(rewards)
    dones = np.array([[1.], [0.]], np.float32)
    adv, returns = advantages(rewards, values, dones, np.array([10.]), .9, 1.)
    assert adv[0, 0] == 1.
    assert returns[1, 0] == pytest.approx(109.)


def test_gradients_update_learned_selector():
    torch.manual_seed(1)
    model = LocalManeuverActorCritic()
    x = torch.randn(16, FEATURE_DIM)
    valid = torch.ones(16, 10, dtype=torch.bool)
    before = model.actor[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = -model.distribution(x, valid).log_prob(torch.full((16,), 3)).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert not torch.equal(before, model.actor[-1].weight)
