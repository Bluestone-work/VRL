from collections import deque
import numpy as np
import torch

from marl.multicluster import ClusterObservation
from marl.local_maneuver_learning import FEATURE_DIM, N_ACTIONS
from marl.temporal_candidate_learning import (HISTORY, CANDIDATE_DIM, INPUT_DIM,
    TemporalCandidateActorCritic, candidate_features, pack_history)


def inputs(n=2):
    rng = np.random.default_rng(49)
    history = deque([rng.normal(size=(n, FEATURE_DIM)).astype(np.float32) for _ in range(HISTORY)], maxlen=HISTORY)
    candidates = rng.normal(size=(n, N_ACTIONS, CANDIDATE_DIM)).astype(np.float32)
    candidates[:, :, 5] = 0.; candidates[:, 0, 5] = 1.
    return torch.from_numpy(pack_history(history, candidates))


def test_initial_temporal_policy_is_shared_joint_baseline():
    model = TemporalCandidateActorCritic()
    x = inputs()
    assert x.shape == (2, INPUT_DIM)
    assert torch.equal(model.distribution(x, torch.ones(2, 10, dtype=torch.bool)).logits.argmax(-1), torch.zeros(2, dtype=torch.long))


def test_candidate_scoring_is_equivariant_after_nonzero_learning_weights():
    torch.manual_seed(49)
    model = TemporalCandidateActorCritic()
    torch.nn.init.normal_(model.score[-1].weight)
    x = inputs()
    permutation = torch.tensor([0, 1, 2, 3, 7, 5, 4, 6, 8, 9])
    y = x.clone()
    y[:, HISTORY*FEATURE_DIM:] = x[:, HISTORY*FEATURE_DIM:].reshape(2, 10, CANDIDATE_DIM)[:, permutation].reshape(2, -1)
    valid = torch.ones(2, 10, dtype=torch.bool)
    a, b = model.distribution(x, valid).probs, model.distribution(y, valid).probs
    assert torch.allclose(a[:, permutation], b, atol=1e-6)


def test_gradients_reach_history_and_candidate_encoders():
    torch.manual_seed(50)
    model = TemporalCandidateActorCritic()
    torch.nn.init.normal_(model.score[-1].weight, std=.1)
    x = inputs(); x.requires_grad_(True)
    valid = torch.ones(2, 10, dtype=torch.bool); valid[:, 7] = False
    d = model.distribution(x, valid)
    assert torch.all(d.probs[:, 7] == 0)
    loss = -d.log_prob(torch.tensor([4, 5])).mean()+model.value(x).square().mean()
    loss.backward()
    assert model.temporal.weight_ih_l0.grad.abs().sum() > 0
    assert model.candidate_encoder[0].weight.grad.abs().sum() > 0
    assert x.grad[:, :FEATURE_DIM].abs().sum() > 0


def test_history_padding_never_uses_future_frames():
    h = deque([np.ones((2, FEATURE_DIM), np.float32)], maxlen=HISTORY)
    packed = pack_history(h, np.zeros((2, 10, CANDIDATE_DIM)))
    assert np.all(packed[:, :(HISTORY-1)*FEATURE_DIM] == 0.)
    assert np.all(packed[:, (HISTORY-1)*FEATURE_DIM:HISTORY*FEATURE_DIM] == 1.)


def test_candidate_descriptors_use_only_measured_packet():
    p = ClusterObservation(np.zeros((2, 111), np.float32), np.zeros((2, 4), int),
        np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.zeros((2, 2), bool), np.ones(2, bool))
    candidates = np.zeros((2, 10, 3))
    details = {'projection_residual': np.zeros((2, 10))}
    x = candidate_features(p, candidates, details)
    assert x.shape == (2, 10, CANDIDATE_DIM) and np.isfinite(x).all()
    assert np.all(x[:, :, 7] == 1.)
