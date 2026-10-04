"""Causal measurement history and shared scoring of executable local actions.

No simulator handle, flow label or future measurement is used. The history
encoder and candidate scorer are learned; the physical projection is shared
with the conventional controls and is not counted as a learning contribution.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from marl.local_maneuver_learning import FEATURE_DIM, N_ACTIONS

HISTORY = 6
CANDIDATE_DIM = 9
INPUT_DIM = HISTORY*FEATURE_DIM+N_ACTIONS*CANDIDATE_DIM


def candidate_features(packet, candidates, details, *, speed=1., horizon=.5, spacing=2.):
    """Features are computable from the same packet and known actuator choices."""
    n = len(candidates)
    f = np.zeros((n, N_ACTIONS, CANDIDATE_DIM), np.float32)
    f[:, :, :3] = candidates
    f[:, :, 3] = np.clip(details['projection_residual'], 0., 5.)
    goal = packet.navigation[:, 13:16]
    f[:, :, 4] = np.einsum('nkd,nd->nk', candidates, goal)
    f[:, 0, 5] = 1.  # distinguished nominal joint-filter candidate
    f[:, :, 6] = np.linalg.norm(candidates, axis=-1)
    f[:, :, 7] = 1.  # no observed peer within local range
    for i in range(n):
        peers = np.flatnonzero(packet.peer_visible[i])
        if len(peers):
            relative = packet.peer_relative_mm[i, peers]
            relative_velocity = packet.peer_relative_velocity_mm_s[i, peers]
            # Approximate forecast from measured velocity, never true flow.
            future = relative[None]+horizon*(relative_velocity[None]
                +speed*(packet.navigation[i, :3]-candidates[i])[:, None])
            f[i, :, 7] = np.clip((np.linalg.norm(future, axis=-1).min(axis=1)-spacing)/spacing, -2., 2.)
    radial = packet.navigation[:, 6:9].astype(float)
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-9)
    f[:, :, 8] = np.einsum('nkd,nd->nk', candidates, radial)
    return f


def pack_history(history, candidates):
    current = history[-1]
    padded = [np.zeros_like(current)]*(HISTORY-len(history))+list(history)
    sequence = np.stack(padded, axis=1)
    return np.concatenate((sequence.reshape(len(current), -1),
                           candidates.reshape(len(current), -1)), axis=1).astype(np.float32)


class TemporalCandidateActorCritic(nn.Module):
    """Recompute each causal window; no hidden state is leaked across resets."""
    def __init__(self, hidden=128):
        super().__init__()
        self.temporal = nn.GRU(FEATURE_DIM, hidden, batch_first=True)
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM, 64), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(hidden+64, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden+64, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.nominal_bias = nn.Parameter(torch.tensor(3.))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def encode(self, x):
        batch = len(x)
        history = x[:, :HISTORY*FEATURE_DIM].reshape(batch, HISTORY, FEATURE_DIM)
        candidates = x[:, HISTORY*FEATURE_DIM:].reshape(batch, N_ACTIONS, CANDIDATE_DIM)
        _, hidden = self.temporal(history)
        embedded = self.candidate_encoder(candidates)
        return hidden[-1], embedded, candidates

    def distribution(self, features, valid):
        hidden, embedded, candidates = self.encode(features)
        context = hidden[:, None].expand(-1, N_ACTIONS, -1)
        logits = self.score(torch.cat((context, embedded), dim=-1)).squeeze(-1)
        logits = logits+self.nominal_bias*candidates[:, :, 5]
        return Categorical(logits=logits.masked_fill(~valid, -1e9))

    def value(self, features):
        hidden, embedded, _ = self.encode(features)
        return self.critic(torch.cat((hidden, embedded.mean(dim=1)), dim=-1)).squeeze(-1)
