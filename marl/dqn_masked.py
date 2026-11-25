"""Q-networks that use the candidate validity mask, plus a dueling head.

The action space is an index into a per-step candidate set, and the observation
already carries `candidate_valid` marking slots that are duplicated or not
executable. Two consequences the stock DQN head does not exploit:

* Masking. An invalid index is canonicalised to an equivalent valid command
  before execution, so without masking the network spends capacity regressing
  values for actions that can never be taken, and the bootstrap target maximises
  over them. Masking the logits to a large negative value removes both effects.
  Unlike PPO, this needs no custom distribution — only the Q vector is touched.

* Dueling. Scoring candidates is naturally "how good is this state" plus "how
  much better is this candidate than the alternatives", which is exactly the
  V/A decomposition. The advantage stream is centred over the *valid* slots only,
  so a varying number of valid candidates does not shift V.
"""
from __future__ import annotations
import torch
from torch import nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

NEG = -1e8          # finite, so masked_fill never produces NaN downstream
KEY = 'candidate_valid'


def _valid_mask(obs, n_actions):
    """[B, n_actions] bool mask; every row keeps at least one action."""
    v = obs[KEY]
    keep = v > .5
    if keep.shape[-1] != n_actions:                 # defensive: shapes must line up
        raise ValueError(f'{KEY} has width {keep.shape[-1]}, expected {n_actions}')
    # The zero command is contractually always valid, but guard anyway: an
    # all-invalid row would make argmax and the max-target meaningless.
    empty = ~keep.any(dim=-1, keepdim=True)
    return keep | empty


class MaskedQNetwork(QNetwork):
    def forward(self, obs):
        q = self.q_net(self.extract_features(obs, self.features_extractor))
        return q.masked_fill(~_valid_mask(obs, q.shape[-1]), NEG)


class DuelingQNetwork(QNetwork):
    """V(s) + centred A(s,a), with centring restricted to valid actions."""

    def __init__(self, *args, masked=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked = masked
        width = self.features_dim
        self.value_stream = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        advantage = self.q_net(features)
        value = self.value_stream(features)
        keep = _valid_mask(obs, advantage.shape[-1])
        w = keep.to(advantage.dtype)
        # Centre the advantage over valid slots so V is identifiable regardless
        # of how many candidates are live this step.
        mean = (advantage * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp(min=1.)
        q = value + advantage - mean
        return q.masked_fill(~keep, NEG) if self.masked else q


class MaskedDQNPolicy(DQNPolicy):
    """DQNPolicy whose Q-net honours the candidate mask (optionally dueling)."""
    dueling = False

    def make_q_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        cls = DuelingQNetwork if self.dueling else MaskedQNetwork
        return cls(**args).to(self.device)


class DuelingMaskedDQNPolicy(MaskedDQNPolicy):
    dueling = True
