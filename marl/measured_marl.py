"""Measurement-only MARL adaptations over a shared local reservation interface.

The event bids adapt MAPPO/IPPO to ordering, and are not author-code reproductions.
All actors receive the same current team packet. Only the centralized critics
pool past histories. No policy object receives an environment reference.
"""
from itertools import permutations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from marl.tpg_learning import LOW_DIM, NODE_DIM, PAIR_DIM, N_LOW, CANDIDATE_DIM


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, out))


class MeasuredEncoder(nn.Module):
    def __init__(self, hidden, clusters=3):
        super().__init__()
        self.temporal = nn.GRU(LOW_DIM, hidden, batch_first=True)
        self.team = mlp(clusters * NODE_DIM + clusters * clusters * PAIR_DIM, hidden, hidden)

    def forward(self, history, nodes, pairs):
        b, n, t, d = history.shape
        _, state = self.temporal(history.reshape(b*n, t, d))
        local = state[-1].reshape(b, n, -1)
        team = self.team(torch.cat((nodes.flatten(1), pairs.flatten(1)), -1))
        return local, team[:, None].expand(-1, n, -1)


class BidActor(nn.Module):
    def __init__(self, hidden=64, clusters=3):
        super().__init__()
        self.encoder = MeasuredEncoder(hidden, clusters)
        self.score = mlp(2*hidden + NODE_DIM, hidden, clusters)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, state):
        local, team = self.encoder(state['history'], state['nodes'], state['pairs'])
        logits = self.score(torch.cat((local, team, state['nodes']), -1))
        return Categorical(logits=logits)


class JointGraphActor(nn.Module):
    def __init__(self, hidden=64, clusters=3):
        super().__init__()
        self.encoder = MeasuredEncoder(hidden, clusters)
        self.current_node = mlp(NODE_DIM, hidden, hidden)
        self.message = mlp(2*hidden + PAIR_DIM, hidden, hidden)
        self.score = mlp(3*hidden + NODE_DIM, hidden, 1)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, state):
        local, team = self.encoder(state['history'], state['nodes'], state['pairs'])
        b, n, _ = local.shape
        pairs, nodes = state['pairs'], state['nodes']
        current = self.current_node(nodes)
        mask = pairs[..., 5:6]
        msg = self.message(torch.cat((local[:, :, None].expand(-1, -1, n, -1),
            current[:, None].expand(-1, n, -1, -1), pairs), -1))*mask
        neighbor = msg.sum(2)/mask.sum(2).clamp_min(1.)
        score = self.score(torch.cat((local, team, neighbor, nodes), -1)).squeeze(-1)
        cost = torch.where(nodes[..., 15] > .5, nodes[..., 14], nodes[..., 17]+.2)
        score = score - .5*(cost + .0001*torch.arange(n, device=nodes.device))
        logits, admitted = [], []
        for order in permutations(range(n)):
            logits.append(sum((n-rank)*score[:, i] for rank, i in enumerate(order)))
            rank = {i: r for r, i in enumerate(order)}
            bad = torch.zeros(b, dtype=torch.bool, device=nodes.device)
            for i in range(n):
                for j in range(n):
                    if rank[i] > rank[j]:
                        bad |= pairs[:, i, j, 4] > .5
            admitted.append(~bad)
        return Categorical(logits=torch.stack(logits, -1).masked_fill(~torch.stack(admitted, -1), -1e9))


class CandidateActor(nn.Module):
    def __init__(self, hidden=64, clusters=3, action_conditioned=False, prior=.5):
        super().__init__()
        self.encoder = MeasuredEncoder(hidden, clusters)
        self.candidate = mlp(CANDIDATE_DIM, 32, 32)
        self.dynamics = mlp(2*hidden + 3, hidden, 3)
        # All variants have the same feature width; no-ac keeps zero slots.
        self.score = mlp(2*hidden + 32 + 3, hidden, 1)
        self.action_conditioned = action_conditioned
        self.prior = prior
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def representations(self, state):
        a, b = self.encoder(state['history'], state['nodes'], state['pairs'])
        return torch.cat((a, b), -1)

    def predict(self, state, commands):
        return self.dynamics(torch.cat((self.representations(state), commands), -1))

    def forward(self, state):
        z = self.representations(state)
        repeated = z[:, :, None].expand(-1, -1, N_LOW, -1)
        candidate = state['candidates']
        predicted = self.dynamics(torch.cat((repeated, candidate[..., :3]), -1))
        if not self.action_conditioned:
            predicted = torch.zeros_like(predicted)
        logits = self.score(torch.cat((repeated, self.candidate(candidate), predicted), -1)).squeeze(-1)
        logits = logits + self.prior*candidate[..., 8]
        return Categorical(logits=logits.masked_fill(~state['valid'], -1e9))


class MeasuredCritic(nn.Module):
    def __init__(self, hidden=64, clusters=3, centralized=True, joint=False):
        super().__init__()
        self.encoder = MeasuredEncoder(hidden, clusters)
        self.centralized, self.joint = centralized, joint
        self.head = mlp((3 if centralized else 2)*hidden, hidden, 1)

    def forward(self, state):
        local, team = self.encoder(state['history'], state['nodes'], state['pairs'])
        w = state['active'][..., None].float()
        parts = [local, team]
        if self.centralized:
            pooled = (local*w).sum(1)/w.sum(1).clamp_min(1.)
            parts.append(pooled[:, None].expand_as(local))
        value = self.head(torch.cat(parts, -1)).squeeze(-1)
        if self.joint:
            return (value*w.squeeze(-1)).sum(-1)/w.sum((1, 2)).clamp_min(1.)
        return value


class MeasuredMARL(nn.Module):
    def __init__(self, variant, hidden=64, clusters=3, prior=.5):
        super().__init__()
        if variant not in ('r_mappo', 'r_ippo', 'vctpg_ac', 'vctpg_no_ac'):
            raise ValueError(variant)
        self.variant = variant
        self.joint_high = variant.startswith('vctpg')
        # Actors constructed before critics: MAPPO/IPPO actor initial weights match.
        self.high_actor = (JointGraphActor if self.joint_high else BidActor)(hidden, clusters)
        self.low_actor = CandidateActor(hidden, clusters, variant == 'vctpg_ac', prior)
        self.high_critic = MeasuredCritic(hidden, clusters, variant != 'r_ippo', self.joint_high)
        self.low_critic = MeasuredCritic(hidden, clusters, variant != 'r_ippo')

    def forward(self, level, state):
        return getattr(self, level+'_actor')(state), getattr(self, level+'_critic')(state)

    def parameters_for(self, level):
        return list(getattr(self, level+'_actor').parameters()) + list(getattr(self, level+'_critic').parameters())


def bids_to_priority(bids, nodes, pairs):
    """Topological sort with learned bids; no truth or environment access.

    Lower bid means earlier preference. Equal bids reproduce measured rule order.
    Any feasible permutation can be expressed by assigning distinct ranks.
    """
    bids = np.asarray(bids)
    n = len(bids)
    costs = np.where(nodes[:, 15] > .5, nodes[:, 14], nodes[:, 17]+.2)
    pending, order = set(range(n)), []
    while pending:
        ready = [i for i in pending if not any(pairs[j, i, 4] > .5 for j in pending)]
        if not ready:
            raise ValueError('Cyclic measured reservation relation')
        chosen = min(ready, key=lambda i: (int(bids[i]), float(costs[i]), i))
        order.append(chosen)
        pending.remove(chosen)
    return list(permutations(range(n))).index(tuple(order))


def temporal_gae(records, bootstrap, gamma, lam):
    """Per-agent or joint GAE, with physical duration and explicit next values.

    Next values are logged at the actual successor before any policy update;
    this also bootstraps censored control rollouts instead of labelling them done.
    """
    values = np.asarray([r['value'] for r in records], np.float32)
    advantages = np.zeros_like(values)
    carry = np.zeros_like(np.asarray(bootstrap, np.float32))
    for t in reversed(range(len(records))):
        r = records[t]
        next_value = np.asarray(r.get('next_value', bootstrap if t == len(records)-1 else records[t+1]['value']))
        mask = 0. if r['done'] else 1.
        discount, trace = gamma**r['duration'], (gamma*lam)**r['duration']
        delta = r['reward'] + mask*discount*next_value - values[t]
        carry = delta + mask*trace*carry
        advantages[t] = carry
    return advantages, advantages + values
