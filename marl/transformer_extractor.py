"""Transformer feature extractor with the same topology as IPPOExtractor.

The six branches, their 32-d widths and the 128-d trunk output are unchanged, so
any policy head built for IPPOExtractor fits without modification. What changes is
how the set-valued inputs are encoded: candidates, obstacle tracks and peers were
flattened or mean-pooled, which discards the relation between set members; here
each set gets self-attention before pooling, and the motion history gets a
transformer encoder on top of the GRU. Validity masks become attention padding
masks, so an invalid slot contributes nothing rather than being averaged in.
"""
from __future__ import annotations
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

WIDTH = 32
TRUNK = 128


def _encoder(width=WIDTH, heads=4, layers=1, ff=64, dropout=0.):
    layer = nn.TransformerEncoderLayer(width, heads, ff, dropout=dropout,
                                       batch_first=True, norm_first=True,
                                       activation='gelu')
    return nn.TransformerEncoder(layer, layers)


def _attend(encoder, tokens, valid):
    """Self-attention over a set, then mean over the valid members.

    A row whose every member is invalid would make the attention output NaN and
    the mean undefined, so such rows attend freely and are zeroed afterwards.
    """
    keep = valid > .5                                   # [B, T]
    empty = ~keep.any(dim=1, keepdim=True)              # [B, 1]
    pad = ~keep & ~empty                                # all-invalid rows: no padding
    out = encoder(tokens, src_key_padding_mask=pad)
    w = keep.to(out.dtype).unsqueeze(-1)
    pooled = (out * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.)
    return pooled * (~empty).to(out.dtype)


BRANCHES = ('hist', 'candidates', 'obstacles', 'peers')


def _pool(encoder, tokens, valid):
    """Attention pooling when the branch is selected, masked mean otherwise."""
    if encoder is not None:
        return _attend(encoder, tokens, valid)
    w = (valid > .5).to(tokens.dtype).unsqueeze(-1)
    return (tokens * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.)


class TransformerIPPOExtractor(BaseFeaturesExtractor):
    """Self-attention on a chosen subset of the set-valued branches.

    `branches` selects which branches use attention; the rest keep the original
    pooling (flatten for candidates, masked mean for obstacles/peers, GRU last
    state for history). This exists so a gain can be attributed to one branch
    instead of to "transformer" as a whole: the N=3 collapse implicates `peers`
    specifically, because it is the only branch whose width tracks the cluster
    count and it was mean-pooled, erasing which neighbour is where.
    """

    def __init__(self, observation_space, heads=4, layers=1, branches=BRANCHES):
        super().__init__(observation_space, features_dim=TRUNK)
        cand = observation_space['candidates'].shape[0]
        self.candidate_count = cand
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(96, WIDTH), nn.Tanh())
        bad = [b for b in branches if b not in BRANCHES]
        if bad:
            raise ValueError(f'unknown branch(es): {bad}; choose from {BRANCHES}')
        self.branches = tuple(branches)
        self.gru = nn.GRU(11, WIDTH, batch_first=True)
        self.hist_attn = _encoder(heads=heads, layers=layers) if 'hist' in self.branches else None
        self.ctx = nn.Sequential(nn.Linear(32, WIDTH), nn.LayerNorm(WIDTH), nn.GELU())
        if 'candidates' in self.branches:
            self.cand_embed = nn.Sequential(nn.Linear(3, WIDTH), nn.GELU())
            self.cand_attn = _encoder(heads=heads, layers=layers)
        else:   # original: flatten the whole candidate matrix
            self.cand_embed = None
            self.candidate = nn.Sequential(nn.Linear(cand * 3, WIDTH), nn.GELU())
        self.obstacle_embed = nn.Sequential(nn.Linear(25, WIDTH), nn.GELU())
        self.obstacle_attn = _encoder(heads=heads, layers=layers) if 'obstacles' in self.branches else None
        self.peer_embed = nn.Sequential(nn.Linear(8, WIDTH), nn.GELU())
        self.peer_attn = _encoder(heads=heads, layers=layers) if 'peers' in self.branches else None
        self.trunk = nn.Sequential(nn.Linear(6 * WIDTH, TRUNK),
                                   nn.LayerNorm(TRUNK), nn.GELU())
        self.flow_adapter = None
        if 'flow' in observation_space.spaces:
            self.flow_adapter = nn.Sequential(
                nn.Linear(int(observation_space['flow'].shape[0]), WIDTH), nn.GELU(),
                nn.Linear(WIDTH, TRUNK))
            nn.init.zeros_(self.flow_adapter[-1].weight)
            nn.init.zeros_(self.flow_adapter[-1].bias)

    def forward(self, obs):
        seq, last = self.gru(obs['hist'].float())
        hvalid = 1. - obs['hmask'].float()
        h = _attend(self.hist_attn, seq, hvalid) if self.hist_attn is not None else last[-1]
        image = self.cnn(obs['image'].float())
        ctx = self.ctx(obs['ctx'].float())
        if self.cand_embed is not None:
            cand = _attend(self.cand_attn, self.cand_embed(obs['candidates'].float()),
                           obs['candidate_valid'].float())
        else:
            cand = self.candidate(obs['candidates'].float().flatten(1))
        obst = _pool(self.obstacle_attn, self.obstacle_embed(obs['obstacles'].float()),
                     obs['obstacle_valid'].float())
        peer = _pool(self.peer_attn, self.peer_embed(obs['peers'].float()),
                     obs['peer_valid'].float())
        z = self.trunk(torch.cat([h, image, ctx, obst, peer, cand], dim=-1))
        if self.flow_adapter is not None:
            z = z + self.flow_adapter(obs['flow'].float())
        return z
