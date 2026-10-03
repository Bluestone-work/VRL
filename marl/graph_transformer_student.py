"""Topology-aware Graph Transformer student for Direct Local control.

One shared network for every robot (parameter sharing), any number of robots,
clots, particles and vessel nodes (padding + masks), any anatomy. Input is the
scene graph of `marl.scene_graph`; no routed features, no allocation.

Layer: multi-head attention over all valid nodes of a scene. Pairwise relations
enter twice, as in edge-aware graph transformers: an MLP of the relation gives a
per-head attention bias, and the attention-weighted relation is added to the
message, so a robot can learn *where* (in its own Frenet frame) the nodes it
attends to lie. Relations:
  * relative position of j in i's local frame (direction, log distance)
  * relative velocity in i's frame, closest-approach time and clearance
  * geodesic distance between the attached centreline stations (log)
  * vessel adjacency, j downstream of i, j upstream of i (rooted tree)
  * node-type pair
Heads, on robot nodes only:
  * action: Frenet direction (3) + stop logit -> [u_par, u_n, u_b, stop]
  * subgoal: score per clot (auxiliary; trained on the teacher's allocation)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from marl.scene_graph import CLOT, PARTICLE, ROBOT, VESSEL, VESSEL_DIM, ROBOT_DIM, CLOT_DIM, PARTICLE_DIM

PAIR_DIM = 3+1+3+2+1+3+16
TYPES = 4


def collate_scenes(scenes, device='cpu'):
    """Pad a list of `extract_scene` dicts into one batch of node and pair tensors."""
    per = []
    for s in scenes:
        kept = s['kept']
        st = np.concatenate((kept, s['robot_station'], s['clot_station'], s['particle_station']))
        pos = np.concatenate((s['station_points'][kept], s['robot_pos'], s['clot_pos'], s['particle_pos']))
        vel = np.concatenate((np.zeros((len(kept), 3), np.float32), s['robot_vel'],
                              np.zeros((len(s['clot_pos']), 3), np.float32), s['particle_vel']))
        typ = np.concatenate((np.full(len(kept), VESSEL), np.full(len(s['robot_pos']), ROBOT),
                              np.full(len(s['clot_pos']), CLOT), np.full(len(s['particle_pos']), PARTICLE)))
        mask = np.concatenate((np.ones(len(kept), bool), s['robot_active'], s['clot_alive'], s['particle_active']))
        feats = [s['vessel'], s['robot'], s['clot'], s['particle']]
        nv = len(kept)
        adj = np.zeros((len(st), len(st)), bool); adj[:nv, :nv] = s['vessel_adjacency']
        per.append(dict(st=st, pos=pos, vel=vel, typ=typ, mask=mask, feats=feats,
                        frame=s['station_frames'][st], geo=s['geodesic'][np.ix_(st, st)], depth=s['depth'][st], adj=adj,
                        n_robot=len(s['robot_pos']), n_clot=len(s['clot_pos']), robot_off=nv, clot_off=nv+len(s['robot_pos']),
                        scale=np.float32(s['robot_radius']+s['particle_radius']), speed=np.float32(s['robot_speed'])))
    B, N = len(per), max(len(p['st']) for p in per)
    R, C = max(p['n_robot'] for p in per), max(p['n_clot'] for p in per)
    out = dict(pos=np.zeros((B, N, 3), np.float32), vel=np.zeros((B, N, 3), np.float32), typ=np.zeros((B, N), np.int64),
               mask=np.zeros((B, N), bool), frame=np.tile(np.eye(3, dtype=np.float32), (B, N, 1, 1)),
               geo=np.zeros((B, N, N), np.float32), depth=np.zeros((B, N), np.float32), adj=np.zeros((B, N, N), bool),
               scale=np.zeros(B, np.float32), speed=np.zeros(B, np.float32),
               robot_index=np.zeros((B, R), np.int64), robot_mask=np.zeros((B, R), bool),
               clot_index=np.zeros((B, C), np.int64), clot_mask=np.zeros((B, C), bool))
    dims = (VESSEL_DIM, ROBOT_DIM, CLOT_DIM, PARTICLE_DIM)
    for k, d in enumerate(dims):
        out[f'x{k}'] = np.zeros((B, N, d), np.float32)
    for b, p in enumerate(per):
        n = len(p['st'])
        for key in ('pos', 'vel', 'typ', 'mask', 'frame', 'depth'):
            out[key][b, :n] = p[key]
        out['geo'][b, :n, :n] = p['geo']; out['adj'][b, :n, :n] = p['adj']
        out['scale'][b], out['speed'][b] = p['scale'], p['speed']
        offset = 0
        for k, f in enumerate(p['feats']):
            out[f'x{k}'][b, offset:offset+len(f)] = f; offset += len(f)
        r, c = p['n_robot'], p['n_clot']
        out['robot_index'][b, :r] = p['robot_off']+np.arange(r); out['robot_mask'][b, :r] = p['mask'][p['robot_off']:p['robot_off']+r]
        out['clot_index'][b, :c] = p['clot_off']+np.arange(c); out['clot_mask'][b, :c] = p['mask'][p['clot_off']:p['clot_off']+c]
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


def pair_features(batch, horizon_s=1.0):
    """[B, N, N, PAIR_DIM] relations, j seen from i (i's Frenet frame)."""
    pos, vel, frame = batch['pos'], batch['vel'], batch['frame']
    rel = pos[:, None, :, :]-pos[:, :, None, :]
    rel_local = torch.einsum('bikj,binj->bink', frame, rel)
    dist = rel.norm(dim=-1, keepdim=True)
    direction = rel_local/dist.clamp_min(1e-6)
    rv = vel[:, None, :, :]-vel[:, :, None, :]
    rv_local = torch.einsum('bikj,binj->bink', frame, rv)
    speed = batch['speed'][:, None, None, None]
    t = (-(rel*rv).sum(-1)/(rv*rv).sum(-1).clamp_min(1e-9)).clamp(0, horizon_s)
    clear = (rel+t[..., None]*rv).norm(dim=-1)-batch['scale'][:, None, None]
    geo, depth = batch['geo'], batch['depth']
    tol = 0.5+0.02*geo
    down = ((depth[:, None, :]-depth[:, :, None]-geo).abs() < tol) & (geo > 0)
    up = ((depth[:, :, None]-depth[:, None, :]-geo).abs() < tol) & (geo > 0)
    typ = batch['typ']
    tp = nn.functional.one_hot(typ[:, :, None]*TYPES+typ[:, None, :], TYPES*TYPES).float()
    return torch.cat((direction, torch.log1p(dist), (rv_local/speed).clamp(-10, 10),
                      (t/horizon_s)[..., None], torch.tanh(clear/0.15)[..., None],
                      torch.log1p(geo)[..., None], batch['adj'].float()[..., None],
                      down.float()[..., None], up.float()[..., None], tp), dim=-1)


PAIR_EMBED = 32


class EdgeAttentionLayer(nn.Module):
    """Attention with a per-head bias from the shared pair embedding and a per-head
    attention-weighted relation term (sum_j a_ij e_ij, then a per-head projection),
    so memory stays O(B N^2 (H + P)) instead of O(B N^2 D)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dk = heads, dim//heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.bias = nn.Linear(PAIR_EMBED, heads)
        self.edge_value = nn.Parameter(torch.randn(heads, PAIR_EMBED, self.dk)*PAIR_EMBED**-.5)
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))

    def forward(self, x, pair_embed, mask):
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.h, self.dk).unbind(2)
        logits = torch.einsum('bihd,bjhd->bhij', q, k)/self.dk**.5 + self.bias(pair_embed).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], -1e9)
        attn = logits.softmax(-1)
        msg = torch.einsum('bhij,bjhd->bihd', attn, v)
        rel = torch.einsum('bhij,bijp->bhip', attn, pair_embed)          # where the attended nodes are
        msg = msg+torch.einsum('bhip,hpd->bihd', rel, self.edge_value)
        x = x+self.out(msg.reshape(B, N, D))
        return x+self.ff(self.norm2(x))


class GraphTransformerStudent(nn.Module):
    def __init__(self, dim=128, heads=4, layers=4):
        super().__init__()
        self.embed = nn.ModuleList(nn.Linear(d, dim) for d in (VESSEL_DIM, ROBOT_DIM, CLOT_DIM, PARTICLE_DIM))
        self.type_embed = nn.Embedding(TYPES, dim)
        self.pair_embed = nn.Sequential(nn.LayerNorm(PAIR_DIM), nn.Linear(PAIR_DIM, 64), nn.GELU(), nn.Linear(64, PAIR_EMBED))
        self.layers = nn.ModuleList(EdgeAttentionLayer(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.action = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        self.subgoal = nn.Sequential(nn.Linear(2*dim+PAIR_EMBED, dim), nn.GELU(), nn.Linear(dim, 1))
        self.config = dict(dim=dim, heads=heads, layers=layers)

    def forward(self, batch):
        typ = batch['typ']
        x = self.type_embed(typ)
        for k, emb in enumerate(self.embed):
            x = x+emb(batch[f'x{k}'])*(typ == k)[..., None]
        pair = self.pair_embed(pair_features(batch))
        mask = batch['mask']
        for layer in self.layers:
            x = layer(x, pair, mask)
        x = self.norm(x)
        ri, ci = batch['robot_index'], batch['clot_index']
        gather = lambda idx: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
        hr, hc = gather(ri), gather(ci)
        out = self.action(hr)
        rc_pair = pair[torch.arange(len(x), device=x.device)[:, None, None], ri[:, :, None], ci[:, None, :]]
        sub = self.subgoal(torch.cat((hr[:, :, None].expand(-1, -1, hc.shape[1], -1),
                                      hc[:, None].expand(-1, hr.shape[1], -1, -1), rc_pair), -1)).squeeze(-1)
        sub = sub.masked_fill(~batch['clot_mask'][:, None, :], -1e9)
        return dict(direction=out[..., :3], stop_logit=out[..., 3], subgoal_logits=sub)


def student_local_action(model, scenes, device='cpu', stop_threshold=0.5):
    """Deterministic Direct Local action [n_robot, 3] per scene: unit Frenet direction or exact stop."""
    with torch.no_grad():
        out = model(collate_scenes(scenes, device))
    direction = nn.functional.normalize(out['direction'], dim=-1)
    stop = torch.sigmoid(out['stop_logit']) > stop_threshold
    action = torch.where(stop[..., None], torch.zeros_like(direction), direction).cpu().numpy()
    return [action[b, :len(s['robot_pos'])] for b, s in enumerate(scenes)]


def save_student(model, path, **meta):
    torch.save(dict(config=model.config, state=model.state_dict(), meta=meta), path)


def load_student(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GraphTransformerStudent(**payload['config']).to(device)
    model.load_state_dict(payload['state']); model.eval()
    model.meta = payload.get('meta', {})
    return model
