"""Cooperative clot allocation for multi-cluster thrombolysis (benchmark v4, scripts/benchmark_lysis.py).

Motivation (v4 reference runs, 2026-10-08): with the deployable classical stack (allocation A + TPG + pursuit +
wall guard) N=2/3 reach 95 % full clearance, but A is a one-shot pre-operative plan of geodesic travel only.
Clusters that finish their own list park while others still work, a cluster stuck at the wall keeps its clots
for the whole episode (e.g. pulmonary_saddle N=2/3 seed 9: 50 % removal, one cluster pinned 180-190 s), and TPG
waits are planned with nominal speeds. Online (re)allocation from deployable feedback is the multi-cluster
decision that determines how much thrombus is cleared and how fast.

Components (all inputs deployable: pre-operative map geodesics, map-matched image estimates, which clots are
still visible, the controller's own commands; simulator truth only in the training reward)
  DynamicTPG        TPG coordinator (marl.tpg_coordinator) rebuilt from the clusters' current estimated stations
                    whenever the allocation changes
  features()        per-cluster observation (clots: visibility, geodesic from me / nearest peer, who targets it,
                    A-plan prior; me: time, stall, hold; peers: geodesic, stall)
  AuctionAllocator  classical online re-allocation (heuristic baseline/ablation): a cluster without an alive
                    clot takes the untargeted alive clot of minimum geodesic, or one still queued for a
                    peer if it can reach it sooner than that peer; stuck clusters release their clot
  AllocPolicy       learned allocator (MAPPO: shared actor on own observation, centralised critic on the team's
                    deployable observations); actor logits = MLP + w * A-prior, so the untrained policy is A
  LearnedAllocator  event-triggered execution of AllocPolicy (target cleared / every DECIDE_S seconds)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

M_CLOTS, K_MAX = 4, 3
PER_CLOT, SELF, PER_PEER = 8, 8, 4
OBS_DIM = M_CLOTS*PER_CLOT+SELF+(K_MAX-1)*PER_PEER
CRITIC_DIM = K_MAX*OBS_DIM+K_MAX
N_ACT = M_CLOTS+1            # clot 0..3, 4 = park (station-keep at the last target)
GEO = 40.                    # mm normalisation
DECIDE_S = 5.
STALL_WINDOW_S, STALL_MM = 3., .15


class DynamicTPG:
    """TPG over the current one-target-per-cluster assignment, starting from the clusters' estimated stations."""
    def __init__(self, ep):
        from marl.tpg_coordinator import TPGCoordinator
        self.ep, self.cls = ep, TPGCoordinator
        _, self.sp = ep.bm._station_paths(ep.env)
        self.coord, self.key = None, None
        self.builds = 0

    def update(self, est, targets):
        key = tuple(int(t) for t in targets)
        if key == self.key:
            return
        env = self.ep.env
        starts = np.array([int(np.argmin(np.linalg.norm(env.transport.points-est.pos[i], axis=1)))
                           for i in range(self.ep.n)])
        proxy = SimpleNamespace(robot_stations=starts, clot_stations=env.clot_stations, transport=env.transport,
                                clot_positions_mm=env.clot_positions_mm, config=env.config, initial_mass=env.initial_mass,
                                positions_mm=env.positions_mm, active=env.active)
        plan = [[int(t)] if t >= 0 else [] for t in targets]
        self.coord = self.cls(proxy, plan, self.sp, d_min_mm=self.ep.d_min)
        self.key = key; self.builds += 1

    def gate(self, pos, active):
        return self.coord.gate(pos, active) if self.coord is not None else np.zeros(self.ep.n, bool)


class StallMonitor:
    """Measured stall: commanded but < STALL_MM net displacement of the estimate over STALL_WINDOW_S."""
    def __init__(self, n, dt):
        self.k = int(round(STALL_WINDOW_S/dt)); self.buf = [[] for _ in range(n)]
        self.stalled = np.zeros(n, bool); self.stall_s = np.zeros(n); self.dt = dt

    def update(self, est, commanded):
        for i in range(len(self.buf)):
            if not commanded[i]:
                self.buf[i].clear(); self.stalled[i] = False; self.stall_s[i] = 0.; continue
            self.buf[i].append(est.pos[i].copy()); self.buf[i] = self.buf[i][-self.k:]
            self.stalled[i] = len(self.buf[i]) == self.k and np.linalg.norm(self.buf[i][-1]-self.buf[i][0]) < STALL_MM
            self.stall_s[i] = self.stall_s[i]+self.dt if self.stalled[i] else 0.


class AllocState:
    """Shared bookkeeping for online allocators."""
    def __init__(self, ep):
        self.ep = ep; env = ep.env
        self.G = ep.bm.station_geodesic(env)
        self.clot_st = np.asarray(env.clot_stations)
        self.mass_rel = np.asarray(env.initial_mass, float)/max(float(np.mean(env.initial_mass)), 1e-12)  # pre-op clot size
        self.targets = np.full(ep.n, -1); self.last = np.full(ep.n, -1)
        self.next_decide = np.zeros(ep.n)
        self.stall = StallMonitor(ep.n, float(env.config.control_dt_s))
        self.tpg = DynamicTPG(ep) if ep.n > 1 else None
        self.hold = np.zeros(ep.n, bool)
        self.parked = np.zeros(ep.n, bool)

    def station(self, est, i):
        return int(np.argmin(np.linalg.norm(self.ep.env.transport.points-est.pos[i], axis=1)))

    def geo(self, est):
        """[n, m] geodesic (mm) from every cluster's estimated station to every clot."""
        return np.stack([self.G[self.station(est, i), self.clot_st] for i in range(self.ep.n)])

    def a_next(self, alive):
        """A-plan prior: next alive clot of each cluster's pre-operative list (-1 if exhausted)."""
        out = np.full(self.ep.n, -1)
        for i, seq in enumerate(self.ep.plan):
            nxt = [j for j in seq if alive[j]]
            out[i] = nxt[0] if nxt else -1
        return out

    def resolve(self, est, choice):
        """choice in 0..M-1 (clot) or M (park: keep the previous target, i.e. station-keep there)."""
        t = np.where(choice < M_CLOTS, choice, self.last)
        self.parked = choice >= M_CLOTS
        self.targets = t.astype(int)
        self.last = np.where(self.targets >= 0, self.targets, self.last)
        return self.targets


def features(st, est, geo=None):
    ep = st.ep; n = ep.n; alive = np.asarray(est.clot_alive, bool)
    geo = st.geo(est) if geo is None else geo
    prior = st.a_next(alive); seqs = ep.plan
    X = np.zeros((n, OBS_DIM), np.float32)
    for i in range(n):
        peers = [p for p in range(n) if p != i and est.active[p]]
        o = X[i]
        for j in range(M_CLOTS):
            s = PER_CLOT*j
            o[s] = float(alive[j]); o[s+1] = min(geo[i, j]/GEO, 3.)
            o[s+2] = min(min([geo[p, j] for p in peers], default=3*GEO)/GEO, 3.)
            o[s+3] = sum(st.targets[p] == j for p in peers)/2.
            o[s+4] = float(prior[i] == j); o[s+5] = float(j in seqs[i]); o[s+6] = st.mass_rel[j]
            o[s+7] = float(st.targets[i] == j)
        s = PER_CLOT*M_CLOTS
        o[s] = ep.env.elapsed_s/ep.horizon; o[s+1] = float(st.stall.stalled[i]); o[s+2] = min(st.stall.stall_s[i]/10., 3.)
        o[s+3] = float(st.hold[i]); o[s+4] = alive.mean(); o[s+5] = n/3.; o[s+6] = float(prior[i] < 0)
        o[s+7] = float(st.targets[i] >= 0 and not alive[st.targets[i]])
        s += SELF
        for k, p in enumerate([p for p in range(n) if p != i][:K_MAX-1]):
            q = s+PER_PEER*k
            o[q] = float(est.active[p]); o[q+1] = min(float(np.linalg.norm(est.pos[p]-est.pos[i]))/GEO, 3.)
            o[q+2] = float(st.stall.stalled[p]); o[q+3] = float(st.targets[p] >= 0 and alive[st.targets[p]])
    prior_act = np.where(prior >= 0, prior, M_CLOTS)
    mask = np.zeros((n, N_ACT), bool); mask[:, :M_CLOTS] = alive[None, :]; mask[:, M_CLOTS] = True
    return X, mask, prior_act


def critic_input(X):
    n = len(X); C = np.zeros((n, CRITIC_DIM), np.float32)
    flat = np.zeros(K_MAX*OBS_DIM, np.float32); flat[:n*OBS_DIM] = X.reshape(-1)
    for i in range(n):
        own = np.roll(flat.reshape(K_MAX, OBS_DIM), -i, axis=0).reshape(-1)     # own observation first
        C[i, :K_MAX*OBS_DIM] = own; C[i, K_MAX*OBS_DIM+n-1] = 1.               # team size one-hot
    return C


class AllocPolicy(nn.Module):
    def __init__(self, d=128, prior_w=4.):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBS_DIM, d), nn.Tanh(), nn.Linear(d, d), nn.Tanh(), nn.Linear(d, N_ACT))
        nn.init.zeros_(self.actor[-1].weight); nn.init.zeros_(self.actor[-1].bias)
        self.prior_w = nn.Parameter(torch.tensor(float(prior_w)))
        self.critic = nn.Sequential(nn.Linear(CRITIC_DIM, 2*d), nn.Tanh(), nn.Linear(2*d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))

    def logits(self, x, mask, prior):
        l = self.actor(x)+self.prior_w*nn.functional.one_hot(prior, N_ACT).float()
        return l.masked_fill(~mask, -1e9)

    def value(self, c):
        return self.critic(c).squeeze(-1)


class LearnedAllocator:
    """Decides at events: own target no longer visible, own target unset, or every DECIDE_S seconds."""
    def __init__(self, ep, policy=None, greedy=True, use_tpg=True, prior_only=False):
        self.st = AllocState(ep); self.policy, self.greedy, self.use_tpg, self.prior_only = policy, greedy, use_tpg, prior_only
        self.decisions = []          # (agent, x, critic, mask, prior, action, logp, value, step)
        self.decision_count = 0

    def due(self, est):
        ep, st = self.st.ep, self.st; alive = np.asarray(est.clot_alive, bool); t = ep.env.elapsed_s
        return np.array([bool(est.active[i]) and ((not st.parked[i] and (st.targets[i] < 0 or not alive[st.targets[i]]))
                                                  or t >= st.next_decide[i]) for i in range(ep.n)])

    def __call__(self, ep, est):
        st = self.st; due = self.due(est)
        if due.any():
            X, mask, prior = features(st, est)
            choice = np.where(st.targets >= 0, st.targets, M_CLOTS).astype(int)
            alive = np.asarray(est.clot_alive, bool)
            choice = np.where((choice < M_CLOTS) & ~alive[np.minimum(choice, M_CLOTS-1)], M_CLOTS, choice)
            if self.prior_only or self.policy is None:
                choice[due] = prior[due]
            else:
                with torch.no_grad():
                    lg = self.policy.logits(torch.as_tensor(X), torch.as_tensor(mask), torch.as_tensor(prior))
                    dist = torch.distributions.Categorical(logits=lg)
                    a = lg.argmax(-1) if self.greedy else dist.sample()
                    lp = dist.log_prob(a); v = self.policy.value(torch.as_tensor(critic_input(X)))
                for i in np.flatnonzero(due):
                    choice[i] = int(a[i])
                    self.decisions.append(dict(agent=int(i), x=X[i], c=critic_input(X)[i], mask=mask[i], prior=int(prior[i]),
                                               a=int(a[i]), logp=float(lp[i]), v=float(v[i]), step=ep.steps))
            # a target only counts as "parked" once its clot is gone; keep pursuing chosen alive clots
            keep = ~due
            choice[keep] = np.where(st.parked[keep] | (st.targets[keep] < 0), M_CLOTS, st.targets[keep])
            st.resolve(est, choice)
            st.next_decide[due] = ep.env.elapsed_s+DECIDE_S
            self.decision_count += int(due.sum())
        if self.use_tpg and st.tpg is not None:
            st.tpg.update(est, st.targets)
        return st.targets.copy()

    def hold(self, ep, est):
        h = self.st.tpg.gate(est.pos, est.active) if (self.use_tpg and self.st.tpg is not None) else np.zeros(ep.n, bool)
        self.st.hold = h
        return h

    def observe_motion(self, est, local):
        self.st.stall.update(est, np.linalg.norm(local, axis=1) > 0)


class AuctionAllocator(LearnedAllocator):
    """Classical online re-allocation (no learning)."""
    def __call__(self, ep, est):
        st = self.st; due = self.due(est)
        if due.any():
            alive = np.asarray(est.clot_alive, bool); geo = st.geo(est); prior = st.a_next(alive)
            choice = np.where(st.targets >= 0, st.targets, M_CLOTS).astype(int)
            for i in np.flatnonzero(due):
                c = prior[i]
                if st.stall.stall_s[i] > 10. and c >= 0:          # stuck: release own clot, try another
                    c = -1
                if c < 0:
                    others = {int(st.targets[p]) for p in range(ep.n) if p != i and est.active[p] and not st.stall.stalled[p]}
                    cand = [j for j in range(M_CLOTS) if alive[j] and j not in others]
                    if not cand:       # steal a clot a peer will reach later than me
                        cand = [j for j in range(M_CLOTS) if alive[j] and
                                all(geo[i, j] < geo[p, j]-2*ep.d_min for p in range(ep.n) if p != i and st.targets[p] == j)]
                    c = min(cand, key=lambda j: geo[i, j]) if cand else M_CLOTS
                choice[i] = c
            keep = ~due
            choice[keep] = np.where(st.parked[keep] | (st.targets[keep] < 0), M_CLOTS, st.targets[keep])
            st.resolve(est, choice); self.decision_count += int(due.sum()); st.next_decide[due] = ep.env.elapsed_s+DECIDE_S
        if self.use_tpg and st.tpg is not None:
            st.tpg.update(est, st.targets)
        return st.targets.copy()
