"""Published-method baselines adapted to benchmark v4 (deployable inputs only; see scripts/benchmark_lysis.py).

PacNMPC  (Gonzales, Polevoy, Kobilarov, Moore, "Multi-Agent Feedback Motion Planning using Probably Approximately
         Correct Nonlinear Model Predictive Control", IEEE CASE 2025, doi:10.1109/CASE58245.2025.11164063,
         arXiv:2501.12234). Distributed sampling NMPC: each cluster optimises a Gaussian distribution over its
         control sequence by rollouts of a stochastic model (own state noise, flow drift, neighbours' shared
         plans with noise) and minimises a PAC upper bound of the expected cost under a PAC upper bound of the
         constraint-violation probability; agents exchange their plans. Adaptation: 3-D first-order cluster
         model (speed 1 mm/s, estimated drift), lumen constraint from the pre-operative map (union of tubes),
         separation constraint d_min between clusters, progress toward a route point ahead on the pre-operative
         route; replaces the TPG and the local controller; allocation A shared. Simplifications: the
         distribution is updated by a cross-entropy step (no time-varying LQR feedback term); the PAC bounds are
         Hoeffding bounds on the sampled cost (range-normalised) and violation rate. No author code exists.
SwitchableTPG (Jiang, Lin, Li, "Speedup Techniques for Switchable Temporal Plan Graph Optimization", AAAI 2025,
         doi:10.1609/aaai.v39i22.34487; STPG of Berndt et al.). The precedence of every pair of conflicting
         path segments is switchable; online, given the measured delays, the orders of zones not yet entered
         are re-optimised to minimise the predicted sum of completion times, subject to acyclicity (a
         deadlock in the predicted execution is infeasible). IGSES is a speed-up of exactly this search; with
         N <= 3 and few zones the search here is exhaustive (exact), so the optimum IGSES would return.
         Replaces the fixed-order TPG; allocation A, pursuit and wall guard shared.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from marl.hierarchical_navigation import tube_clearance


def corridor_clearance(points, sdf, body, edges):
    """Map clearance (union of the tubes in `edges`), as hierarchical_navigation.tube_clearance."""
    idx = np.asarray(sorted(edges), int)
    start, delta, length2 = sdf.a[idx], sdf.ab[idx], sdf.l2[idx]
    rel = np.asarray(points, float)[..., None, :]-start
    f = np.clip(np.sum(rel*delta, axis=-1)/length2, 0., 1.)
    radial = np.linalg.norm(rel-f[..., None]*delta, axis=-1)
    return np.max(sdf.r0[idx]+f*(sdf.r1[idx]-sdf.r0[idx])-radial, axis=-1)-body
from marl.vessel_sdf import VesselSDF


class PacNMPC:
    def __init__(self, ep, horizon=10, samples=96, iters=3, sigma=.6, eps=.2, delta=.05,
                 wall_margin=.02, w_wall=1000., w_sep=50., w_track=8., w_goal=4., w_smooth=.2, w_ref=2., ahead_mm=2.):
        self.ep = ep; self.H, self.K, self.iters = horizon, samples, iters
        self.sigma0, self.eps, self.delta = sigma, eps, delta
        self.wall_margin, self.w_wall, self.w_sep, self.w_track, self.w_goal, self.w_smooth = (
            wall_margin, w_wall, w_sep, w_track, w_goal, w_smooth)
        self.w_ref = w_ref
        self.ahead = ahead_mm
        self.sdf = VesselSDF(ep.sensor); self.body = float(ep.ctl.body)
        n = ep.n; self.mean = np.zeros((n, horizon, 3)); self.plans = np.zeros((n, horizon, 3))
        self.drift = np.zeros((n, 3)); self.prev_cmd = np.zeros((n, 3)); self.prev_pos = None
        self.rng = np.random.default_rng(ep.seed+31)
        self.dt = float(ep.env.config.control_dt_s); self.speed = float(ep.env.config.robot_speed_mm_s)
        self.pos_sigma = float(getattr(ep.sensor.cfg, 'position_sigma_mm', .05))
        self.viol_rate = []; self.bound_fail = 0; self.calls = 0
        ends = np.asarray(ep.sensor.ends, int)
        self.seg = {(int(a), int(b)): e for e, (a, b) in enumerate(ends)}; self.seg.update({(b, a): e for (a, b), e in list(self.seg.items())})

    def _route_window(self, i):
        """Pre-operative route ahead of the cluster's route progress, resampled at 0.1 mm, with arclength."""
        ctl = self.ep.ctl; R = ctl.route[i]
        P = ctl.pts[R[int(ctl.prog[i]):]]
        if len(P) < 2:
            return P, np.zeros(len(P))
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1); arc = np.r_[0., np.cumsum(seg)]
        grid = np.arange(0., min(arc[-1], self.ahead)+1e-9, .1)
        W = np.stack([np.interp(grid, arc, P[:, k]) for k in range(3)], 1)
        return W, grid

    def __call__(self, ep, est, tgt, rule, hold):
        """rule: pursuit command (local frame); zero means dwell at the clot / no target -> stay."""
        n = ep.n; F = ep.ctl.frames(est); out = np.zeros((n, 3))
        if self.prev_pos is not None:                       # flow drift estimate: measured minus commanded
            meas_v = (est.pos-self.prev_pos)/self.dt
            # only the component orthogonal to the command is attributed to flow: a blocked cluster (stenosis,
            # wall) must not be modelled as flow opposing the command; magnitude capped at 0.3 mm/s
            r = meas_v-self.speed*self.prev_cmd
            u = self.prev_cmd/np.maximum(np.linalg.norm(self.prev_cmd, axis=1, keepdims=True), 1e-9)
            r = r-np.minimum((r*u).sum(1, keepdims=True), 0.)*u
            self.drift = .95*self.drift+.05*r
            self.drift *= np.minimum(1., .3/np.maximum(np.linalg.norm(self.drift, axis=1, keepdims=True), 1e-9))
        self.prev_pos = est.pos.copy()
        for i in range(n):
            if not est.active[i] or not np.any(rule[i]):
                self.mean[i] = 0.; self.plans[i] = est.pos[i]; self.prev_cmd[i] = 0.; continue
            window, arc = self._route_window(i); goal = window[-1]
            # lumen corridor: matched edge and its junction neighbours plus every map segment of the route ahead
            ctl = ep.ctl; R = ctl.route[i][int(ctl.prog[i]):int(ctl.prog[i])+40]
            edges = set(self.sdf.neigh[int(est.edge[i])].tolist()) | {int(est.edge[i])}
            edges |= {self.seg[(int(a), int(b))] for a, b in zip(R[:-1], R[1:]) if (int(a), int(b)) in self.seg}
            clr = lambda P: corridor_clearance(P, self.sdf, self.body, edges)
            # final approach: within 0.6 mm of the target clot the speed is optimised too (bounded actuator), so the
            # cluster can settle on the clot instead of circling it at constant speed
            near = tgt[i] >= 0 and float(np.linalg.norm(ep.env.clot_positions_mm[tgt[i]]-est.pos[i])) < .6
            if near:
                goal = ep.env.clot_positions_mm[tgt[i]]; window = goal[None]; arc = np.zeros(1)
            peers = [p for p in range(n) if p != i and est.active[p]]
            # nominal control sequence = the route-pursuit command (PAC-NMPC optimises around a nominal plan)
            u_ref = F[i].T@rule[i]
            mu = np.roll(self.mean[i], -1, 0); mu[-1] = u_ref
            if not np.any(mu):
                mu[:] = u_ref
            c0 = float(tube_clearance(est.pos[i][None], self.sdf, self.body, int(est.edge[i]))[0])
            sig = np.full((self.H, 3), float(np.clip(self.sigma0*c0/.3, .1, self.sigma0)))   # narrow lumen: explore less
            best = None
            for _ in range(self.iters):
                U = mu[None]+sig[None]*self.rng.normal(size=(self.K, self.H, 3))
                U /= np.maximum(np.linalg.norm(U, axis=-1, keepdims=True), 1.)
                disp = np.cumsum(self.dt*(self.speed*U+self.drift[i]), axis=1)                    # [K, H, 3]
                # cost on the nominal rollout (estimated start); the violation probability for the PAC bound on
                # rollouts with sampled state / neighbour uncertainty (independent draws per sample)
                X = est.pos[i][None, None]+disp
                Xn = X+self.rng.normal(0., self.pos_sigma, (self.K, 1, 3))
                clear = clr(X)
                wall_v = np.maximum(self.wall_margin-clear, 0.)
                viol_nom = (clear < 0.).any(1)
                viol = (clr(Xn) < 0.).any(1)
                sep = np.zeros(self.K)
                for p in peers:
                    d = np.linalg.norm(X-self.plans[p][None], axis=-1)
                    sep += np.maximum(ep.d_min-d, 0.).sum(1); viol_nom |= (d < ep.d_min).any(1)
                    Yn = self.plans[p][None]+self.rng.normal(0., self.pos_sigma, (self.K, 1, 3))
                    viol |= (np.linalg.norm(Xn-Yn, axis=-1) < ep.d_min).any(1)
                dd = np.linalg.norm(X[:, :, None]-window[None, None], axis=-1)            # [K, H, W]
                dw = dd.min(2)
                progress = arc[dd[:, -1].argmin(1)]                                       # route arclength reached
                term = np.linalg.norm(X[:, -1]-goal, axis=1) if near else -progress
                cost = (self.w_wall*(wall_v**2).sum(1)+self.w_sep*sep+self.w_track*dw.mean(1)
                        + self.w_ref*((U-u_ref)**2).sum(-1).mean(1)
                        + self.w_goal*term
                        + self.w_smooth*np.linalg.norm(np.diff(np.concatenate([self.prev_cmd[i][None, None].repeat(self.K, 0), U], 1), axis=1), axis=-1).sum(1))
                # PAC (Hoeffding) bounds of the candidate distribution: violation rate and range-normalised cost
                hoeff = math.sqrt(math.log(1/self.delta)/(2*self.K))
                v_ub = float(viol.mean())+hoeff
                c_lo, c_hi = float(cost.min()), float(cost.max())
                c_ub = float(cost.mean())+(c_hi-c_lo)*hoeff
                cand = (v_ub > self.eps, c_ub, mu.copy())
                if best is None or cand[:2] < best[:2]:
                    best = cand
                elite = np.argsort(cost+1e3*viol_nom)[:max(8, self.K//8)]
                mu = U[elite].mean(0); sig = np.maximum(U[elite].std(0), .05)
                if not near:
                    mu /= np.maximum(np.linalg.norm(mu, axis=-1, keepdims=True), 1e-9)   # cruise at full speed
            self.bound_fail += int(best[0]); self.calls += 1
            self.mean[i] = best[2]
            cmd = best[2][0]/max(np.linalg.norm(best[2][0]), 1. if near else 1e-9)
            self.plans[i] = est.pos[i]+np.cumsum(self.dt*(self.speed*best[2]+self.drift[i]), axis=0)
            self.prev_cmd[i] = cmd
            out[i] = F[i]@cmd
        return out


class SwitchableTPG:
    """Wraps a built TPGCoordinator: periodically re-optimises the order of zones neither cluster has entered."""
    def __init__(self, coord, period_s=2., max_free=10):
        self.c = coord; self.period = period_s; self.max_free = max_free; self.next_t = 0.
        self.switches = 0; self.solves = 0; self.infeasible = 0

    def _simulate(self, order, prog, dt=.25, t_max=600.):
        c = self.c; n = c.n; s = np.array(prog, float)*c.step; ends = np.array([len(p)-1 for p in c.paths])*c.step
        dwell_left = [{cs: c.dwell for cs in c.clot_s[i] if cs > s[i]+1e-9} for i in range(n)]
        t, finish = 0., np.full(n, np.nan)
        while t < t_max:
            moved = False
            for i in range(n):
                if not np.isnan(finish[i]):
                    continue
                if s[i] >= ends[i]-1e-9:
                    finish[i] = t; continue
                due = [cs for cs in dwell_left[i] if cs <= s[i]+1e-9]
                if due:
                    dwell_left[i][due[0]] -= dt
                    if dwell_left[i][due[0]] <= 0:
                        del dwell_left[i][due[0]]
                    moved = True; continue
                blocked = False
                for zi, z in enumerate(c.zones):
                    if i not in (z['i'], z['j']):
                        continue
                    first = order.get(zi)
                    if first is None or first == i:
                        continue
                    a0 = (z['s0'] if z['i'] == i else z['u0'])*c.step
                    f1 = (z['s1'] if z['i'] == first else z['u1'])*c.step
                    if s[first] <= f1 and a0-c.stop_margin-1. <= s[i] < a0:
                        blocked = True; break
                if not blocked:
                    s[i] = min(s[i]+c.speed*dt, ends[i]); moved = True
            if np.all(~np.isnan(finish)):
                return float(finish.sum())
            if not moved:
                return np.inf                       # predicted deadlock: infeasible ordering
            t += dt
        return np.inf

    def update(self, t):
        c = self.c
        if t < self.next_t or not c.zones:
            return
        self.next_t = t+self.period
        prog = c.progress.copy(); free = []
        for zi, z in enumerate(c.zones):
            if zi not in c.order:
                continue
            si, sj = prog[z['i']], prog[z['j']]
            if si < z['s0'] and sj < z['u0']:       # neither cluster has entered: precedence still switchable
                free.append(zi)
        if not free:
            return
        free = free[:self.max_free]
        base = dict(c.order); best = (self._simulate(base, prog), base)
        for flips in itertools.product((0, 1), repeat=len(free)):
            if not any(flips):
                continue
            o = dict(base)
            for zi, f in zip(free, flips):
                if f:
                    z = c.zones[zi]; o[zi] = z['j'] if base[zi] == z['i'] else z['i']
            cost = self._simulate(o, prog)
            if cost < best[0]-1e-9:
                best = (cost, o)
        self.solves += 1
        if not np.isfinite(best[0]):
            self.infeasible += 1
        if best[1] != c.order:
            self.switches += 1; c.order = best[1]
