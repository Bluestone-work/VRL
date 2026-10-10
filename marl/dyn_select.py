"""EXP0090: prediction-assisted command selection with an uncertainty fallback (deployable).

Per cluster and control step a small candidate set of constant commands is scored with a short-horizon motion
predictor (learned ensemble marl.dyn_predictor.Ensemble, or the non-learned OnlineAffine baseline):
  candidates  the deployable rule command (SwitchSettle: Fixed Settle when the camera frame age > 0.15 s, else
              Adaptive Settle), the zero-residual route-pursuit command with settling (= nav_tf_v3 learner-off),
              stop, and pursuit directions toward the carrot / re-aimed by +-25 deg along the route normal and
              binormal / toward the target clot (if within 1.5 mm), each at speeds {0.25, 0.5, 0.75, 1.0}.
              |command| <= 1 always (actuator limit unchanged).
  cost        - W_PROG * predicted route progress over 1.2 s (mm)
              - W_APPR * predicted approach to the clot (mm) - W_DWELL * fraction of horizons within contact range
              + W_WALL * sum_h relu(WALL_MARGIN - (map clearance - k sigma)) / WALL_MARGIN
              + W_SPACE * sum_h relu(d_min + 0.3 - peer distance) / d_min        (peers: constant-velocity forecast)
              + W_SIGMA * predicted std at 1.2 s (mm)
  The executed command passes through the same WallGuard recovery projection, TPG hold and spacing shield as every
  method. Fallback (variant E): the rule command is executed when the ensemble's epistemic std at 1.2 s exceeds
  FALLBACK_STD (fixed from training-split data before evaluation). Simulator truth is never read here.
"""
from __future__ import annotations

import time

import numpy as np

from marl.dyn_predictor import H_FUT, HORIZONS

W_PROG, W_APPR, W_DWELL, W_WALL, W_SPACE, W_SIGMA = 1., 2., 1., 2., 2., .5
WALL_MARGIN, WALL_K, CONTACT_MM = .05, 1., .12
# v2 (2026-10-11 01:0x, before any registered evaluation; smoke test on 18 tuning episodes): with these risk terms the
# selector chose 'stop' in 50-90 % of live steps (moving candidates looked risky in narrow lumens because the
# predicted sigma was added to the wall term). Rule-anchored selection: the rule command is kept unless another
# candidate improves the predicted cost by DELTA; wall term on mean prediction + WALL_K sigma (tuned set W0-W2).
DEFAULT = dict(prog=1., appr=2., dwell=1., wall=2., space=2., sigma=.2, delta=.2, wall_k=.5, wall_margin=.03, idle=.3)
SPEEDS = (.25, .5, .75, 1.)
TILT = np.tan(np.radians(25.))


class RuleSettle:
    """SwitchSettle settling logic applied to an already wall-guarded local command (no second WallGuard call)."""
    def __init__(self, ep, radius=.3, tau=.15):
        n = ep.n; self.radius, self.tau = radius, tau
        self.hist = [[] for _ in range(n)]; self.release_until = np.zeros(n); self.resp = np.ones(n)
        self.prev_cmd = np.zeros((n, 3))

    def __call__(self, ep, est, tgt, guarded):
        out = guarded.copy(); t = ep.env.elapsed_s; F = ep.ctl.frames(est)
        fixed = t-float(getattr(est, 'frame_time_s', t)) > self.tau
        for i, c in enumerate(tgt):
            w = F[i].T@self.prev_cmd[i]; sp = float(np.linalg.norm(w))
            if sp > .5:
                self.resp[i] = .95*self.resp[i]+.05*float(np.clip(est.vel[i]@w/sp**2, 0., 2.))
            if c < 0:
                self.hist[i].clear(); continue
            d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
            self.hist[i].append(d); self.hist[i] = self.hist[i][-10:]
            if fixed:
                if d < self.radius:
                    out[i] *= d/self.radius
                continue
            r = self.radius*float(np.clip(self.resp[i], .5, 1.5))
            if d < r and t >= self.release_until[i]:
                if len(self.hist[i]) == 10 and self.hist[i][0]-d < .03 and d > .12:
                    self.release_until[i] = t+2.
                else:
                    out[i] *= d/r
        return out

    def note(self, executed_local):
        self.prev_cmd = executed_local.copy()


def candidates(ep, est, tgt, i, rule_world, zero_world):
    """[K, 3] world commands and labels."""
    ctl = ep.ctl; pos = est.pos[i]
    d0 = ctl.carrot[i]-pos; d0 = d0/max(np.linalg.norm(d0), 1e-9)
    st = int(np.argmin(np.linalg.norm(ctl.pts-ctl.carrot[i], axis=1)))
    nrm, bin_ = ep.env.tree.normals[st].astype(float), ep.env.tree.binormals[st].astype(float)
    dirs = [d0]+[(d0+s*TILT*v)/np.linalg.norm(d0+s*TILT*v) for v in (nrm, bin_) for s in (1., -1.)]
    if tgt[i] >= 0:
        dc = ep.env.clot_positions_mm[tgt[i]]-pos; L = float(np.linalg.norm(dc))
        if 1e-6 < L < 1.5:
            dirs.append(dc/L)
    C = [rule_world, zero_world, np.zeros(3)]+[sp*d for d in dirs for sp in SPEEDS]
    return np.asarray(C, float)


def route_arclength(ep, i, X):
    """Arclength (mm) along cluster i's current route of the nearest route point to each world point in X."""
    ctl = ep.ctl; R = ctl.route[i]
    if R is None:
        return np.zeros(len(X))
    k0 = int(ctl.prog[i]); P = ctl.pts[R[max(k0-2, 0):k0+60]]
    if len(P) < 2:
        return np.zeros(len(X))
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    d = np.linalg.norm(X[:, None]-P[None], axis=-1)
    return s[np.argmin(d, axis=1)]


class PredictiveSelector:
    def __init__(self, ep, predictor='learned', ensemble=None, fallback_std=None, weights=None, belief_candidates=False):
        from scripts.benchmark_lysis import WallGuard
        from marl.dyn_predictor import DynFeatures, OnlineAffine
        from marl.vessel_sdf import VesselSDF
        self.ep, self.kind, self.ens, self.fallback_std = ep, predictor, ensemble, fallback_std
        self.belief_candidates = bool(belief_candidates)
        self.guard = WallGuard(ep); self.rule = RuleSettle(ep); self.feat = DynFeatures(ep)
        self.online = OnlineAffine(ep) if predictor == 'online' else None
        self.sdf = VesselSDF(ep.sensor); self.body = float(ep.env.config.robot_radius_mm)
        self.w = dict(DEFAULT)
        if weights:
            self.w.update(weights)
        self.stats = dict(steps=0, live=0, fallback=0, rule_chosen=0, zero_chosen=0, stop_chosen=0, grid_chosen=0,
                          infer_s=0., epi_std=[])

    def _clearance(self, X, edge, i=None):
        """Map clearance over the lumen corridor: matched edge, its junction neighbours and every map segment of
        the route ahead (v3 fix: the one-hop neighbourhood alone reported false wall contact 1 mm ahead)."""
        from marl.lysis_baselines import corridor_clearance
        edges = set(self.sdf.neigh[int(edge)].tolist()) | {int(edge)}
        ctl = self.ep.ctl
        if i is not None and ctl.route[i] is not None:
            if not hasattr(self, '_seg'):
                ends = np.asarray(self.ep.sensor.ends, int)
                self._seg = {(int(a), int(b)): e for e, (a, b) in enumerate(ends)}
                self._seg.update({(b, a): e for (a, b), e in list(self._seg.items())})
            R = ctl.route[i][max(int(ctl.prog[i])-2, 0):int(ctl.prog[i])+40]
            edges |= {self._seg[(int(a), int(b))] for a, b in zip(R[:-1], R[1:]) if (int(a), int(b)) in self._seg}
        return corridor_clearance(X, self.sdf, self.body, edges)

    def __call__(self, ep, est, tgt, rule, hold):
        t0 = time.perf_counter(); n = ep.n
        self.feat.push(ep, est, hold)
        if self.online is not None:
            self.online.update(ep, est)
        guarded = self.guard(ep, est, tgt, rule, hold)                  # once per step (stall re-planning state)
        rule_local = self.rule(ep, est, tgt, guarded)
        zero_local = self._settle_fixed(ep, est, tgt, guarded)     # nav_tf_v3 learner-off speed prior (fixed settling)
        out = rule_local.copy(); F = ep.ctl.frames(est); self.stats['steps'] += 1
        live = (np.asarray(tgt) >= 0) & est.active & ~np.asarray(hold, bool)
        peers_now = est.pos; peers_vel = est.vel
        for i in range(n):
            if not live[i]:
                continue
            self.stats['live'] += 1
            C = candidates(ep, est, tgt, i, F[i].T@rule_local[i], F[i].T@zero_local[i])
            if self.belief_candidates:
                C = np.vstack([C, self._inversion(ep, est, tgt, i, F[i], C[0])])
            plans = np.repeat(C[:, None, :], H_FUT, 1)
            if self.kind == 'online':
                mu, var = self.online.predict(ep, est, i, F[i], plans); epi = np.zeros_like(var)
            else:
                X, M = self.feat.history(ep, est, i, F[i]); ctx = self.feat.context(ep, est, tgt, i, F[i])
                K = len(C)
                mu, var, epi = self.ens.predict(np.repeat(X[None], K, 0), np.repeat(M[None], K, 0),
                                                np.repeat(ctx[None], K, 0),
                                                np.einsum('ab,khb->kha', F[i], plans).astype(np.float32))
            Pw = est.pos[i][None, None]+np.einsum('ab,kha->khb', F[i], mu)          # [K, 5, 3] world
            sig = np.sqrt(var.sum(-1)+epi.sum(-1))                                    # [K, 5]
            K = len(C)
            prog = route_arclength(ep, i, Pw[:, -1])-route_arclength(ep, i, Pw[:, 0])
            cost = -self.w['prog']*prog
            if tgt[i] >= 0:
                q = ep.env.clot_positions_mm[tgt[i]]; dq = np.linalg.norm(Pw-q, axis=-1)
                if float(np.linalg.norm(q-est.pos[i])) < 1.5:
                    cost -= self.w['appr']*(dq[:, 0]-dq[:, -2:].min(1))+self.w['dwell']*(dq[:, 2:] < CONTACT_MM).mean(1)
            clr = self._clearance(Pw.reshape(-1, 3), est.edge[i], i).reshape(K, len(HORIZONS))
            m_ = self.w['wall_margin']
            cost += self.w['wall']*np.maximum(m_-(clr-self.w['wall_k']*sig), 0.).sum(1)/m_
            near_clot = tgt[i] >= 0 and float(np.linalg.norm(ep.env.clot_positions_mm[tgt[i]]-est.pos[i])) < .3
            if not near_clot:                                  # idling away from the clot costs time
                cost += self.w['idle']*(np.linalg.norm(C, axis=1) < .3)
            for j in range(n):
                if j == i or not est.active[j]:
                    continue
                Q = peers_now[j][None]+np.asarray(HORIZONS, float)[:, None]*ep.env.config.control_dt_s*peers_vel[j][None]
                dist = np.linalg.norm(Pw-Q[None], axis=-1)
                cost += self.w['space']*np.maximum(ep.d_min+.3-dist, 0.).sum(1)/ep.d_min
            cost += self.w['sigma']*sig[:, -1]
            k = int(np.argmin(cost))
            if cost[0]-cost[k] < self.w['delta']:            # rule-anchored: keep the rule unless clearly better
                k = 0
            e = float(np.sqrt(epi[k, -1].sum()))
            self.stats['epi_std'].append(e)
            if self.fallback_std is not None and e > self.fallback_std:
                self.stats['fallback'] += 1; continue                                # out[i] = rule command
            key = 'rule_chosen' if k == 0 else 'zero_chosen' if k == 1 else 'stop_chosen' if k == 2 else 'grid_chosen'
            self.stats[key] += 1
            if k == 0:
                continue
            if k == 1:
                out[i] = zero_local[i]; continue
            cand_local = np.zeros((n, 3)); cand_local[i] = F[i]@C[k]
            out[i] = self.guard.exec.act(ep.env.elapsed_s, est, ep.ctl, cand_local, hold)[i]
        self.rule.note(out)
        self.stats['infer_s'] += time.perf_counter()-t0
        return out

    def _inversion(self, ep, est, tgt, i, F, rule_world):
        """Round-2 candidates: invert the LEARNED flow / response belief (aux head of the predictor, conditioned on the
        rule plan): c = (v_des - u_flow_hat) / g_hat for desired velocities toward the carrot (1, 0.6 mm/s) and, near
        the clot, toward the clot (2 d mm/s), and toward the slower near-wall region; clipped to |c| <= 1."""
        if self.kind == 'online':
            g = float(self.online.om.gain[i]); u = self.online.om.drift[i]/self.online.speed
        else:
            X, M = self.feat.history(ep, est, i, F); ctx = self.feat.context(ep, est, tgt, i, F)
            plan = np.repeat((F@rule_world)[None, None], H_FUT, 1).astype(np.float32)
            self.ens.predict(X[None], M[None], ctx[None], plan); aux = self.ens.last_aux[0]
            u = F.T@aux[:3]; g = float(np.clip(aux[3], .3, 2.))
        ctl = ep.ctl; pos = est.pos[i]; d0 = ctl.carrot[i]-pos; d0 = d0/max(np.linalg.norm(d0), 1e-9)
        want = [d0, .6*d0]
        if tgt[i] >= 0:
            dc = ep.env.clot_positions_mm[tgt[i]]-pos; L = float(np.linalg.norm(dc))
            if L < 1.:
                want.append(dc*min(2., 1./max(L, 1e-6)))
        ax, r, rad = ep.sensor.map_coordinates(est, i); off = pos-ax; ro = float(np.linalg.norm(off))
        if ro > 1e-6:
            want.append(d0+1.5*(.85*max(r-self.body, 0.)-ro)*off/ro)
        out = []
        for v in want:
            c = (v-u)/g; m = float(np.linalg.norm(c))
            out.append(c/max(m, 1.))
        return np.asarray(out)

    def _settle_fixed(self, ep, est, tgt, guarded):
        out = guarded.copy()
        for i, c in enumerate(tgt):
            if c >= 0:
                d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
                if d < self.rule.radius:
                    out[i] *= d/self.rule.radius
        return out

    def summary(self):
        s = dict(self.stats); e = np.asarray(s.pop('epi_std')) if s['epi_std'] else np.zeros(1)
        L = max(s['live'], 1)
        return dict(sel_live_steps=s['live'], sel_fallback_frac=s['fallback']/L, sel_rule_frac=s['rule_chosen']/L,
                    sel_zero_frac=s['zero_chosen']/L, sel_stop_frac=s['stop_chosen']/L, sel_grid_frac=s['grid_chosen']/L,
                    sel_learned_takeover_frac=(s['stop_chosen']+s['grid_chosen'])/L,
                    infer_ms_per_step=1000*s['infer_s']/max(s['steps'], 1), epi_std_p50=float(np.median(e)),
                    epi_std_p90=float(np.percentile(e, 90)))


class Behaviour:
    """Data-collection behaviour (training anatomies only): per cluster, every U{6..12} steps either follow the rule
    command or hold a random candidate constant (world frame), p = 0.5 each; records the intended world command."""
    def __init__(self, ep, rng):
        from scripts.benchmark_lysis import WallGuard
        self.guard = WallGuard(ep); self.rule = RuleSettle(ep); self.rng = rng; n = ep.n
        self.left = np.zeros(n, int); self.mode = np.zeros(n, int); self.cmd = np.zeros((n, 3)); self.intended = np.zeros((n, 3))

    def __call__(self, ep, est, tgt, rule, hold):
        guarded = self.guard(ep, est, tgt, rule, hold); rl = self.rule(ep, est, tgt, guarded); out = rl.copy()
        F = ep.ctl.frames(est)
        for i in range(ep.n):
            self.intended[i] = F[i].T@rl[i]
            if tgt[i] < 0 or not est.active[i]:
                continue
            if self.left[i] <= 0:
                self.left[i] = int(self.rng.integers(6, 13)); self.mode[i] = int(self.rng.random() < .5)
                if self.mode[i]:
                    C = candidates(ep, est, tgt, i, np.zeros(3), np.zeros(3))[2:]
                    self.cmd[i] = C[self.rng.integers(len(C))]
            self.left[i] -= 1
            if self.mode[i]:
                cl = np.zeros((ep.n, 3)); cl[i] = F[i]@self.cmd[i]
                out[i] = self.guard.exec.act(ep.env.elapsed_s, est, ep.ctl, cl, hold)[i]
                self.intended[i] = self.cmd[i]
        self.rule.note(out)
        return out
