"""Route-frame RL navigation (ours, v5): PPO with a causal Transformer over the deployable observation history.

Motivation (2026-10-08 v4/v5 results): every learned *residual* on a near-optimal classical command degraded it,
while a residual learned from the weak guard-only prior discovered dwell control and solved the high-flow anatomy
(coronary_rca 10/10) that the hand-written settling rule fails. In v5 the dynamics are patient-specific and
time-varying (marl.physio_variation): pulsatile flow, unknown cluster response, near-wall drag/adhesion, unknown
lysis rate. Geometry stays known (pre-operative CTA). The navigation decisions that depend on the unknown dynamics are
*how fast* to move along the route (wait out systolic flow, settle on the clot, slow in narrow lumens) and *where in
the cross-section* to travel (away from the wall, toward the faster or slower part of the flow profile). The policy
therefore acts in the frame of the pre-operative route:
  a0 -> speed  s = clip(1 + a0, 0, 1)                        (a0 = 0: full speed, -1: stop)
  a1, a2 -> lateral aim point  carrot + 0.7 (r_map - r_body) (a1 n + a2 b)   (route normal / binormal)
  command = s * unit(aim - position)
It does not add a rule command: at a = 0 it reproduces plain pursuit toward the carrot (no settling, no wall
term); speed modulation and lateral placement are learned. The shared map wall guard and stall re-planning
(scripts.benchmark_lysis.WallGuard) and the spacing shield are applied to every method's output, ours included.
Observation token (world frame, deployable): carrot direction, route tangents now / +1 mm / +2 mm, map radius here
and at the carrot, radial offset (ratio, vector, clearance), estimated velocity and its route component, previous
action, route / Euclidean distance to the target clot, junction flag, TPG hold, time left, two peers.
Reward (truth, training only), v2: +1 route progress (mm, clipped 0.2/step) +50 team removal -3 wall-contact s
-10 lost -0.5 spacing -0.02 |a - a_prev|^2 -0.03 per live step; gamma 0.995 (20 s), GAE lambda 0.95.
Control-prior regularisation (v2; cf. Rana et al., "Bayesian controller fusion", IJRR 2023, cited by Turbo): the
policy mean is pulled toward the strongest adaptive classical action in the same action space (speed of the adaptive
settling rule, no lateral offset) with weight beta(t) = 1 -> 0.05 (cosine); PPO is free to deviate where it pays.
v1 (no prior term, wall -10, gamma 0.98, no time cost) collapsed to stopping within 30 min (speed 0.1, success
0.7 -> 0.0): with a 5-s horizon, removal was out of reach while every moving step risked wall penalty under v5.
Domain randomisation: v5 strength s ~ 0 (p 0.2) or U[0, 1.25]; image sensing latency U{1,2}. Train anatomies only,
N ~ U{1,2,3}, seeds 1818000000+.
usage: train_lysis_nav.py --out DIR --minutes 120 --workers 12
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_lysis_local import gae as _gae, route_remaining


def gae(buf, boot):
    return _gae(buf, boot, gamma=.995, lam=.95)


def weighted_prior_loss(action_mean, prior, learn_weight):
    """Per-sample prior regularization; avoids [B] x [B,1] broadcasting."""
    error=((action_mean-prior)**2).sum(-1)
    return (error*learn_weight).sum()/learn_weight.sum().clamp(min=1)

ROLLOUT = 256
ECG_DIM = 2      # v5-ECG: sin / cos of the cardiac phase (ECG is routinely monitored during endovascular procedures)
# training anatomies with the most v4/v5 failures of the classical stack (train split only; held-out ones such as
# coronary_rca / basilar_vertebral are never trained on)
HARD = {'mca_m1_lvo', 'pulmonary_saddle', 'ica_terminus_t', 'coronary_lm_bifurcation', 'popliteal_calf_dvt'}
DIM = 39


def _ahead_tangent(P, k, dist):
    acc, j = 0., k
    while j+1 < len(P) and acc < dist:
        acc += float(np.linalg.norm(P[j+1]-P[j])); j += 1
    if j+1 < len(P):
        d = P[j+1]-P[j]
    elif j > 0:
        d = P[j]-P[j-1]
    else:
        return np.zeros(3)
    return d/max(np.linalg.norm(d), 1e-9)


def cardiac_phase(ep):
    """ECG-derived phase (deployable). Without physiological variation a nominal 75 bpm clock is used."""
    v = ep.var; t = ep.env.elapsed_s
    return (2*np.pi*v.hr/60.*t+v.phase0) if v is not None else 2*np.pi*75./60.*t


def token(ep, est, tgt, hold, prev, ecg=False, observable_history=False):
    env, ctl, n = ep.env, ep.ctl, ep.n
    spd = env.config.robot_speed_mm_s; body = float(env.config.robot_radius_mm)
    T = np.zeros((n, DIM+(ECG_DIM if ecg else 0)+int(observable_history)), np.float32)
    if observable_history:
        if not hasattr(est, 'frame_time_s'):
            raise ValueError('Observable-history policy requires camera acquisition timestamps')
        T[:, -1] = max(0., float(env.elapsed_s)-est.frame_time_s)  # age in seconds
    if ecg:
        ph = cardiac_phase(ep); T[:, DIM] = np.sin(ph); T[:, DIM+1] = np.cos(ph)
    for i in range(n):
        if not est.active[i] or tgt[i] < 0 or ctl.route[i] is None:
            continue
        o = T[i]; R = ctl.route[i]; P = ctl.pts[R]; k = int(ctl.prog[i])
        c = ctl.carrot[i]-est.pos[i]; o[0:3] = c/max(np.linalg.norm(c), 1e-9)
        o[3:6] = _ahead_tangent(P, k, 0.); o[6:9] = _ahead_tangent(P, k, 1.); o[9:12] = _ahead_tangent(P, k, 2.)
        ax, r, rad = ep.sensor.map_coordinates(est, i)
        rc = float(ep.sensor.healthy[int(np.argmin(np.linalg.norm(ctl.pts-ctl.carrot[i], axis=1)))])
        o[12] = r/(r+.5); o[13] = rc/(rc+.5); o[14] = rad/max(r, 1e-9)
        o[15:18] = (est.pos[i]-ax)/max(r, 1e-9); o[18] = np.clip((r-rad-body)/.5, -1, 3)
        o[19:22] = est.vel[i]/spd; o[22] = float(est.vel[i]@o[3:6])/spd
        o[23:26] = prev[i]
        o[26] = min(route_remaining(ctl, i, est.pos[i])/10., 5.)
        o[27] = min(float(np.linalg.norm(env.clot_positions_mm[tgt[i]]-est.pos[i]))/2., 3.)
        o[28] = float(np.any(ctl.deg[R[max(k-3, 0):k+6]] >= 3)); o[29] = float(hold[i])
        o[30] = max(0., 1-env.elapsed_s/env.config.episode_duration_s)
        vis = np.flatnonzero(est.peers_vis[i]); vis = vis[np.argsort(np.linalg.norm(est.peers_rel[i, vis], axis=1))][:2]
        for j, q in enumerate(vis):
            s = 31+4*j; o[s:s+3] = est.peers_rel[i, q]/6.; o[s+3] = 1.
    return T


def command_legacy(ep, est, a, live, s_prior=None):
    """Route-frame action -> local-frame command for ep.step (world -> controller frame).
    v3: speed = clip(s_prior + 0.5 a0, 0, 1) when a prior speed is given (adaptive classical settling speed)."""
    ctl = ep.ctl; body = float(ep.env.config.robot_radius_mm); F = ctl.frames(est)
    a = np.clip(a, -1, 1); out = np.zeros((ep.n, 3))
    for i in range(ep.n):
        if not live[i]:
            continue
        st = int(np.argmin(np.linalg.norm(ctl.pts-ctl.carrot[i], axis=1)))
        nrm, bin_ = ep.env.tree.normals[st].astype(float), ep.env.tree.binormals[st].astype(float)
        lat = .7*max(float(ep.sensor.healthy[st])-body, 0.)
        aim = ctl.carrot[i]+lat*(a[i, 1]*nrm+a[i, 2]*bin_)
        d = aim-est.pos[i]; d = d/max(np.linalg.norm(d), 1e-9)
        sp = float(np.clip(1+a[i, 0], 0, 1)) if s_prior is None else float(np.clip(s_prior[i]+.5*a[i, 0], 0, 1))
        out[i] = F[i]@(sp*d)
    return out


def speed_from_prior(action0, prior, residual_scale=1.0):
    return float(np.clip(prior + residual_scale * float(np.clip(action0, -1, 1)), 0., 1.))


def lateral_scale_for_targets(ep, est, tgt, cfg):
    scale = float(cfg.get('lateral_residual_scale', 1.))
    radius = cfg.get('lateral_near_radius')
    if radius is None:
        return np.full(ep.n, scale)
    # Registered target positions and image estimates only; no physical truth.
    return np.array([scale if t >= 0 and np.linalg.norm(ep.env.clot_positions_mm[t]-est.pos[i]) < radius
                     else 1. for i, t in enumerate(tgt)])


def command(ep, est, a, live, s_prior=None, prior_residual_scale=1.0, lateral_residual_scale=1.0):
    """Action mapping with full legal residual authority around a speed prior.

    ``command_legacy`` is retained for the mechanism ablation.  With the new
    mapping a near-target prior of zero can still produce any speed in [0,1],
    including a non-zero flow-compensation command.
    """
    ctl = ep.ctl; F = ctl.frames(est); a = np.clip(a, -1, 1); out = np.zeros((ep.n, 3))
    lateral_scales = np.broadcast_to(lateral_residual_scale, (ep.n,))
    for i in range(ep.n):
        if not live[i]: continue
        st = int(np.argmin(np.linalg.norm(ctl.pts-ctl.carrot[i], axis=1)))
        nrm, bin_ = ep.env.tree.normals[st].astype(float), ep.env.tree.binormals[st].astype(float)
        lat = .7*max(float(ep.sensor.healthy[st])-float(ep.env.config.robot_radius_mm), 0.)
        aim = ctl.carrot[i]+lateral_scales[i]*lat*(a[i,1]*nrm+a[i,2]*bin_)
        d = aim-est.pos[i]; d /= max(np.linalg.norm(d), 1e-9)
        sp = float(np.clip(1+a[i, 0], 0, 1)) if s_prior is None else speed_from_prior(a[i, 0], s_prior[i], prior_residual_scale)
        out[i] = F[i]@(sp*d)
    return out


class PriorSpeed:
    """Speed of the adaptive classical settling rule (scripts.benchmark_lysis.AdaptiveSettleGuard), as a route-frame
    action prior a = (speed - 1, 0, 0). Deployable inputs only."""
    def __init__(self, n, radius=.3, adaptive=False):
        self.radius = radius; self.hist = [[] for _ in range(n)]; self.release = np.zeros(n); self.resp = np.ones(n)
        self.adaptive = adaptive   # False: plain settling (classical_settle); True: AdaptiveSettleGuard rule

    def __call__(self, ep, est, tgt, prev_world):
        t = ep.env.elapsed_s; out = np.zeros((ep.n, 3))
        for i, c in enumerate(tgt):
            w = prev_world[i]; sp = float(np.linalg.norm(w))
            if sp > .5:
                self.resp[i] = .95*self.resp[i]+.05*float(np.clip(est.vel[i]@w/sp**2, 0., 2.))
            if c < 0:
                self.hist[i].clear(); continue
            d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
            r = self.radius*float(np.clip(self.resp[i], .5, 1.5)) if self.adaptive else self.radius
            self.hist[i].append(d); self.hist[i] = self.hist[i][-10:]; speed = 1.
            if d < r and t >= self.release[i]:
                if self.adaptive and len(self.hist[i]) == 10 and self.hist[i][0]-d < .03 and d > .12:
                    self.release[i] = t+2.
                else:
                    speed = d/r
            out[i, 0] = speed-1.
        return out


class History:
    def __init__(self, n, window, dim=DIM):
        self.seq = np.zeros((n, window, dim), np.float32); self.mask = np.ones((n, window), bool)

    def push(self, T):
        self.seq = np.roll(self.seq, -1, 1); self.mask = np.roll(self.mask, -1, 1)
        self.seq[:, -1] = T; self.mask[:, -1] = False
        return self.seq.copy(), self.mask.copy()


def make_policy(cfg):
    dim = DIM+(ECG_DIM if cfg.get('ecg') else 0)+int(cfg.get('observable_history', False))
    if cfg.get('matrix_arm'):
        from marl.lysis_matrix import MatrixPolicy
        return MatrixPolicy(cfg['matrix_arm'], arch=cfg['arch'], layers=cfg['layers'], window=cfg['window'], dim=dim)
    if cfg.get('abcd'):
        from marl.lysis_abcd import FlowPolicy
        return FlowPolicy(history=cfg['abcd'] in ('C', 'D'), arch=cfg['arch'], layers=cfg['layers'], window=cfg['window'], dim=dim)
    from marl.obstacle_control import TemporalPolicy
    return TemporalPolicy(cfg['arch'], layers=cfg['layers'], window=cfg['window'], dim=dim)


class NavController:
    """Evaluation wrapper (deterministic mean action) + shared wall guard; `low` for benchmark_lysis.rollout."""
    def __init__(self, ep, ckpt, zero_residual=False, diagnostics=False):
        """zero_residual=True: learner-off control (EXP0090) - identical history, prior, action mapping, WallGuard,
        TPG and shield, with the network output replaced by a = 0. diagnostics=True records per step the raw
        policy action, prior speed, mapped command, WallGuard output and hold flags (shield output is read from
        ep.prev_local after the step by the evaluation loop)."""
        from scripts.benchmark_lysis import WallGuard
        torch.set_num_threads(1)
        self.zero_residual, self.diagnostics = bool(zero_residual), bool(diagnostics); self.diag = []
        c = torch.load(ckpt, map_location='cpu', weights_only=False); self.cfg = c['cfg']
        self.net = make_policy(self.cfg); self.net.load_state_dict(c['state']); self.net.eval()
        self.ecg = bool(self.cfg.get('ecg'))
        self.observable_history = bool(self.cfg.get('observable_history'))
        self.hist = History(ep.n, self.cfg['window'], DIM+(ECG_DIM if self.ecg else 0)+int(self.observable_history)); self.prev = np.zeros((ep.n, 3)); self.guard = WallGuard(ep)
        self.prior = PriorSpeed(ep.n) if self.cfg.get('speed_prior') else None; self.prev_world = np.zeros((ep.n, 3))

    def __call__(self, ep, est, tgt, rule, hold):
        if self.observable_history:
            self.prev = ep.sent_world.copy()
            self.prev_world = ep.sent_world.copy()
        elif self.cfg.get('abcd'):
            # Match the training token: last safety-filtered local command.
            self.prev = ep.prev_local.copy()
        seq, mask = self.hist.push(token(ep, est, tgt, hold, self.prev, self.ecg, self.observable_history))
        with torch.no_grad():
            # Use the same actor path as training.  In particular, MatrixPolicy
            # applies its uncertainty gate in dist(); calling pi(backbone)
            # directly would silently disable that gate at evaluation time.
            a = self.net.dist(torch.as_tensor(seq), torch.as_tensor(mask))[0].loc.numpy().astype(np.float64)
        live = (np.asarray(tgt) >= 0) & est.active & ~hold
        a_raw = np.clip(a, -1, 1)
        if self.zero_residual:
            a = np.zeros_like(a_raw)
        a = np.clip(a, -1, 1); a[~live] = 0.; self.prev = a.copy()
        sp = 1.+self.prior(ep, est, tgt, self.prev_world)[:, 0] if self.prior is not None else None
        mapped = command(ep, est, a, live, sp, self.cfg.get('prior_residual_scale', .5), lateral_scale_for_targets(ep, est, tgt, self.cfg))
        out = self.guard(ep, est, tgt, mapped, hold)
        if self.diagnostics:
            zero = command(ep, est, np.zeros_like(a), live, sp, self.cfg.get('prior_residual_scale', .5), lateral_scale_for_targets(ep, est, tgt, self.cfg))
            self.diag.append(dict(t=float(ep.env.elapsed_s), a_raw=a_raw.tolist(), live=live.tolist(), hold=np.asarray(hold).tolist(),
                                  prior_speed=(sp.tolist() if sp is not None else None),
                                  zero_cmd=ep.ctl.to_world(zero, est).tolist(), mapped=ep.ctl.to_world(mapped, est).tolist(),
                                  guarded=ep.ctl.to_world(out, est).tolist()))
        self.prev_world = ep.ctl.to_world(out, est)
        return out


def worker(wid, conn, seed0, cfg):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    torch.manual_seed(cfg.get('training_seed', 0)+wid)
    from marl.deployable_sensing import DeployableConfig
    from scripts.benchmark_lysis import LysisEpisode, WallGuard
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); count = 0; policy = make_policy(cfg)

    def new_episode():
        nonlocal count
        while True:
            seed = seed0+count; count += 1
            w = np.array([cfg.get('hard_w', 1.) if a_ in HARD else 1. for a_ in train]); an = train[rng.choice(len(train), p=w/w.sum())]
            n = int(rng.integers(1, 4))
            s = 0. if rng.random() < .2 else float(rng.uniform(cfg.get('s_min', 0.), cfg['s_max']))
            # Flow is part of the training domain randomisation.  It is never
            # exposed to the policy; only the delayed image/history is used.
            flow = float(rng.choice(cfg.get('flow_levels', (0.05,))))
            if cfg.get('abcd'):
                draw = np.random.default_rng([seed, 620]).choice([.025, .05])
                flow = .025 if cfg['abcd'] == 'A' else float(draw)
                s = 0.
            try:
                ep = LysisEpisode(n, an, seed, sense_cfg=DeployableConfig(latency_steps=int(rng.integers(1, cfg.get('latency_max', 2)+1))),
                                  variation=s if s > 0 else None, flow_inlet_mm_s=flow)
                if cfg.get('abcd'):
                    # Fixed response; retain the existing spatial field and waveform.
                    from marl.physio_variation import Variation
                    ep.var = Variation(ep, 0., np.random.default_rng([seed, 5150]))
                    ep.var.amp = .12 if an in ('cerebral_venous_sinus', 'popliteal_calf_dvt', 'iliac_may_thurner') else .4
                break
            except (ValueError, RuntimeError):
                continue
        ep.hist = History(n, cfg['window'], DIM+(ECG_DIM if cfg.get('ecg') else 0)+int(cfg.get('observable_history', False))); ep.prev = np.zeros((n, 3)); ep.prev_raw = np.zeros((n, 3)); ep.id = count; ep.ret = 0.; ep.anatomy = an
        if cfg.get('matrix_arm') == 'belief_rolling_tpg' and n > 1:
            from marl.lysis_baselines import SwitchableTPG
            ep.rolling_tpg = SwitchableTPG(ep.coord, period_s=2.)
        ep.guard = WallGuard(ep); ep.s = s; ep.prior = PriorSpeed(n); ep.prev_world = np.zeros((n, 3)); ep.flow = flow; ep.scene_seed = seed
        return ep

    def obs(ep):
        est = ep.observe(); tgt = ep.plan_targets_now(est); ep.ctl.act(tgt, est)
        if hasattr(ep, 'rolling_tpg'):
            ep.rolling_tpg.update(ep.env.elapsed_s); hold = ep.rolling_tpg.c.gate(est.pos, est.active)
        else:
            hold = ep.hold(est)
        if cfg.get('observable_history'):
            ep.prev = ep.sent_world.copy()
            ep.prev_world = ep.sent_world.copy()
        seq, mask = ep.hist.push(token(ep, est, tgt, hold, ep.prev, cfg.get('ecg', False), cfg.get('observable_history', False)))
        live = (np.asarray(tgt) >= 0) & est.active & ~hold
        ep.cur = (est, tgt, hold, seq, mask, live, ep.prior(ep, est, tgt, ep.prev_world))

    ep = new_episode(); obs(ep); finished = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg)
        buf = dict(seq=[], mask=[], act=[], logp=[], val=[], rew=[], done=[], learn=[], stream=[], prior=[], motion=[], executed=[], motion_valid=[], dynamics=[], dynamics_valid=[])
        trunc_boot = {}
        for _ in range(ROLLOUT):
            env, n = ep.env, ep.n
            est, tgt, hold, seq, mask, live, a_prior = ep.cur
            with torch.no_grad():
                d, v = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            a_np = np.clip(a.numpy().astype(np.float64), -1, 1); a_np[~live] = 0.
            sp = 1.+a_prior[:, 0] if cfg.get('speed_prior') else None
            local = ep.guard(ep, est, tgt, command(ep, est, a_np, live, sp, cfg.get('prior_residual_scale', 1.0), lateral_scale_for_targets(ep, est, tgt, cfg)), hold)
            ep.prev_world = ep.ctl.to_world(local, est)
            P0 = env.positions_mm[:n].copy()
            g0 = np.array([route_remaining(ep.ctl, i, P0[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            done, out = ep.step(est, local, hold)
            P1 = env.positions_mm[:n]
            g1 = np.array([route_remaining(ep.ctl, i, P1[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            prog = np.clip(g0-g1, -.2, .2)*(np.asarray(tgt) >= 0)
            r = prog+50.*out['removed']-3.*out['wall']-10.*out['lost']-.02*((a_np-ep.prev_raw)**2).sum(1)-.03*live
            ep.prev_raw = a_np.copy()
            if n > 1:
                D = np.linalg.norm(P1[:, None]-P1[None], axis=-1)+np.eye(n)*99
                r -= .5*((D < ep.d_min).any(1))
            ep.prev = a_np.copy()
            if cfg.get('abcd'):
                # Token column 23:26 is the actual command in the controller
                # local frame; keep this coordinate system across reset,
                # training and evaluation.
                ep.prev = ep.prev_local.copy()
            # Exactly one sensor observation per step, including the terminal state.
            obs(ep)
            next_motion = ep.cur[3][:, -1, 19:22].copy()
            dyn_label = None
            if cfg.get('matrix_arm') and cfg.get('matrix_arm') != 'nav_tf_v3' and cfg.get('aux_weight', .01) > 0:
                from marl.lysis_matrix import training_dynamics_labels
                dyn_label = training_dynamics_labels(ep)
            final_seq, final_mask = ep.cur[3], ep.cur[4]
            for i in range(n):
                if out['active_before'][i]:
                    buf['seq'].append(seq[i]); buf['mask'].append(mask[i]); buf['act'].append(a[i].numpy())
                    buf['logp'].append(float(lp[i])); buf['val'].append(float(v[i])); buf['rew'].append(float(r[i]))
                    # Time-limit truncation bootstraps; task termination or
                    # robot exit cuts the stream. Held/TPG-gated rows remain in
                    # the stream for GAE continuity but are excluded from PPO.
                    buf['done'].append(bool(out['terminated'] or not env.active[i])); buf['learn'].append(bool(live[i]))
                    buf['stream'].append((wid, ep.id, i))
                    buf['prior'].append(a_prior[i])
                    buf['motion'].append(next_motion[i])
                    buf['executed'].append(ep.ctl.to_world(ep.prev_local, est)[i])
                    buf['motion_valid'].append(bool(env.active[i]))
                    if dyn_label is not None:
                        buf['dynamics'].append(dyn_label[i]); buf['dynamics_valid'].append(bool(env.active[i]))
            ep.ret += float(r.sum())
            if done:
                if out.get('truncated', False) and not out.get('terminated', False):
                    with torch.no_grad():
                        _, vb = policy.dist(torch.as_tensor(final_seq), torch.as_tensor(final_mask))
                    for i in range(n):
                        if out['active_before'][i] and env.active[i]:
                            trunc_boot[(wid, ep.id, i)] = float(vb[i])
                row = ep.row('train')
                finished.append(dict(anatomy=ep.anatomy, n=n, s=ep.s, success=bool(row['task_success']), removal=row['removal'],
                                     wall=row['wall_contact_s'], t90=row['t90_s'] or 300., ret=ep.ret,
                                     scenario_seed=ep.scene_seed, flow_inlet_mm_s=ep.flow))
                ep.close(); ep = new_episode()
                obs(ep)
        _, _, _, seq, mask, _, _ = ep.cur
        with torch.no_grad():
            _, vb = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
        final_boot={(wid, ep.id, i): float(vb[i]) for i in range(ep.n)}
        final_boot.update(trunc_boot)
        conn.send(dict(buf=buf, boot=final_boot, finished=finished)); finished = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=120)
    p.add_argument('--workers', type=int, default=12); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arch', default='transformer', choices=('transformer', 'gru', 'mlp'))
    p.add_argument('--layers', type=int, default=3); p.add_argument('--window', type=int, default=32)
    p.add_argument('--s-max', type=float, default=1.25); p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--init', type=Path, help='warm start checkpoint')
    p.add_argument('--speed-prior', action='store_true', help='v3: speed action relative to the adaptive classical speed')
    p.add_argument('--prior-residual-scale', type=float, default=1.0)
    p.add_argument('--lateral-residual-scale', type=float, default=1.0,
                   help='route lateral aim offset multiplier; old checkpoints default to 1')
    p.add_argument('--lateral-near-radius', type=float, default=None,
                   help='apply lateral scaling only inside this observed target distance (mm)')
    p.add_argument('--aux-weight', type=float, default=.01,
                   help='matrix auxiliary weight; 0 avoids collecting privileged labels')
    p.add_argument('--observable-history', action='store_true',
                   help='history uses final world-frame commands and camera frame age (40 tokens)')
    p.add_argument('--no-privileged-supervision', action='store_true',
                   help='reject pretrained initialization and privileged matrix auxiliary targets')
    p.add_argument('--beta0', type=float, default=1.); p.add_argument('--beta-end', type=float, default=.05)
    p.add_argument('--s-min', type=float, default=0.); p.add_argument('--hard-w', type=float, default=1.)
    p.add_argument('--ecg', action='store_true', help='add the ECG cardiac phase (sin, cos) to the observation')
    p.add_argument('--adaptive-history', action='store_true', help='default deployable adaptation recipe: GRU, 8 control steps')
    p.add_argument('--abcd', choices=('A', 'B', 'C', 'D'))
    p.add_argument('--updates', type=int, default=0, help='fixed rollout budget, overrides minutes')
    p.add_argument('--agent-steps', type=int, default=0, help='retained active-agent transition budget; overrides updates/minutes')
    p.add_argument('--scene-seed-base', type=int, default=2100000000)
    p.add_argument('--latency-max', type=int, choices=(2,3), default=3)
    p.add_argument('--flow-levels', default='0.025,0.05,0.1',
                   help='comma-separated nominal inlet flow levels used for domain randomisation')
    p.add_argument('--matrix-arm', choices=('nav_tf_v3','flow_aux','belief','belief_rolling_tpg'), default='nav_tf_v3')
    a = p.parse_args()
    if a.no_privileged_supervision and (a.init is not None or a.ecg or a.abcd is not None or
            (a.matrix_arm != 'nav_tf_v3' and a.aux_weight > 0)):
        p.error('No-privileged-supervision requires fresh initialization, no ECG/legacy ABCD, and no privileged matrix auxiliary')
    a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    if a.adaptive_history:
        a.arch, a.window = 'gru', 8
    flow_levels = tuple(float(x) for x in a.flow_levels.split(',') if x.strip())
    if not flow_levels or any(x <= 0 for x in flow_levels):
        raise ValueError('--flow-levels must contain positive comma-separated values')
    cfg = dict(mode='route', arch=a.arch, layers=a.layers, window=a.window, s_max=a.s_max, speed_prior=a.speed_prior, prior_residual_scale=a.prior_residual_scale, s_min=a.s_min, hard_w=a.hard_w, ecg=a.ecg, adaptive_history=a.adaptive_history, matrix_arm=a.matrix_arm, flow_levels=flow_levels)
    cfg.update(abcd=a.abcd, training_seed=a.seed, updates=a.updates, agent_step_budget=a.agent_steps, scene_seed_base=a.scene_seed_base)
    cfg.update(latency_max=a.latency_max, training_revision='prior_truncation_hold_fixed_local_history')
    cfg['lateral_residual_scale'] = a.lateral_residual_scale
    cfg['lateral_near_radius'] = a.lateral_near_radius
    cfg['aux_weight'] = a.aux_weight
    cfg.update(observable_history=a.observable_history, no_privileged_supervision=a.no_privileged_supervision)
    (a.out/'config.json').write_text(json.dumps(dict(vars(a), **cfg), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe()
        # Keep the training scene registry explicit.  Earlier runs used a
        # hidden seed formula for non-ABCD arms, which made seen/unseen audits
        # harder to reproduce.
        seed_base = a.scene_seed_base + a.seed*10000000
        pr = ctx.Process(target=worker, args=(w, child, seed_base+w*200000, cfg)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = make_policy(cfg)
    if a.init:
        policy.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['state'])
    else:
        with torch.no_grad():
            policy.log_std.fill_(-1.2)
    policy = policy.to(a.device)
    opt = torch.optim.AdamW(policy.parameters(), 3e-4, betas=(.9, .98), weight_decay=.01)
    t0, it, steps, log = time.time(), 0, 0, (a.out/'log.jsonl').open('a'); recent = []
    save = lambda path: torch.save(dict(state=policy.state_dict(), optimizer=opt.state_dict(), it=it, agent_steps=steps, environment_steps=it*a.workers*ROLLOUT, cfg=cfg), path)
    milestones_saved = set()
    episodes_log = (a.out/'episodes.jsonl').open('a')
    while (steps < a.agent_steps) if a.agent_steps else ((it < a.updates) if a.updates else (time.time()-t0 < a.minutes*60)):
        frac = steps/a.agent_steps if a.agent_steps else (it/a.updates if a.updates else min((time.time()-t0)/(a.minutes*60), 1.))
        cos = .5*(1+math.cos(math.pi*frac))
        for g in opt.param_groups:
            g['lr'] = 5e-5+(a.lr-5e-5)*cos
        clip = .05+(.2-.05)*cos; ent = 3e-3*cos; beta = a.beta_end+(a.beta0-a.beta_end)*cos
        sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        for c in pipes:
            c.send(sd)
        data = [c.recv() for c in pipes]
        buf = {k: sum((d['buf'][k] for d in data), []) for k in data[0]['buf']}
        boot = {}; [boot.update(d['boot']) for d in data]; [recent.extend(d['finished']) for d in data]
        for d in data:
            for episode in d['finished']:
                episodes_log.write(json.dumps(dict(episode, training_seed=a.seed))+'\n')
        episodes_log.flush()
        if not buf['rew']:
            continue
        adv, ret = gae(buf, boot)
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=a.device)
        seq, mask, act, lp0 = T(buf['seq']), T(buf['mask'], torch.bool), T(buf['act']), T(buf['logp'])
        PRI = T(buf['prior'])*(0. if cfg.get('speed_prior') else 1.)
        motion, executed, motion_valid = T(buf['motion']), T(buf['executed']), T(buf['motion_valid'])
        learn = T(buf['learn'])
        aux_records = []
        A, R = T(adv), T(ret); A = (A-A.mean())/(A.std()+1e-8)
        for _ in range(3):
            for b in torch.randperm(len(A), device=a.device).split(2048):
                d, v = policy.dist(seq[b], mask[b]); lp = d.log_prob(act[b]).sum(-1); ratio = (lp-lp0[b]).exp()
                w = learn[b]; denom=w.sum().clamp(min=1)
                l_pi = (-torch.min(ratio*A[b], ratio.clamp(1-clip, 1+clip)*A[b])*w).sum()/denom
                prior_loss = weighted_prior_loss(d.mean, PRI[b], w)
                l = l_pi+.5*(((v-R[b])**2)*w).sum()/denom-ent*(d.entropy().sum(-1)*w).sum()/denom+beta*prior_loss
                if a.abcd == 'D':
                    pred = policy.predict_motion(seq[b], mask[b], executed[b])
                    aux = (nn.functional.smooth_l1_loss(pred, motion[b], reduction='none').mean(-1)*motion_valid[b]).sum()/motion_valid[b].sum().clamp(min=1)
                    mse = ((pred-motion[b]).square().mean(-1)*motion_valid[b]).sum()/motion_valid[b].sum().clamp(min=1)
                    aux_records.append((float(aux.detach()), float(mse.detach()), float(l.detach())))
                    l = l+.01*aux
                elif cfg.get('matrix_arm') and cfg.get('matrix_arm') != 'nav_tf_v3' and cfg['aux_weight'] > 0:
                    labels = T(buf['dynamics'])[b]; valid = T(buf['dynamics_valid'])[b]
                    aux, mse = policy.auxiliary(seq[b], mask[b], labels, valid)
                    aux_records.append((float(aux.detach()), float(mse.detach()), float(l.detach())))
                    l = l + cfg['aux_weight']*aux
                if not torch.isfinite(l):
                    raise FloatingPointError('nonfinite PPO loss')
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.); opt.step()
        it += 1; steps += len(A); recent = recent[-300:]
        m = lambda k, sel=None: (float(np.mean([e[k] for e in recent if sel is None or sel(e)])) if recent else None)
        row = dict(it=it, agent_steps=steps, environment_steps=it*a.workers*ROLLOUT, learning_rows=int(learn.sum().item()), minutes=round((time.time()-t0)/60, 2), episodes=len(recent),
                   success=m('success'), removal=m('removal'), wall=m('wall'), t90=m('t90'), ret=m('ret'),
                   success_s0=m('success', lambda e: e['s'] == 0) if any(e['s'] == 0 for e in recent) else None,
                   lr=opt.param_groups[0]['lr'], beta=beta, mean_a0=float(act[:, 0].mean()), log_std=policy.log_std.tolist())
        log.write(json.dumps(row)+'\n'); log.flush()
        if aux_records:
            with (a.out/'auxiliary.jsonl').open('a') as f:
                avg = np.mean(aux_records, axis=0)
                f.write(json.dumps(dict(it=it, huber=float(avg[0]), mse=float(avg[1]), weighted_loss=float(.01*avg[0]), ppo_loss=float(avg[2])))+'\n')
        if it % 10 == 0:
            save(a.out/'policy.pt')
        if it % 100 == 0 or it == 1:
            save(a.out/f'policy_it{it}.pt')
        for milestone in (250000, 500000, 1000000, 2000000):
            if steps >= milestone and milestone not in milestones_saved:
                save(a.out/f'policy_steps{milestone}.pt')
                milestones_saved.add(milestone)
    save(a.out/'policy.pt')
    for c in pipes:
        c.send(None)
    for pr in procs:
        pr.join(timeout=10)


if __name__ == '__main__':
    main()
