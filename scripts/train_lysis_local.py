"""PPO training of a temporal local controller in benchmark v4 (scripts/benchmark_lysis.py).

Two uses, identical environment / budget / reward, different action definition:
  --mode direct    Turbo baseline (An et al., Nat. Mach. Intell. 2026): causal Transformer PPO over the observation
                   history outputs the 3-D velocity command itself (no rule prior); here with the vascular task's
                   deployable observation and multi-level domain randomisation. Allocation A + TPG + spacing shield are
                   shared with every method (only the local controller is replaced).
  --mode residual  ours (low level): bounded residual on the classical pursuit + wall-guard command,
                   zero-initialised output so the untrained policy is the classical controller.
Observation token (per cluster, world frame, deployable): route carrot direction (pre-operative route), the guard
command (residual mode only; zero in direct mode), estimated velocity, previous command, distance to target,
map radial offset ratio, map wall clearance, junction flag, TPG hold, time left, radial offset vector, two peers
(relative position, visible).
Reward per cluster and step (truth; never an input): +1 x route progress (mm, clipped 0.2) +50 x team removal
increment -10 x wall-contact s -0.5 if closer than d_min to a peer -0.01|a|^2 (residual) ; lost -10.
Domain randomisation: actuation gain U[0.75, 1.25] per cluster, execution noise N(0, 0.025), sensing latency
U{1, 2} frames. Training anatomies only, N ~ U{1,2,3}, seeds 1717000000+.
usage: train_lysis_local.py --out DIR --mode direct --minutes 100 --workers 8
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

ROLLOUT = 256
# v2 (2026-10-08 01:2x): 3 -> 50. Diagnosis: route pursuit spends 295 steps within 1 mm of its clot but only 81 in
# lysis contact (PAC-NMPC 111 of 153); with weight 3 the removal signal (~0.0075 per lysing step) was far below the
# progress term (0.1 per step) and settling on the clot carried no reward. v1 run aborted at 20 min.
REMOVAL_W = 50.
TOK = 3+3+3+3+1+5+3+8


def token(ep, est, rule, tgt, hold, prev, mode):
    env, ctl, n = ep.env, ep.ctl, ep.n
    F = ctl.frames(est); spd = env.config.robot_speed_mm_s; body = float(env.config.robot_radius_mm)
    T = np.zeros((n, TOK), np.float32); rule_w = np.zeros((n, 3))
    for i in range(n):
        if not est.active[i] or tgt[i] < 0:
            continue
        o = T[i]; rule_w[i] = F[i].T@rule[i]
        o[0:3] = F[i].T@ctl.nominal[i]
        o[3:6] = rule_w[i] if mode == 'residual' else 0.
        o[6:9] = est.vel[i]/spd; o[9:12] = prev[i]
        o[12] = np.linalg.norm(env.clot_positions_mm[tgt[i]]-est.pos[i])/10.
        ax, r, rad = ep.sensor.map_coordinates(est, i)
        R = ctl.route[i]; pk = ctl.prog[i]
        o[13] = rad/max(r, 1e-9); o[14] = (r-rad-body)/max(r, 1e-9)
        o[15] = float(R is not None and np.any(ctl.deg[R[max(pk-3, 0):pk+6]] >= 3))
        o[16] = float(hold[i]); o[17] = max(0., 1-env.elapsed_s/env.config.episode_duration_s)
        o[18:21] = (est.pos[i]-ax)/max(r, 1e-9)
        vis = np.flatnonzero(est.peers_vis[i]); vis = vis[np.argsort(np.linalg.norm(est.peers_rel[i, vis], axis=1))][:2]
        for j, q in enumerate(vis):
            s = 21+4*j; o[s:s+3] = est.peers_rel[i, q]/6.; o[s+3] = 1.
    return T, rule_w


def compose(rule_w, a, mode):
    from marl.obstacle_control import DEADZONE
    a = np.clip(a, -1, 1)
    u = rule_w+.6*a if mode == 'residual' else a
    u = u/np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1.)
    if mode == 'residual':
        u[np.linalg.norm(rule_w, axis=-1) == 0] = 0.     # the rule stops (dwell / hold): stay
    if mode == 'direct':                                  # Turbo-style stop dead zone; the residual keeps the prior's
        u[np.linalg.norm(u, axis=-1) < DEADZONE] = 0.     # small settling commands (bounded actuator)
    return u


def route_remaining(ctl, i, pos):
    R = ctl.route[i]
    if R is None:
        return 0.
    P = ctl.pts[R]; k = int(ctl.prog[i])
    seg = np.linalg.norm(np.diff(P[k:], axis=0), axis=1).sum() if k < len(P)-1 else 0.
    return float(seg+np.linalg.norm(pos-P[k]))


class LocalHistory:
    def __init__(self, n, window):
        self.seq = np.zeros((n, window, TOK), np.float32); self.mask = np.ones((n, window), bool)

    def push(self, T):
        self.seq = np.roll(self.seq, -1, 1); self.mask = np.roll(self.mask, -1, 1)
        self.seq[:, -1] = T; self.mask[:, -1] = False
        return self.seq.copy(), self.mask.copy()


def make_policy(cfg):
    from marl.obstacle_control import TemporalPolicy
    return TemporalPolicy(cfg['arch'], layers=cfg['layers'], window=cfg['window'], dim=TOK)


class LocalController:
    """Evaluation wrapper (deterministic mean action); `low` callable for scripts/benchmark_lysis.rollout."""
    def __init__(self, ep, ckpt, post_guard=False):
        torch.set_num_threads(1)
        c = torch.load(ckpt, map_location='cpu', weights_only=False); self.cfg = c['cfg']
        self.net = make_policy(self.cfg); self.net.load_state_dict(c['state']); self.net.eval()
        self.hist = LocalHistory(ep.n, self.cfg['window']); self.prev = np.zeros((ep.n, 3))
        self.guard = None
        if self.cfg['mode'] == 'residual':
            from scripts.benchmark_lysis import SettleGuard, WallGuard
            self.guard = SettleGuard(ep) if self.cfg.get('prior', 'guard') == 'settle' else WallGuard(ep)
        # post_guard (evaluation option, 2026-10-08): the map wall guard is applied again to the learned command,
        # so the residual cannot steer outward inside the guard band (the prior's guard only shapes its input)
        self.post = None
        if post_guard:
            from marl.hierarchical_navigation import NavigationConfig, WallRecoveryExecutor
            self.post = WallRecoveryExecutor(ep.sensor, ep.n, ep.ctl.body, NavigationConfig(recovery_margin_mm=.2, recovery_gain=.4))

    def __call__(self, ep, est, tgt, rule, hold):
        if self.guard is not None:
            rule = self.guard(ep, est, tgt, rule, hold)
        T, rule_w = token(ep, est, rule, tgt, hold, self.prev, self.cfg['mode'])
        seq, mask = self.hist.push(T)
        with torch.no_grad():
            a = self.net.pi(self.net.backbone(torch.as_tensor(seq), torch.as_tensor(mask))).numpy().astype(np.float64)
        live = (np.asarray(tgt) >= 0) & est.active & ~hold
        u = compose(rule_w, a, self.cfg['mode']); u[~live] = 0.
        self.prev = u.copy()
        out = np.einsum('nij,nj->ni', ep.ctl.frames(est), u)
        if self.post is not None:
            out = self.post.act(ep.env.elapsed_s, est, ep.ctl, out, hold)
        return out


def worker(wid, conn, seed0, cfg):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    from marl.deployable_sensing import DeployableConfig
    from scripts.benchmark_lysis import LysisEpisode, SettleGuard, WallGuard
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); count = 0; policy = make_policy(cfg); mode = cfg['mode']

    def new_episode():
        nonlocal count
        while True:
            seed = seed0+count; count += 1
            an = train[rng.integers(len(train))]; n = int(rng.integers(1, 4))
            sc = DeployableConfig(latency_steps=int(rng.integers(1, 3))) if cfg['dr'] else None
            try:
                sv = 0. if (cfg.get('s_max', 0.) <= 0 or rng.random() < .2) else float(rng.uniform(0, cfg['s_max']))
                ep = LysisEpisode(n, an, seed, sense_cfg=sc, rng=rng, variation=sv if sv > 0 else None,
                                  actuation=dict(gain=(.75, 1.25), noise=.025) if (cfg['dr'] and sv == 0) else None)
                break
            except (ValueError, RuntimeError):
                continue
        ep.hist = LocalHistory(n, cfg['window']); ep.prev = np.zeros((n, 3)); ep.id = count; ep.ret = 0.; ep.anatomy = an
        ep.guard = (SettleGuard(ep) if cfg.get('prior') == 'settle' else WallGuard(ep)) if mode == 'residual' else None
        return ep

    def obs(ep):
        est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
        if ep.guard is not None:
            rule = ep.guard(ep, est, tgt, rule, hold)
        T, rule_w = token(ep, est, rule, tgt, hold, ep.prev, mode)
        seq, mask = ep.hist.push(T)
        live = (np.asarray(tgt) >= 0) & est.active & ~hold
        ep.cur = (est, tgt, hold, rule_w, seq, mask, live)

    ep = new_episode(); obs(ep); finished = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg)
        buf = dict(seq=[], mask=[], act=[], logp=[], val=[], rew=[], done=[], stream=[])
        for _ in range(ROLLOUT):
            env, n = ep.env, ep.n
            est, tgt, hold, rule_w, seq, mask, live = ep.cur
            with torch.no_grad():
                d, v = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            a_np = a.numpy().astype(np.float64)
            u = compose(rule_w, a_np, mode); u[~live] = 0.
            ep.prev = u.copy()
            local = np.einsum('nij,nj->ni', ep.ctl.frames(est), u)
            P0 = env.positions_mm[:n].copy()
            g0 = np.array([route_remaining(ep.ctl, i, P0[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            done, out = ep.step(est, local, hold)
            P1 = env.positions_mm[:n]
            g1 = np.array([route_remaining(ep.ctl, i, P1[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            prog = np.clip(g0-g1, -.2, .2)*(np.asarray(tgt) >= 0)
            r = 1.*prog+REMOVAL_W*out['removed']-10.*out['wall']-10.*out['lost']
            if mode == 'residual':
                r -= .01*(np.clip(a_np, -1, 1)**2).sum(1)
            if n > 1:
                D = np.linalg.norm(P1[:, None]-P1[None], axis=-1)+np.eye(n)*99
                r -= .5*((D < ep.d_min).any(1))
            for i in range(n):
                # informative samples: steps where the cluster moves under its own decision
                if out['active_before'][i] and live[i]:
                    buf['seq'].append(seq[i]); buf['mask'].append(mask[i]); buf['act'].append(a[i].numpy())
                    buf['logp'].append(float(lp[i])); buf['val'].append(float(v[i])); buf['rew'].append(float(r[i]))
                    buf['done'].append(bool(done or not env.active[i])); buf['stream'].append((wid, ep.id, i))
            ep.ret += float(r.sum())
            if done:
                row = ep.row('train')
                finished.append(dict(anatomy=ep.anatomy, n=n, success=bool(row['task_success']), removal=row['removal'],
                                     wall=row['wall_contact_s'], relaxed=row['relaxed_success'], t=row['elapsed_s'], ret=ep.ret))
                ep.close(); ep = new_episode()
            obs(ep)
        _, _, _, _, seq, mask, _ = ep.cur
        with torch.no_grad():
            _, vb = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
        conn.send(dict(buf=buf, boot={(wid, ep.id, i): float(vb[i]) for i in range(ep.n)}, finished=finished)); finished = []


def gae(buf, boot, gamma=.98, lam=.95):
    idx = {}
    for k, s in enumerate(buf['stream']):
        idx.setdefault(s, []).append(k)
    adv = np.zeros(len(buf['rew'])); ret = np.zeros(len(buf['rew']))
    for s, ks in idx.items():
        last_v = 0. if buf['done'][ks[-1]] else boot.get(s, 0.); g = 0.
        for k in reversed(ks):
            delta = buf['rew'][k]+gamma*last_v*(1-buf['done'][k])-buf['val'][k]
            g = delta+gamma*lam*(1-buf['done'][k])*g
            adv[k] = g; ret[k] = g+buf['val'][k]; last_v = buf['val'][k]
    return adv, ret


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=100)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mode', required=True, choices=('direct', 'residual'))
    p.add_argument('--arch', default='transformer', choices=('transformer', 'gru', 'mlp'))
    p.add_argument('--layers', type=int, default=3); p.add_argument('--window', type=int, default=32)
    p.add_argument('--no-dr', action='store_true'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--s-max', type=float, default=0., help='v5 physiological variation DR (marl.physio_variation)')
    p.add_argument('--prior', default='settle', choices=('guard', 'settle'), help='residual mode: classical prior')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    cfg = dict(mode=a.mode, arch=a.arch, layers=a.layers, window=a.window, dr=not a.no_dr, prior=a.prior, s_max=a.s_max)
    (a.out/'config.json').write_text(json.dumps(dict(vars(a), **cfg), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(w, child, 1717000000+a.seed*10000000+w*200000, cfg)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = make_policy(cfg).to(a.device)
    with torch.no_grad():
        policy.log_std.fill_(-.5 if a.mode == 'direct' else -1.2)
    opt = torch.optim.AdamW(policy.parameters(), 3e-4, betas=(.9, .98), weight_decay=.01)
    t0, it, steps, log = time.time(), 0, 0, (a.out/'log.jsonl').open('a'); recent = []
    save = lambda path: torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps, cfg=cfg), path)
    while time.time()-t0 < a.minutes*60:
        frac = min((time.time()-t0)/(a.minutes*60), 1.)
        cos = .5*(1+math.cos(math.pi*frac))                      # cosine schedules (Turbo)
        for g in opt.param_groups:
            g['lr'] = 5e-5+(a.lr-5e-5)*cos
        clip = .05+(.2-.05)*cos; ent = 3e-3*cos
        sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        for c in pipes:
            c.send(sd)
        data = [c.recv() for c in pipes]
        buf = {k: sum((d['buf'][k] for d in data), []) for k in data[0]['buf']}
        boot = {}; [boot.update(d['boot']) for d in data]; [recent.extend(d['finished']) for d in data]
        if not buf['rew']:
            continue
        adv, ret = gae(buf, boot)
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=a.device)
        seq, mask, act, lp0 = T(buf['seq']), T(buf['mask'], torch.bool), T(buf['act']), T(buf['logp'])
        A, R = T(adv), T(ret); A = (A-A.mean())/(A.std()+1e-8)
        for _ in range(3):
            for b in torch.randperm(len(A), device=a.device).split(2048):
                d, v = policy.dist(seq[b], mask[b]); lp = d.log_prob(act[b]).sum(-1); ratio = (lp-lp0[b]).exp()
                l_pi = -torch.min(ratio*A[b], ratio.clamp(1-clip, 1+clip)*A[b]).mean()
                l = l_pi+.5*((v-R[b])**2).mean()-ent*d.entropy().sum(-1).mean()
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.); opt.step()
        it += 1; steps += len(A); recent = recent[-300:]
        m = lambda k: float(np.mean([e[k] for e in recent])) if recent else None
        row = dict(it=it, agent_steps=steps, minutes=round((time.time()-t0)/60, 2), episodes=len(recent),
                   success=m('success'), relaxed=m('relaxed'), removal=m('removal'), wall=m('wall'), ret=m('ret'),
                   lr=opt.param_groups[0]['lr'], log_std=policy.log_std.tolist())
        log.write(json.dumps(row)+'\n'); log.flush()
        if it % 10 == 0:
            save(a.out/'policy.pt')
        if it % 100 == 0:
            save(a.out/f'policy_it{it}.pt')
    save(a.out/'policy.pt')
    for c in pipes:
        c.send(None)
    for pr in procs:
        pr.join(timeout=10)


if __name__ == '__main__':
    main()
