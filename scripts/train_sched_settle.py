"""PPO training of Scheduled-Settle RL (marl.sched_settle): bounded learned parameter residuals on Adaptive Settle.

No privileged supervision: the actor and the critic see only the deployable token history (scripts.train_lysis_nav
token with observable history = sent world command + camera frame age), the controller's own internal state and the
previous action. Simulator truth enters only the reward and the termination flags.
Reward per cluster and step, aligned with the Strict criterion (task success, no lost cluster, wall < 1 robot-s,
spacing compliant): +1 route progress (mm, clipped 0.2) +50 team removal -10 wall-contact s -10 lost
-0.5 spacing (closer than d_min) -0.03 per live step -0.01 |a - a_prev|^2; at episode end +20 to every cluster
when the episode is Strict-successful. gamma 0.995, GAE lambda 0.95.
Output layer zero-initialised (a = 0 is Adaptive Settle); log std -1.6; an L2 pull of the action mean toward 0
(beta 0.05 -> 0.005, cosine) keeps the policy at the prior unless deviating pays.
Training domain (train anatomies only, N ~ U{1,2,3}): healthy inlet speed {0.025, 0.05, 0.1} mm/s, image latency
U{1,2,3} steps, v5 variation s = 0 (p 0.3) or U[0, 1.25]. Scene seeds 1919000000+ (disjoint from the dev pools).
usage: train_sched_settle.py --out DIR --minutes 150 --workers 16 --seed 0
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

from marl.sched_settle import ACT2_DIM, ACT_DIM, ACT_FLOW_DIM, FlowResidual, ScheduledSettle, SwitchResidual
from scripts.train_lysis_local import route_remaining
from scripts.train_lysis_nav import DIM, History, gae, token

ROLLOUT = 256
TOK = DIM+1+4+ACT_DIM          # base token + frame age + controller state + previous action


def spec(cfg):
    """(controller class, action dim, token dim) for the available safe residual priors."""
    if cfg.get('prior', 'adaptive') == 'flow':
        return FlowResidual, ACT_FLOW_DIM, DIM+1+11+ACT_FLOW_DIM
    if cfg.get('prior', 'adaptive') == 'switch':
        return SwitchResidual, ACT2_DIM, DIM+1+5+ACT2_DIM
    return ScheduledSettle, ACT_DIM, TOK


def full_token(ep, est, tgt, hold, ctl, prev_a):
    T = token(ep, est, tgt, hold, ep.sent_world.copy(), observable_history=True)
    return np.concatenate([T, ctl.state(ep.n).astype(np.float32), prev_a.astype(np.float32)], 1)


def make_policy(cfg):
    from marl.obstacle_control import TemporalPolicy
    _, adim, tdim = spec(cfg)
    p = TemporalPolicy(cfg['arch'], layers=cfg['layers'], window=cfg['window'], dim=tdim)
    d = p.embed[0].out_features
    p.pi = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, adim))
    nn.init.zeros_(p.pi[-1].weight); nn.init.zeros_(p.pi[-1].bias)
    p.log_std = nn.Parameter(torch.full((adim,), cfg.get('init_log_std', -1.6)))
    return p


class SchedController:
    """Deterministic evaluation wrapper; `low` callable for scripts.benchmark_lysis / evaluate_hard_baselines."""
    def __init__(self, ep, ckpt):
        torch.set_num_threads(1)
        c = torch.load(ckpt, map_location='cpu', weights_only=False); self.cfg = c['cfg']
        self.net = make_policy(self.cfg); self.net.load_state_dict(c['state']); self.net.eval()
        C, adim, tdim = spec(self.cfg)
        self.ctl = C(ep); self.hist = History(ep.n, self.cfg['window'], tdim)
        self.prev_a = np.zeros((ep.n, adim)); self.actions = []

    def __call__(self, ep, est, tgt, rule, hold):
        seq, mask = self.hist.push(full_token(ep, est, tgt, hold, self.ctl, self.prev_a))
        with torch.no_grad():
            a = self.net.pi(self.net.backbone(torch.as_tensor(seq), torch.as_tensor(mask))).numpy().astype(np.float64)
        live = (np.asarray(tgt) >= 0) & est.active & ~hold
        a = np.clip(a, -1, 1); a[~live] = 0.; self.prev_a = a.copy(); self.actions.append(a.copy())
        return self.ctl(ep, est, tgt, rule, hold, a)


def strict_success(row):
    return bool(row.get('strict_success', False))


def worker(wid, conn, seed0, cfg):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1); torch.manual_seed(cfg['seed']*1000+wid)
    from marl.deployable_sensing import DeployableConfig
    from scripts.benchmark_lysis import LysisEpisode
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); count = 0; policy = make_policy(cfg)

    def new_episode():
        nonlocal count
        while True:
            seed = seed0+count; count += 1
            an = train[rng.integers(len(train))]; n = int(rng.integers(1, 4))
            s = 0. if rng.random() < .3 else float(rng.uniform(0, cfg['s_max']))
            flow = float(rng.choice(cfg['flow_levels'])); lat = int(rng.integers(1, cfg['latency_max']+1))
            try:
                ep = LysisEpisode(n, an, seed, sense_cfg=DeployableConfig(latency_steps=lat),
                                  variation=s if s > 0 else None, flow_inlet_mm_s=flow)
                break
            except (ValueError, RuntimeError):
                continue
        C, adim, tdim = spec(cfg)
        ep.ctl2 = C(ep); ep.hist = History(n, cfg['window'], tdim); ep.prev_a = np.zeros((n, adim))
        ep.id = count; ep.ret = 0.; ep.meta = dict(anatomy=an, n=n, s=s, flow=flow, latency=lat)
        return ep

    def obs(ep):
        est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
        seq, mask = ep.hist.push(full_token(ep, est, tgt, hold, ep.ctl2, ep.prev_a))
        live = (np.asarray(tgt) >= 0) & est.active & ~hold
        ep.cur = (est, tgt, rule, hold, seq, mask, live)

    ep = new_episode(); obs(ep); finished = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg)
        buf = dict(seq=[], mask=[], act=[], logp=[], val=[], rew=[], done=[], learn=[], stream=[])
        boot = {}
        for _ in range(ROLLOUT):
            env, n = ep.env, ep.n
            est, tgt, rule, hold, seq, mask, live = ep.cur
            with torch.no_grad():
                d, v = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            a_np = np.clip(a.numpy().astype(np.float64), -1, 1); a_np[~live] = 0.
            local = ep.ctl2(ep, est, tgt, rule, hold, a_np)
            P0 = env.positions_mm[:n].copy()
            g0 = np.array([route_remaining(ep.ctl, i, P0[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            done, out = ep.step(est, local, hold)
            P1 = env.positions_mm[:n]
            g1 = np.array([route_remaining(ep.ctl, i, P1[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            r = (np.clip(g0-g1, -.2, .2)*(np.asarray(tgt) >= 0)+50.*out['removed']-10.*out['wall']-10.*out['lost']
                 -.03*live-.01*((a_np-ep.prev_a)**2).sum(1))
            if n > 1:
                D = np.linalg.norm(P1[:, None]-P1[None], axis=-1)+np.eye(n)*99
                r -= .5*((D < ep.d_min).any(1))
            row = None
            if done:
                row = ep.row('train')
                if strict_success(row):
                    r += 20.
            ep.prev_a = a_np.copy()
            for i in range(n):
                if out['active_before'][i]:
                    buf['seq'].append(seq[i]); buf['mask'].append(mask[i]); buf['act'].append(a[i].numpy())
                    buf['logp'].append(float(lp[i])); buf['val'].append(float(v[i])); buf['rew'].append(float(r[i]))
                    buf['done'].append(bool(out['terminated'] or not env.active[i])); buf['learn'].append(bool(live[i]))
                    buf['stream'].append((wid, ep.id, i))
            ep.ret += float(r.sum())
            if done:
                if out['truncated'] and not out['terminated']:      # time limit: bootstrap from the final state
                    obs(ep)
                    with torch.no_grad():
                        _, vb = policy.dist(torch.as_tensor(ep.cur[4]), torch.as_tensor(ep.cur[5]))
                    for i in range(n):
                        if out['active_before'][i] and env.active[i]:
                            boot[(wid, ep.id, i)] = float(vb[i])
                finished.append(dict(ep.meta, strict=strict_success(row), success=bool(row['task_success']),
                                     removal=row['removal'], wall=row['wall_contact_s'], t90=row['t90_s'] or 300., ret=ep.ret))
                ep.close(); ep = new_episode()
            obs(ep)
        with torch.no_grad():
            _, vb = policy.dist(torch.as_tensor(ep.cur[4]), torch.as_tensor(ep.cur[5]))
        for i in range(ep.n):
            boot.setdefault((wid, ep.id, i), float(vb[i]))
        conn.send(dict(buf=buf, boot=boot, finished=finished)); finished = []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=150)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arch', default='transformer', choices=('transformer', 'gru', 'mlp'))
    p.add_argument('--layers', type=int, default=3); p.add_argument('--window', type=int, default=32)
    p.add_argument('--s-max', type=float, default=1.25); p.add_argument('--latency-max', type=int, default=3)
    p.add_argument('--flow-levels', default='0.025,0.05,0.1'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--beta0', type=float, default=.05)
    p.add_argument('--beta-end', type=float, default=.005)
    p.add_argument('--prior', default='switch', choices=('adaptive', 'switch', 'flow'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    cfg = dict(mode='sched', arch=a.arch, layers=a.layers, window=a.window, s_max=a.s_max, latency_max=a.latency_max,
               flow_levels=[float(x) for x in a.flow_levels.split(',')], seed=a.seed, privileged_supervision=False,
               prior=a.prior)
    (a.out/'config.json').write_text(json.dumps(dict(vars(a), **cfg), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(w, child, 1919000000+a.seed*10000000+w*200000, cfg)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = make_policy(cfg).to(a.device)
    opt = torch.optim.AdamW(policy.parameters(), 3e-4, betas=(.9, .98), weight_decay=.01)
    t0, it, steps, log = time.time(), 0, 0, (a.out/'log.jsonl').open('a'); recent = []
    save = lambda path: torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps, cfg=cfg), path)
    save(a.out/'policy_it0.pt')
    while time.time()-t0 < a.minutes*60:
        frac = min((time.time()-t0)/(a.minutes*60), 1.); cos = .5*(1+math.cos(math.pi*frac))
        for g in opt.param_groups:
            g['lr'] = 5e-5+(a.lr-5e-5)*cos
        clip = .05+(.2-.05)*cos; ent = 1e-3*cos; beta = a.beta_end+(a.beta0-a.beta_end)*cos
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
        L = T(buf['learn'])
        seq, mask, act, lp0 = T(buf['seq']), T(buf['mask'], torch.bool), T(buf['act']), T(buf['logp'])
        A, R = T(adv), T(ret); sel = L > 0
        A = (A-A[sel].mean())/(A[sel].std()+1e-8)
        for _ in range(3):
            for b in torch.randperm(len(A), device=a.device).split(2048):
                d, v = policy.dist(seq[b], mask[b]); lp = d.log_prob(act[b]).sum(-1); ratio = (lp-lp0[b]).exp()
                w = L[b]; nw = w.sum().clamp(min=1)
                l_pi = -(torch.min(ratio*A[b], ratio.clamp(1-clip, 1+clip)*A[b])*w).sum()/nw
                l_reg = ((d.mean**2).sum(-1)*w).sum()/nw
                l = l_pi+.5*((v-R[b])**2).mean()-ent*(d.entropy().sum(-1)*w).sum()/nw+beta*l_reg
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.); opt.step()
        it += 1; steps += int(sel.sum()); recent = recent[-400:]
        m = lambda k, f=None: (float(np.mean([e[k] for e in recent if f is None or f(e)])) if any(f is None or f(e) for e in recent) else None)
        row = dict(it=it, agent_steps=steps, minutes=round((time.time()-t0)/60, 2), episodes=len(recent),
                   strict=m('strict'), success=m('success'), removal=m('removal'), wall=m('wall'), t90=m('t90'), ret=m('ret'),
                   strict_lat3=m('strict', lambda e: e['latency'] == 3), strict_lat1=m('strict', lambda e: e['latency'] == 1),
                   strict_f01=m('strict', lambda e: e['flow'] == .1),
                   a_mean=act[sel].mean(0).tolist() if sel.any() else None, beta=beta, log_std=policy.log_std.tolist())
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
