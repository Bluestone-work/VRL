"""PPO training of the temporal obstacle-avoidance policy (marl.obstacle_control.TemporalPolicy) inside the
deployable v3 pipeline (allocation A -> TPG hold -> route pursuit (+APF) -> policy -> spacing shield).

Variants (identical budget, environment and reward; only the named factor changes)
  --arch transformer --residual   T-IRPPO (ours): causal Transformer over a 16-step memory, residual on APF rule
  --arch gru / mlp                temporal-architecture ablations
  --direct                        no rule prior: the policy outputs the command itself (Turbo-style from scratch)
  --no-dr                         no multi-level domain randomisation
Multi-level domain randomisation (cf. Turbo, Supplementary Table 4), per episode
  environment  obstacle number U[12,24], static share U[0.1,0.9], sizes, fragment speeds (marl.obstacle_field)
  perception   cluster position noise U[0.02,0.08] mm, latency U{1,2} steps, dropout U[0,0.05]
  actuation    command gain U[0.75,1.25] per cluster, execution noise N(0, 0.025)
Reward per cluster and step (simulator outcomes; never a policy input)
  +3 x team removal fraction  +0.3 x decrease of distance to own target (mm)  -10 x wall-contact s
  -5 x obstacle near-miss s (0.15 mm margin)  -0.5 if closer than d_min to a peer  -0.01 |a|^2 (residual)
  (v3.1, after round 1 learned a near-zero residual: wall direction added to the token, wall / near-miss
  weights raised, dwell steps without nearby obstacles skipped, learning-rate floor 1e-4)
  v3.2 (--reward local, default): the policy only changes local behaviour, so the reward keeps only local
  terms on a short horizon (gamma 0.95, lambda 0.9): +1 x progress along the pre-operative route (mm, clipped
  to 0.2 per step), -10 x wall-contact s, -5 x obstacle near-miss s, -0.5 spacing, -0.01|a|^2, lost -10;
  v3.3: collisions -10 per new event and -20 per contact second WITHOUT ending the episode (v3.2's terminal
  collision was exploited: with negative step rewards an early collision shortened the penalty stream). Removal / completion bonuses are dropped (the rule and TPG handle completion).
  --reward team restores v3.1.
  v3.4: obstacle slots as Turbo (relative position, size, gap; no velocity, --obs-vel restores it as an
  ablation) and +0.05 safe-distance step reward (no wall contact, all obstacles beyond the margin).
  -0.005; terminal: obstacle collision -20 and the episode ends (safety violation, as Turbo);
  cluster lost -10; safe completion +20 to every cluster
Training anatomies only (anatomy_holdout_v1.train), N ~ U{1,2,3}, seeds disjoint from the benchmark pools.
usage: train_obstacle_drl.py --out DIR --arch transformer --minutes 100 --workers 4 --seed 0
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
CFG = {}


def worker(wid, conn, seed0, cfg):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    from marl.deployable_sensing import DeployableConfig
    from marl.obstacle_control import History, TemporalPolicy, compose, token, token_dim
    from scripts.benchmark_obstacles import Episode
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); dim = token_dim(cfg['obs_vel']); policy = TemporalPolicy(cfg['arch'], dim=dim); count = 0

    def new_episode():
        nonlocal count
        while True:
            seed = seed0+count; count += 1
            an = train[rng.integers(len(train))]; n = int(rng.integers(1, 4))
            sc = (DeployableConfig(position_sigma_mm=float(rng.uniform(.02, .08)), latency_steps=int(rng.integers(1, 3)),
                                   dropout_prob=float(rng.uniform(0, .05))) if cfg['dr'] else DeployableConfig())
            try:
                ep = Episode(n, an, seed, sense_cfg=sc)
                break
            except (ValueError, RuntimeError):
                continue
        ep.gain = rng.uniform(.75, 1.25, n) if cfg['dr'] else np.ones(n)
        ep.hist = History(n, dim=dim); ep.prev = np.zeros((n, 3)); ep.id = count; ep.ret = 0.; ep.anatomy = an
        ep.dist0 = None
        return ep

    def obs(ep):
        est, tgt, rule, hold = ep.observe()
        T, rule_w = token(ep.env, ep.sensor, est, ep.ctl, rule, hold, tgt, ep.prev, cfg['obs_vel'])
        seq, mask = ep.hist.push(T)
        live = (np.asarray(tgt) >= 0) & est.active
        ep.cur = (est, tgt, rule, hold, rule_w, seq, mask, live)

    ep = new_episode(); obs(ep); finished = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg)
        buf = dict(seq=[], mask=[], act=[], logp=[], val=[], rew=[], done=[], stream=[])
        for _ in range(ROLLOUT):
            env, n = ep.env, ep.n
            est, tgt, rule, hold, rule_w, seq, mask, live = ep.cur
            with torch.no_grad():
                d, v = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            a_np = a.numpy().astype(np.float64)
            u = compose(rule_w, a_np, cfg['residual'], cfg['scale']); u[~live] = 0.
            ep.prev = u.copy()
            F = ep.ctl.frames(est); local = np.einsum('nij,nj->ni', F, u)
            if cfg['dr']:                                       # actuation-level randomisation
                local = local*ep.gain[:, None]+np.random.default_rng().normal(0., .025, local.shape)*(np.abs(local).sum(1, keepdims=True) > 0)
            P0 = env.positions_mm[:n].copy()
            d0 = np.array([np.linalg.norm(env.clot_positions_mm[t]-P0[i]) if t >= 0 else 0. for i, t in enumerate(tgt)])
            g0 = np.array([route_remaining(ep.ctl, i, P0[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            done, out = ep.step(est, local, hold)
            P1 = env.positions_mm[:n]
            g1 = np.array([route_remaining(ep.ctl, i, P1[i]) if tgt[i] >= 0 else 0. for i in range(n)])
            if cfg['reward'] == 'local':
                # v3.2: only what the local decision changes, on a ~2 s horizon (gamma 0.95): progress along
                # the pre-operative route, wall contact, obstacle near-miss; collision ends the episode
                prog = np.clip(g0-g1, -.2, .2)*(np.asarray(tgt) >= 0)
                r = 1.*prog-10.*out['wall']-5.*out['obs_near']
            else:
                d1 = np.array([np.linalg.norm(env.clot_positions_mm[t]-P1[i]) if t >= 0 else 0. for i, t in enumerate(tgt)])
                r = (3.*out['removed']+.3*(d0-d1)*(np.asarray(tgt) >= 0)-10.*out['wall']-5.*out['obs_near']-.005)
            if cfg['residual']:
                r -= .01*(np.clip(a_np, -1, 1)**2).sum(1)
            if n > 1:
                D = np.linalg.norm(P1[:, None]-P1[None], axis=-1)+np.eye(n)*99
                r -= .5*((D < 2.).any(1))
            collided = out['obs_events'] > 0
            local_r = cfg['reward'] == 'local'
            if local_r:
                # v3.3: no termination on collision. With only negative per-step terms, ending the episode by a
                # collision was cheaper than continuing (v3.2 pilot: return rose -280 -> -65 while collisions
                # rose 14 -> 36 %). Each new collision costs -10 and every second in contact -20; the episode
                # continues, so a collision never shortens the penalty stream.
                r -= 10.*out['obs_events']+20.*out['obs_contact']+10.*out['lost']
                # v3.4: positive safe-distance step reward (Turbo: +0.5 safe distance, scaled): no wall contact and
                # every obstacle outside the 0.15 mm safety margin
                r += .05*((out['wall'] <= 0) & (out['obs_near'] <= 0) & (np.asarray(tgt) >= 0))
            else:
                r -= 20.*collided+10.*out['lost']
                done = done or bool(collided.any())
            safe = done and not collided.any() and ep.info['success'] and ep.safe()
            if safe and not local_r:
                r += 20.
            # informative samples only: dwelling on its own clot with no obstacle within 1 mm carries no
            # decision (the rule stops there); those steps are skipped instead of diluting the batch (v3.1)
            busy = np.array([tgt[i] < 0 or d0[i] > .15 or any(np.linalg.norm(rel)-rr < 1. for rel, _, rr in est.obstacles[i])
                             for i in range(n)])
            for i in range(n):
                if out['active_before'][i] and live[i] and (busy[i] or done or not env.active[i]):
                    buf['seq'].append(seq[i]); buf['mask'].append(mask[i]); buf['act'].append(a[i].numpy())
                    buf['logp'].append(float(lp[i])); buf['val'].append(float(v[i])); buf['rew'].append(float(r[i]))
                    buf['done'].append(bool(done or not env.active[i])); buf['stream'].append((wid, ep.id, i))
            ep.ret += float(r.sum())
            if done:
                finished.append(dict(anatomy=ep.anatomy, n=n, safe=bool(safe), collided=bool(collided.any() or ep.field.events.sum() > 0),
                                     success=bool(ep.info['success']), t=float(ep.info['elapsed_s']), ret=ep.ret))
                ep.close(); ep = new_episode()
            obs(ep)
        _, _, _, _, _, seq, mask, _ = ep.cur
        with torch.no_grad():
            _, vb = policy.dist(torch.as_tensor(seq), torch.as_tensor(mask))
        conn.send(dict(buf=buf, boot={(wid, ep.id, i): float(vb[i]) for i in range(ep.n)}, finished=finished)); finished = []


def route_remaining(ctl, i, pos):
    """Remaining pre-operative route length to the cluster's target (mm), from its current route progress."""
    R = ctl.route[i]
    if R is None:
        return 0.
    P = ctl.pts[R]; k = int(ctl.prog[i])
    seg = np.linalg.norm(np.diff(P[k:], axis=0), axis=1).sum() if k < len(P)-1 else 0.
    return float(seg+np.linalg.norm(pos-P[k]))


def gae(buf, boot, gamma=.995, lam=.95):
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
    from marl.obstacle_control import TemporalPolicy, token_dim
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=100)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arch', default='transformer', choices=('transformer', 'gru', 'mlp'))
    p.add_argument('--direct', action='store_true'); p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--no-dr', action='store_true'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--reward', default='local', choices=('local', 'team')); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--init-std', type=float, default=-1.2)
    p.add_argument('--obs-vel', action='store_true', help='ablation: give finite-difference obstacle velocities')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    cfg = dict(arch=a.arch, residual=not a.direct, scale=a.scale, dr=not a.no_dr, reward=a.reward, obs_vel=a.obs_vel)
    (a.out/'config.json').write_text(json.dumps(dict(vars(a), **cfg), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(w, child, 1515000000+a.seed*1000000+w*20000, cfg)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = TemporalPolicy(a.arch, dim=token_dim(a.obs_vel)).to(a.device)
    with torch.no_grad():
        policy.log_std.fill_(-.5 if a.direct else a.init_std)
    opt = torch.optim.AdamW(policy.parameters(), 3e-4, betas=(.9, .98), weight_decay=.01)
    t0, it, steps, log = time.time(), 0, 0, (a.out/'log.jsonl').open('a'); recent = []
    save = lambda path: torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps, cfg=cfg), path)
    while time.time()-t0 < a.minutes*60:
        frac = min((time.time()-t0)/(a.minutes*60), 1.)
        cos = .5*(1+math.cos(math.pi*frac))                       # cosine schedules (Turbo)
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
        adv, ret = gae(buf, boot, *((.95, .9) if cfg['reward'] == 'local' else (.995, .95)))
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
        row = dict(it=it, agent_steps=steps, minutes=round((time.time()-t0)/60, 2), episodes=len(recent),
                   safe=float(np.mean([e['safe'] for e in recent])) if recent else None,
                   collided=float(np.mean([e['collided'] for e in recent])) if recent else None,
                   success=float(np.mean([e['success'] for e in recent])) if recent else None,
                   ret=float(np.mean([e['ret'] for e in recent])) if recent else None,
                   lr=opt.param_groups[0]['lr'], clip=clip, log_std=policy.log_std.tolist())
        log.write(json.dumps(row)+'\n'); log.flush()
        if it % 10 == 0:
            save(a.out/'policy.pt')
        if it % 200 == 0:
            save(a.out/f'policy_it{it}.pt')
    save(a.out/'policy.pt')
    for c in pipes:
        c.send(None)
    for pr in procs:
        pr.join(timeout=10)


if __name__ == '__main__':
    main()
