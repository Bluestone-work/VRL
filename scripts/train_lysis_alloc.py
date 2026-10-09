"""MAPPO training of the cooperative clot allocator (marl.lysis_alloc.AllocPolicy) in benchmark v4.

Pipeline per episode: learned allocator (event-triggered, SMDP) -> dynamic TPG -> topology-fixed pursuit + wall
guard (classical low level) -> spacing shield -> physics. Image sensing; training anatomies only
(anatomy_holdout_v1.train); N ~ {1: .2, 2: .4, 3: .4}; seeds 1616000000+ (disjoint from the benchmark pools).
Domain randomisation: per-cluster actuation gain U[0.8, 1.2] and execution noise N(0, 0.02).
Team reward per control step (simulator truth, never an input):
  +10 x removal increment  -0.3 x wall-contact robot-s  -2 x spacing-violation pair-s  -0.02 x dt while clots remain
  +5 when all clots are cleared
SMDP: each agent's transition spans its two consecutive decisions; reward discounted per step (gamma 0.999,
i.e. 0.99 per s), GAE lambda 0.95 over decisions. Shared actor (own observation), centralised critic (team).
usage: train_lysis_alloc.py --out DIR --minutes 90 --workers 12 --seed 0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

GAMMA, LAM = .999, .95


def worker(wid, conn, seed0, cfg):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    from marl.lysis_alloc import AllocPolicy, LearnedAllocator
    from scripts.benchmark_lysis import LysisEpisode, WallGuard
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); count = 0; policy = AllocPolicy()
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg['state']); trans, stats = [], []
        for _ in range(msg['episodes']):
            while True:
                seed = seed0+count; count += 1
                an = train[rng.integers(len(train))]; n = int(rng.choice([1, 2, 3], p=cfg['p_n']))
                try:
                    ep = LysisEpisode(n, an, seed, horizon=cfg['horizon'], tpg=False,
                                      actuation=dict(gain=(.8, 1.2), noise=.02) if cfg['dr'] else None, rng=rng)
                    break
                except (ValueError, RuntimeError):
                    continue
            alloc = LearnedAllocator(ep, policy, greedy=False); low = WallGuard(ep)
            rew = []; viol0 = 0.
            while True:
                est = ep.observe(); tgt = alloc(ep, est); rule = ep.ctl.act(tgt, est); hold = alloc.hold(ep, est)
                local = low(ep, est, tgt, rule, hold)
                alloc.observe_motion(est, np.where(hold[:, None], 0., local))
                done, out = ep.step(est, local, hold)
                viol = ep.spacing.summary()['spacing_violation_pair_s']
                dt = float(ep.info['step_duration_s'])
                r = 10.*out['removed']-.3*float(out['wall'].sum())-2.*(viol-viol0)-.02*dt*float(ep.env.masses.sum() > 0)
                viol0 = viol
                if done and ep.info['success']:
                    r += 5.
                rew.append(r)
                if done:
                    break
            T = len(rew); rew = np.asarray(rew); disc = GAMMA**np.arange(T+1)
            csum = np.concatenate([[0.], np.cumsum(rew*disc[:T])])
            by = {}
            for d in alloc.decisions:
                by.setdefault(d['agent'], []).append(d)
            for i, ds in by.items():
                for k, d in enumerate(ds):
                    s0 = d['step']; s1 = ds[k+1]['step'] if k+1 < len(ds) else T
                    R = (csum[s1]-csum[s0])/disc[s0]
                    trans.append(dict(x=d['x'], c=d['c'], mask=d['mask'], prior=d['prior'], a=d['a'], logp=d['logp'], v=d['v'],
                                      R=R, g=GAMMA**(s1-s0), last=k+1 == len(ds), stream=(wid, seed, i), k=k))
            row = ep.row('train')
            stats.append(dict(n=n, anatomy=an, removal=row['removal'], cleared=row['task_success'], auc=row['removal_auc'],
                              t100=row['t100_s'], wall=row['wall_contact_s'], viol=row['spacing_violation_pair_s'],
                              ret=float(rew.sum()), decisions=len(alloc.decisions),
                              deviate=float(np.mean([d['a'] != d['prior'] for d in alloc.decisions])) if alloc.decisions else 0.))
            ep.close()
        conn.send(dict(trans=trans, stats=stats))


def gae(trans):
    """GAE over each agent's decision chain; last decision of an episode is terminal (time is in the input)."""
    adv = np.zeros(len(trans)); ret = np.zeros(len(trans)); chains = {}
    for j, t in enumerate(trans):
        chains.setdefault(t['stream'], []).append(j)
    for js in chains.values():
        js = sorted(js, key=lambda j: trans[j]['k']); g = 0.; nv = 0.
        for j in reversed(js):
            t = trans[j]; nv = 0. if t['last'] else nv
            delta = t['R']+t['g']*nv-t['v']
            g = delta+(0. if t['last'] else t['g']*LAM*g)
            adv[j] = g; ret[j] = g+t['v']; nv = t['v']
    return adv, ret


def main():
    from marl.lysis_alloc import AllocPolicy
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=90)
    p.add_argument('--workers', type=int, default=12); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=2, help='episodes per worker per iteration')
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--ent', type=float, default=.01)
    p.add_argument('--prior-w', type=float, default=4.); p.add_argument('--no-dr', action='store_true')
    p.add_argument('--horizon', type=float, default=300.)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    cfg = dict(p_n=[.2, .4, .4], horizon=a.horizon, dr=not a.no_dr)
    (a.out/'config.json').write_text(json.dumps(dict(vars(a), **cfg), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(w, child, 1616000000+a.seed*10000000+w*200000, cfg)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = AllocPolicy(prior_w=a.prior_w); opt = torch.optim.Adam(policy.parameters(), a.lr, eps=1e-5)
    t0, it, log = time.time(), 0, (a.out/'log.jsonl').open('a'); n_trans = 0
    save = lambda path: torch.save(dict(state=policy.state_dict(), it=it, transitions=n_trans, cfg=cfg), path)
    save(a.out/'policy_it0.pt')
    while time.time()-t0 < a.minutes*60:
        frac = min((time.time()-t0)/(a.minutes*60), 1.)
        for g in opt.param_groups:
            g['lr'] = a.lr*(1-.9*frac)
        sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        for c in pipes:
            c.send(dict(state=sd, episodes=a.episodes))
        data = [c.recv() for c in pipes]
        trans = sum((d['trans'] for d in data), []); stats = sum((d['stats'] for d in data), [])
        if not trans:
            continue
        adv, ret = gae(trans)
        X = torch.as_tensor(np.stack([t['x'] for t in trans])); C = torch.as_tensor(np.stack([t['c'] for t in trans]))
        Mk = torch.as_tensor(np.stack([t['mask'] for t in trans])); Pr = torch.as_tensor([t['prior'] for t in trans])
        A_ = torch.as_tensor([t['a'] for t in trans]); LP = torch.as_tensor([t['logp'] for t in trans], dtype=torch.float32)
        ADV = torch.as_tensor(adv, dtype=torch.float32); RET = torch.as_tensor(ret, dtype=torch.float32)
        ADV = (ADV-ADV.mean())/(ADV.std()+1e-8)
        for _ in range(4):
            for b in torch.randperm(len(trans)).split(512):
                lg = policy.logits(X[b], Mk[b], Pr[b]); d = torch.distributions.Categorical(logits=lg)
                ratio = (d.log_prob(A_[b])-LP[b]).exp()
                l_pi = -torch.min(ratio*ADV[b], ratio.clamp(.8, 1.2)*ADV[b]).mean()
                l_v = ((policy.value(C[b])-RET[b])**2).mean()
                loss = l_pi+.5*l_v-a.ent*d.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), .5); opt.step()
        it += 1; n_trans += len(trans)
        f = lambda k: float(np.mean([s[k] for s in stats]))
        multi = [s for s in stats if s['n'] > 1]
        row = dict(it=it, minutes=round((time.time()-t0)/60, 2), transitions=n_trans, episodes=len(stats),
                   removal=f('removal'), cleared=f('cleared'), auc=f('auc'), wall=f('wall'), viol=f('viol'), ret=f('ret'),
                   deviate=f('deviate'), auc_multi=float(np.mean([s['auc'] for s in multi])) if multi else None,
                   prior_w=float(policy.prior_w), l_v=float(l_v), lr=opt.param_groups[0]['lr'])
        log.write(json.dumps(row)+'\n'); log.flush()
        if it % 5 == 0:
            save(a.out/'policy.pt')
        if it % 25 == 0:
            save(a.out/f'policy_it{it}.pt')
    save(a.out/'policy.pt')
    for c in pipes:
        c.send(None)
    for pr in procs:
        pr.join(timeout=10)


if __name__ == '__main__':
    main()
