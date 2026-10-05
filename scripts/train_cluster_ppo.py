"""Pure-RL baseline for the cluster benchmark: parameter-shared PPO (IPPO) from scratch.

Same information as the fair local methods: 111-d fair observation + allocation-A target slot.
Random training anatomy (anatomy_holdout_v1.train) and N in {1,2,3} per episode, registered teacher/
training seed range. Reward per cluster (fair quantities for shaping, simulator truth for outcomes):
  +10 * team removal increment, +0.3 * decrease of straight-line distance to its own target (mm),
  -1 * own wall-contact seconds, -5 * own particle collision events, -0.5 per step closer than d_min
  to a peer, -0.01 per step, +10 team bonus on cluster-safe completion.
Policy: MLP (marl.local_student architecture), Gaussian over the Frenet command, commands shorter than
0.35 execute as a stop. Workers simulate on CPU; the update runs on the GPU.
usage: train_cluster_ppo.py --out DIR --minutes 60 --workers 16
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROLLOUT = 256


class Policy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        from marl.partial_obs import OBS_DIM, CLOT_SLOTS
        self.k = CLOT_SLOTS
        def mlp(out):
            return nn.Sequential(nn.Linear(OBS_DIM+CLOT_SLOTS, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, out))
        self.pi, self.v = mlp(3), mlp(1)
        self.log_std = nn.Parameter(torch.full((3,), -.5))

    def inp(self, nav, slot):
        oh = nn.functional.one_hot(slot.clamp(min=0), self.k).float()*(slot >= 0)[..., None]
        return torch.cat((nav, oh), -1)

    def dist(self, nav, slot):
        x = self.inp(nav, slot)
        return torch.distributions.Normal(self.pi(x), self.log_std.exp()), self.v(x).squeeze(-1)


def execute(a):
    a = np.clip(a, -1, 1); n = np.linalg.norm(a, axis=-1, keepdims=True)
    a = a/np.maximum(n, 1.)
    a[np.linalg.norm(a, axis=-1) < .35] = 0.
    return a


MODE = {'obs': 'fair', 'junction': 'graph'}


def worker(wid, conn, seed0):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    from environments.mca_physical_env import DynamicsConfig
    from marl.multicluster import MultiClusterConfig
    from marl.multicluster_observation import ClusterSensorAdapter
    from marl.teacher import TEACHER_CONFIG
    import scripts.benchmark_multicluster as bm
    reg = json.load(open('configs/evaluation_splits.json')); train = reg['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); policy = Policy(); episode = 0
    def new_episode():
        nonlocal episode
        while True:
            seed = seed0+episode; episode += 1
            a = train[rng.integers(len(train))]; n = int(rng.integers(1, 4))
            try:
                env, _ = bm.paired_environment(replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=a, episode_duration_s=300.,
                                                       junction_model=MODE['junction']), n, seed)
                break
            except (ValueError, RuntimeError):
                continue
        mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n, min_spacing_mm=2. if n > 1 else 0.)
        sensor = ClusterSensorAdapter(env, mc); sensor.reset(seed)
        plan, _ = bm.preoperative_plan(env)
        ep = dict(env=env, n=n, sensor=sensor, targets=bm.PlanTargets(plan), ret=0., anatomy=a)
        if MODE['obs'] == 'deployable':
            from marl.deployable_sensing import DeployableObserver, DeployableSensor
            ds = DeployableSensor(env, seed=seed); ep['dsensor'] = ds; ep['dobs'] = DeployableObserver(env, ds)
        return ep
    ep = new_episode(); finished = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg)
        buf = dict(nav=[], slot=[], act=[], logp=[], val=[], rew=[], done=[], agent=[], stream=[])
        for _ in range(ROLLOUT):
            env, n = ep['env'], ep['n']
            nav, slot, tgt, est = observe(ep)
            with torch.no_grad():
                d, v = policy.dist(torch.as_tensor(nav), torch.as_tensor(slot, dtype=torch.long))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            act = execute(a.numpy().astype(np.float64)); act[slot < 0] = 0.
            mass0 = float(env.masses.sum()); P0 = env.positions_mm[:n].copy()
            d0 = np.array([np.linalg.norm(env.clot_positions_mm[t]-P0[i]) if t >= 0 else 0. for i, t in enumerate(tgt)])
            active = env.active[:n].copy()
            _, _, term, trunc, info = env.step(execute_cmd(ep, act, est))
            P1 = env.positions_mm[:n]
            d1 = np.array([np.linalg.norm(env.clot_positions_mm[t]-P1[i]) if t >= 0 else 0. for i, t in enumerate(tgt)])
            team = 10.*(mass0-float(env.masses.sum()))/float(env.initial_mass.sum())
            r = team+.3*(d0-d1)*(tgt >= 0)-np.asarray(info['wall_contact_s'])-5.*np.asarray(info['particle_collision_events'])-.01
            if n > 1:
                D = np.linalg.norm(P1[:, None]-P1[None], axis=-1)+np.eye(n)*99
                r -= .5*((D < 2.).any(1))
            done = term or trunc
            if done and info['success']:
                r = r+10.
            r = r*active
            for i in range(n):
                if active[i]:
                    buf['nav'].append(nav[i]); buf['slot'].append(slot[i]); buf['act'].append(a[i].numpy()); buf['logp'].append(float(lp[i]))
                    buf['val'].append(float(v[i])); buf['rew'].append(float(r[i])); buf['done'].append(bool(done or not env.active[i]))
                    buf['stream'].append((wid, episode, i))
            ep['ret'] += float(r.sum())
            if done:
                finished.append(dict(anatomy=ep['anatomy'], n=n, success=bool(info['success']), removal=float(1-info['remaining_mass']/env.initial_mass.sum()), ret=ep['ret']))
                env.close(); ep = new_episode()
        # bootstrap values for unfinished streams
        env, n = ep['env'], ep['n']
        nav, slot, _, _ = observe(ep)
        with torch.no_grad():
            _, vb = policy.dist(torch.as_tensor(nav), torch.as_tensor(slot, dtype=torch.long))
        conn.send(dict(buf=buf, boot={(wid, episode, i): float(vb[i]) for i in range(n)}, finished=finished)); finished = []


def observe(ep):
    """(nav [n,111], target slot [n], targets [n], estimate or None) for the configured information model."""
    import scripts.benchmark_multicluster as bm
    env, n = ep['env'], ep['n']
    if 'dobs' in ep:
        est = ep['dsensor'].observe()
        tgt = ep['targets'].targets(env, est.pos)
        nav, ids = ep['dobs'].observe(est)
        slot = np.array([int(np.flatnonzero(ids[i] == t)[0]) if t >= 0 and (ids[i] == t).any() else -1 for i, t in enumerate(tgt)])
        return nav.astype(np.float32), slot, tgt, est
    pk = ep['sensor'].observe(); tgt = ep['targets'].targets(env, env.positions_mm[:n])
    return pk.navigation.astype(np.float32), bm.slots_for(pk, tgt), tgt, None


def execute_cmd(ep, act, est):
    if est is None:
        return ep['sensor'].execute(act)
    ep['dobs'].record(act)
    return np.einsum('nji,nj->ni', ep['dobs'].frames(est), act)*est.active[:, None]


def gae(buf, boot, gamma=.99, lam=.95):
    idx = {}
    for k, s in enumerate(buf['stream']):
        idx.setdefault(s, []).append(k)
    adv = np.zeros(len(buf['rew'])); ret = np.zeros(len(buf['rew']))
    for s, ks in idx.items():
        last_v = 0. if buf['done'][ks[-1]] else boot.get(s, 0.); g = 0.
        for k in reversed(ks):
            nv = last_v
            delta = buf['rew'][k]+gamma*nv*(1-buf['done'][k])-buf['val'][k]
            g = delta+gamma*lam*(1-buf['done'][k])*g
            adv[k] = g; ret[k] = g+buf['val'][k]; last_v = buf['val'][k]
    return adv, ret


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=60)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cuda:0')
    p.add_argument('--obs', default='fair', choices=('fair', 'deployable')); p.add_argument('--junction', default='graph', choices=('graph', 'union'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    MODE['obs'], MODE['junction'] = a.obs, a.junction
    (a.out/'config.json').write_text(json.dumps(vars(a), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe(); pr = ctx.Process(target=worker, args=(w, child, 1112000000+a.seed*1000000+w*20000)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = Policy().to(a.device); opt = torch.optim.Adam(policy.parameters(), 3e-4)
    t0, it, steps, log = time.time(), 0, 0, (a.out/'log.jsonl').open('a'); recent = []
    while time.time()-t0 < a.minutes*60:
        sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        for c in pipes:
            c.send(sd)
        data = [c.recv() for c in pipes]
        buf = {k: sum((d['buf'][k] for d in data), []) for k in data[0]['buf']}
        boot = {}; [boot.update(d['boot']) for d in data]; [recent.extend(d['finished']) for d in data]
        adv, ret = gae(buf, boot)
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=a.device)
        nav, slot, act, lp0 = T(buf['nav']), T(buf['slot'], torch.long), T(buf['act']), T(buf['logp'])
        A, R = T(adv), T(ret); A = (A-A.mean())/(A.std()+1e-8)
        for _ in range(4):
            for b in torch.randperm(len(A), device=a.device).split(4096):
                d, v = policy.dist(nav[b], slot[b]); lp = d.log_prob(act[b]).sum(-1); ratio = (lp-lp0[b]).exp()
                l_pi = -torch.min(ratio*A[b], ratio.clamp(.8, 1.2)*A[b]).mean()
                l = l_pi+.5*((v-R[b])**2).mean()-.003*d.entropy().sum(-1).mean()
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(policy.parameters(), .5); opt.step()
        it += 1; steps += len(A); recent = recent[-200:]
        row = dict(it=it, agent_steps=steps, minutes=round((time.time()-t0)/60, 2), episodes=len(recent),
                   success=float(np.mean([e['success'] for e in recent])) if recent else None,
                   removal=float(np.mean([e['removal'] for e in recent])) if recent else None, log_std=policy.log_std.tolist())
        log.write(json.dumps(row)+'\n'); log.flush()
        if it % 20 == 0:
            torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps), a.out/'policy.pt')
    torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps), a.out/'policy.pt')
    for c in pipes:
        c.send(None)
    for pr in procs:
        pr.join(timeout=10)


if __name__ == '__main__':
    main()
