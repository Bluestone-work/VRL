"""Train the image-conditioned residual PPO controller (marl.drl_local) inside the deployable pipeline.

Every episode runs exactly the benchmarked pipeline (allocation A -> TPG hold -> pursuit rule -> spacing
shield) with ImageSensor perception under per-episode camera randomisation (marl.image_sensing.random_camera)
and union-of-tubes physics; only the residual is learned. Training anatomies only (anatomy_holdout_v1.train),
N ~ U{1,2,3}, training seed range disjoint from the benchmark pools.
Reward per cluster and step (simulator outcomes are used only for the reward, never as policy input):
  +10 x team removal fraction increment
  -3 x own wall-contact seconds  -5 x own particle collision events  -10 x own particle contact seconds
  -0.5 if closer than d_min to a peer   -0.05 |residual|^2   -0.005
  terminal: +20 to every cluster on a safe completion, -10 to a cluster that is washed out (lost)
usage: train_drl_local.py --out DIR --minutes 180 --workers 6 --seed 0 [--reward v2 --res-scale 1.0 --init-std -1.0] [--no-image]
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


USE_IMAGE = {'on': True}
CFG = {'reward': 'v1', 'scale': .6}
# v2 (2026-10-06, after v1 plateaued at the rule's level: Safe 77.1 vs 77.1 % at N=3, particle events -24 %
# with CI spanning 0, wall contact slightly up). Particle collisions are rare (~0.5 per 2000-step episode),
# so a -5 event penalty is a sparse, noisy signal next to the dense removal reward. v2 adds the dense near-miss
# signal the simulator already reports (particle_near_s: time inside the 0.15 mm safety neighbourhood),
# strengthens event / contact / wall penalties and down-weights removal (progress is the rule's job).
REWARD = {'v1': dict(removal=10., wall=3., event=5., contact=10., near=0., spacing=.5, res=.05),
          'v2': dict(removal=3., wall=5., event=10., contact=20., near=4., spacing=.5, res=.02)}


def worker(wid, conn, seed0):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    from environments.mca_physical_env import DynamicsConfig
    from marl.deployable_sensing import DeployablePursuit
    from marl.drl_local import IRPolicy, compose, observe
    from marl.image_sensing import ImageSensor, random_camera
    from marl.multicluster import MultiClusterConfig
    from marl.teacher import TEACHER_CONFIG
    from marl.tpg_coordinator import TPGCoordinator
    import scripts.benchmark_multicluster as bm
    from scripts.benchmark_deployable import packet_from
    from scripts.multicluster_protocol import paired_environment
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed0); policy = IRPolicy(use_image=USE_IMAGE['on']); count = 0
    W = REWARD[CFG['reward']]

    def new_episode():
        nonlocal count
        while True:
            seed = seed0+count; count += 1
            an = train[rng.integers(len(train))]; n = int(rng.integers(1, 4))
            try:
                env, _ = paired_environment(replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=an,
                                                    episode_duration_s=300., junction_model='union'), n, seed)
                break
            except (ValueError, RuntimeError):
                continue
        sensor = ImageSensor(env, random_camera(rng), seed=seed)
        mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n,
                                min_spacing_mm=2. if n > 1 else 0.)
        plan, _ = bm.preoperative_plan(env)
        bm.FALLBACK['mode'] = 'park' if n > 1 else 'help'
        coord = None
        if n > 1:
            _, sp = bm._station_paths(env); coord = TPGCoordinator(env, plan, sp, d_min_mm=2.)
        return dict(env=env, n=n, sensor=sensor, ctl=DeployablePursuit(env, sensor), targets=bm.PlanTargets(plan),
                    coord=coord, shield=bm.Shield(mc, robot_speed_mm_s=env.config.robot_speed_mm_s,
                                                  control_dt_s=env.config.control_dt_s),
                    prev=np.zeros((n, 3)), anatomy=an, ret=0., wall=0., id=count, cur=None)

    def step_obs(ep):
        env = ep['env']; est = ep['sensor'].observe()
        bm.FALLBACK['mode'] = 'park' if ep['n'] > 1 else 'help'
        tgt = ep['targets'].targets(env, est.pos); rule = ep['ctl'].act(tgt, est)
        hold = ep['coord'].gate(est.pos, est.active) if ep['coord'] is not None else np.zeros(ep['n'], bool)
        img, vec, rule_w = observe(env, ep['sensor'], est, ep['ctl'], rule, hold, tgt)
        live = (np.asarray(tgt) >= 0) & est.active
        ep['cur'] = (est, tgt, hold, img, vec, rule_w, live)

    ep = new_episode(); step_obs(ep); finished = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        policy.load_state_dict(msg)
        buf = dict(img=[], vec=[], act=[], logp=[], val=[], rew=[], done=[], stream=[])
        for _ in range(ROLLOUT):
            env, n = ep['env'], ep['n']
            est, tgt, hold, img, vec, rule_w, live = ep['cur']
            with torch.no_grad():
                d, v = policy.dist(torch.as_tensor(img), torch.as_tensor(vec))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            a_np = a.numpy().astype(np.float64); a_np[~live] = 0.
            u = compose(rule_w, a_np, CFG['scale']); u[~live] = 0.
            F = ep['ctl'].frames(est); local = np.einsum('nij,nj->ni', F, u)
            local[hold] = 0.
            if n > 1:
                local = ep['shield'].filtered(local, packet_from(est, F, ep['prev']))
            ep['prev'] = local.copy()
            mass0 = float(env.masses.sum()); active = env.active[:n].copy()
            _, _, term, trunc, info = env.step(ep['ctl'].to_world(local, est))
            team = W['removal']*(mass0-float(env.masses.sum()))/float(env.initial_mass.sum())
            wall = np.asarray(info['wall_contact_s'], float); ep['wall'] += float(wall.sum())
            r = (team-W['wall']*wall-W['event']*np.asarray(info['particle_collision_events'], float)
                 -W['contact']*np.asarray(info['particle_contact_s'], float)-W['near']*np.asarray(info['particle_near_s'], float)
                 -W['res']*(np.clip(a_np, -1, 1)**2).sum(1)-.005)
            if n > 1:
                P = env.positions_mm[:n]; D = np.linalg.norm(P[:, None]-P[None], axis=-1)+np.eye(n)*99
                r -= W['spacing']*((D < 2.).any(1))
            done = bool(term or trunc)
            lost_now = active & ~env.active[:n]
            safe = bool(done and info['success'] and info['episode_particle_collision_events'] == 0 and ep['wall'] < 1.
                        and info['lost_robots'] == 0 and info['robot_pair_contact_s'] <= 1e-12)
            r = r-10.*lost_now+(20. if safe else 0.)
            for i in range(n):
                if active[i] and live[i]:
                    buf['img'].append(img[i]); buf['vec'].append(vec[i]); buf['act'].append(a[i].numpy())
                    buf['logp'].append(float(lp[i])); buf['val'].append(float(v[i])); buf['rew'].append(float(r[i]))
                    buf['done'].append(bool(done or not env.active[i])); buf['stream'].append((wid, ep['id'], i))
            ep['ret'] += float(r.sum())
            if done:
                finished.append(dict(anatomy=ep['anatomy'], n=n, safe=safe, success=bool(info['success']),
                                     removal=float(1-info['remaining_mass']/env.initial_mass.sum()),
                                     particle_events=int(info['episode_particle_collision_events']), ret=ep['ret']))
                env.close(); ep = new_episode()
            step_obs(ep)
        _, _, _, img, vec, _, _ = ep['cur']
        with torch.no_grad():
            _, vb = policy.dist(torch.as_tensor(img), torch.as_tensor(vec))
        conn.send(dict(buf=buf, boot={(wid, ep['id'], i): float(vb[i]) for i in range(ep['n'])}, finished=finished))
        finished = []


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
    from marl.drl_local import IRPolicy
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--minutes', type=float, default=150)
    p.add_argument('--workers', type=int, default=6); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--no-image', action='store_true')
    p.add_argument('--reward', default='v1', choices=sorted(REWARD)); p.add_argument('--res-scale', type=float, default=.6)
    p.add_argument('--init-std', type=float, default=-1.2)
    a = p.parse_args(); USE_IMAGE['on'] = not a.no_image; CFG['reward'], CFG['scale'] = a.reward, a.res_scale; a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed)
    (a.out/'config.json').write_text(json.dumps(vars(a), default=str))
    ctx = mp.get_context('fork'); pipes, procs = [], []
    for w in range(a.workers):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(w, child, 1313000000+a.seed*1000000+w*20000)); pr.start()
        pipes.append(parent); procs.append(pr)
    policy = IRPolicy(use_image=USE_IMAGE['on']).to(a.device)
    with torch.no_grad():
        policy.log_std.fill_(a.init_std); opt = torch.optim.Adam(policy.parameters(), 3e-4)
    t0, it, steps, log = time.time(), 0, 0, (a.out/'log.jsonl').open('a'); recent = []
    while time.time()-t0 < a.minutes*60:
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
        img, vec, act, lp0 = T(buf['img']), T(buf['vec']), T(buf['act']), T(buf['logp'])
        A, R = T(adv), T(ret); A = (A-A.mean())/(A.std()+1e-8)
        for _ in range(4):
            for b in torch.randperm(len(A), device=a.device).split(2048):
                d, v = policy.dist(img[b], vec[b]); lp = d.log_prob(act[b]).sum(-1); ratio = (lp-lp0[b]).exp()
                l_pi = -torch.min(ratio*A[b], ratio.clamp(.8, 1.2)*A[b]).mean()
                l = l_pi+.5*((v-R[b])**2).mean()-.001*d.entropy().sum(-1).mean()
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(policy.parameters(), .5); opt.step()
        it += 1; steps += len(A); recent = recent[-200:]
        row = dict(it=it, agent_steps=steps, minutes=round((time.time()-t0)/60, 2), episodes=len(recent),
                   safe=float(np.mean([e['safe'] for e in recent])) if recent else None,
                   success=float(np.mean([e['success'] for e in recent])) if recent else None,
                   particle_events=float(np.mean([e['particle_events'] for e in recent])) if recent else None,
                   log_std=policy.log_std.tolist())
        log.write(json.dumps(row)+'\n'); log.flush()
        if it % 10 == 0:
            torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps, use_image=USE_IMAGE['on'], res_scale=CFG['scale'], reward=CFG['reward']), a.out/'policy.pt')
        if it % 100 == 0:
            torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps, use_image=USE_IMAGE['on'], res_scale=CFG['scale'], reward=CFG['reward']), a.out/f'policy_it{it}.pt')
    torch.save(dict(state=policy.state_dict(), it=it, agent_steps=steps, use_image=USE_IMAGE['on'], res_scale=CFG['scale'], reward=CFG['reward']), a.out/'policy.pt')
    for c in pipes:
        c.send(None)
    for pr in procs:
        pr.join(timeout=10)


if __name__ == '__main__':
    main()
