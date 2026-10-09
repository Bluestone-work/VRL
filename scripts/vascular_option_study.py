"""EXP0060: matched measured-information option MAPPO/IPPO and fixed controls.

Truth is used exclusively by sensing simulation, reward and independent metrics.
Frozen preoperative proxy prevents accidental online truth access by controllers.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch

from marl.vascular_option_rl import OptionPolicy, OptionController, OBS_DIM, HISTORY, OPTIONS

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/validation/EXP0060_VASCULAR_OPTION_RL_20261006'
TRAIN_BASE, DEV_BASE = 2800000000, 2810000000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Episode:
    def __init__(self, anatomy, n, seed, sensing='noise', tpg=True, horizon=300.):
        from environments.mca_physical_env import DynamicsConfig
        from marl.teacher import TEACHER_CONFIG
        from marl.deployable_sensing import DeployableSensor
        from marl.multicluster import MultiClusterConfig
        from marl.tpg_coordinator import TPGCoordinator
        from scripts.multicluster_protocol import paired_environment, SpacingTracker, attach_spacing_monitor
        from scripts.safe_metrics import WallTracker
        import scripts.benchmark_multicluster as bm
        cfg = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy,
                      episode_duration_s=horizon, junction_model='union')
        self.env, self.manifest = paired_environment(cfg, n, seed)
        from scripts.multicluster_protocol import digest
        self.initial_state_hash = digest(dict(positions=self.env.positions_mm.tolist(),
                  masses=self.env.masses.tolist(), edges=self.env.edges.tolist(),
                  selected=self.manifest['selected_start_indices'], scene=self.manifest['scenario_hash']))
        self.n, self.seed, self.anatomy, self.sensing = n, seed, anatomy, sensing
        if sensing == 'image':
            from marl.image_sensing import ImageSensor
            self.sensor = ImageSensor(self.env, seed=seed)
        else:
            self.sensor = DeployableSensor(self.env, seed=seed)
        self.est = self.sensor.observe()
        env = self.env
        # Starts are map-matched from the first noisy measurement, not sim edge IDs.
        # Registered clot positions and healthy map are explicitly preoperative assumptions.
        preop = SimpleNamespace(tree=env.tree, transport=SimpleNamespace(points=env.transport.points.copy()),
                  config=env.config, num_robots=n, robot_stations=self.est.station.copy(),
                  clot_stations=env.clot_stations.copy(), clot_positions_mm=env.clot_positions_mm.copy())
        self.plan, _ = bm.preoperative_plan(preop)
        self.ctl = OptionController(preop, self.sensor)
        _, path = bm._station_paths(preop)
        # Fixed planned dwell avoids reading initial/remaining true clot masses for scheduling.
        self.coord = TPGCoordinator(preop, self.plan, path, dwell_s=15.) if tpg and n > 1 else None
        mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n,
                                min_spacing_mm=2. if n > 1 else 0.)
        self.shield = bm.Shield(mc, robot_speed_mm_s=cfg.robot_speed_mm_s, control_dt_s=cfg.control_dt_s)
        self.spacing = SpacingTracker(env.positions_mm[:n], env.active[:n], 2. if n > 1 else 0.)
        attach_spacing_monitor(env, self.spacing)
        self.walls = WallTracker(n); self.initial = float(env.initial_mass.sum())
        self.history = deque([np.zeros((3, OBS_DIM), np.float32) for _ in range(HISTORY)], maxlen=HISTORY)
        self.prev_local = np.zeros((n, 3)); self.pair = 0.; self.auc = 0.; self.t100 = None
        self.option_count = np.zeros(len(OPTIONS), int); self.switches = 0; self.last_option = np.zeros(n, int)
        self.holds = 0; self.info = None; self.done = False; self.trace = []
        self._prepare(); self._append_history()

    def _prepare(self):
        est = self.est
        self.targets = np.array([next((c for c in seq if est.clot_alive[c]),
                                seq[-1] if seq and est.clot_alive.any() else -1) for seq in self.plan])
        self.hold = self.coord.gate(est.pos, est.active) if self.coord is not None else np.zeros(self.n, bool)
        self.x, self.candidates = self.ctl.prepare(self.targets, est, self.hold,
                                                  self.env.elapsed_s, self.env.config.episode_duration_s)

    def _append_history(self):
        pad = np.zeros((3, OBS_DIM), np.float32); pad[:self.n] = self.x
        self.history.append(pad)

    def observation(self):
        active = np.zeros(3, np.float32); active[:self.n] = self.est.active
        return np.stack(self.history, axis=1), active

    def step(self, options, capture=False):
        from scripts.benchmark_deployable import packet_from
        env, n = self.env, self.n
        options = np.asarray(options[:n], int)
        self.option_count += np.bincount(options, minlength=len(OPTIONS))
        self.switches += int((options != self.last_option).sum()); self.last_option = options.copy()
        reward = np.zeros(3, np.float32)
        for _ in range(5):
            local = self.candidates[np.arange(n), options].copy()
            F = self.ctl.frames(self.est)
            if n > 1:
                local = self.shield.filtered(local, packet_from(self.est, F, self.prev_local))
            # Preserve mandatory holds even if the common spacing filter adds repulsion.
            local[self.hold] = 0.; self.holds += int(self.hold.sum())
            self.prev_local = local.copy()
            before = float(env.masses.sum()); active = env.active[:n].copy()
            prev_violation = self.spacing.pair_violation_s
            old_x = self.x.copy(); old_targets = self.targets.copy()
            if capture:
                self.trace.append(dict(time=float(env.elapsed_s), positions=env.positions_mm[:n].tolist(),
                                  particles=env.positions_mm[n:].tolist(), masses=env.masses.tolist(),
                                  estimates=self.est.pos.tolist(), options=options.tolist(), holds=self.hold.tolist()))
            self.spacing.begin_step()
            _, _, term, trunc, info = env.step(self.ctl.to_world(local, self.est))
            self.spacing.end_step(); self.info = info
            self.walls.update(info, active, info['step_duration_s']); self.pair += info['robot_pair_contact_s']
            removal = 1-float(info['remaining_mass'])/self.initial
            self.auc += .5*(1-before/self.initial+removal)*info['step_duration_s']
            if self.t100 is None and removal >= 1-1e-12:
                self.t100 = float(info['elapsed_s'])
            # Ground-truth training outcomes. Never part of actor/critic observations.
            r = (40.*(before-float(env.masses.sum()))/self.initial
                 -4.*np.asarray(info['wall_contact_s'])-8.*np.asarray(info['particle_collision_events'])
                 -4.*(self.spacing.pair_violation_s-prev_violation)-.002)
            r -= 5.*(active & ~env.active[:n])
            self.est = self.sensor.observe(); self._prepare()
            # Measured route progress only while target unchanged; clip noise-driven spikes.
            r += .2*np.clip((old_x[:, 14]-self.x[:, 14])*10., -.2, .2)*(old_targets == self.targets)
            reward[:n] += r*active
            self.done = bool(term or trunc)
            if self.done:
                safe = self.metrics()['cluster_safe_success']
                reward[:n] += (10. if safe else 2. if info['success'] else 0.)*active
                break
        self._append_history()
        return self.observation(), reward, self.done

    def metrics(self):
        from scripts.safe_metrics import episode_metrics
        m = episode_metrics(self.info, self.walls, self.initial); sp = self.spacing.summary()
        return dict(**m, cluster_safe_success=bool(m['safe_collision_free'] and sp['spacing_compliant'] and self.pair <= 1e-12),
                    scenario_hash=self.manifest['scenario_hash'], seed=self.seed, anatomy=self.anatomy, clusters=self.n,
                    initial_state_hash=self.initial_state_hash,
                    sensing=self.sensing, spacing=sp, particle_events=int(self.info['episode_particle_collision_events']),
                    lost=int(self.info['lost_robots']), robot_pair_contact_s=float(self.pair), t100_s=self.t100,
                    # Extend completed trajectories with their final removal to the fixed horizon.
                    removal_auc=(self.auc+(self.env.config.episode_duration_s-self.env.elapsed_s)*m['removal'])/self.env.config.episode_duration_s,
                    option_counts=self.option_count.tolist(), option_switches=self.switches, hold_control_steps=self.holds,
                    selected_start_indices=self.manifest['selected_start_indices'],
                    accepted_seed=self.manifest['accepted_seed'], rejected_attempts=self.manifest['rejected_attempts'])


def worker(conn, wid, seed, horizon):
    torch.set_num_threads(1)
    anatomies = json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']['train']
    rng = np.random.default_rng(seed*100+wid); count = 0
    def reset():
        nonlocal count
        scene_seed = TRAIN_BASE+seed*100000+wid*5000+count; count += 1
        if count >= 5000:
            raise RuntimeError('Registered worker seed block exhausted')
        return Episode(str(rng.choice(anatomies)), int(rng.integers(1, 4)), scene_seed, horizon=horizon)
    ep = reset(); conn.send(ep.observation())
    try:
        while True:
            action = conn.recv()
            if action is None: break
            ob, reward, done = ep.step(action); row = ep.metrics() if done else None
            if done:
                ep.env.close(); ep = reset(); ob = ep.observation()
            conn.send((ob, reward, done, row))
    finally:
        ep.env.close(); conn.close()


def advantages(reward, value, done, boot, gamma=.995, lam=.95):
    out = np.zeros_like(reward); g = np.zeros_like(boot)
    for t in reversed(range(len(reward))):
        mask = 1-done[t, :, None]
        nxt = boot if t == len(reward)-1 else value[t+1]
        g = reward[t]+gamma*nxt*mask-value[t]+gamma*lam*mask*g
        out[t] = g
    return out, out+value


def train(a):
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    memory, central = a.arm != 'ff_mappo', a.arm != 'memory_ippo'
    policy = OptionPolicy(memory, central); opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    config = dict(vars(a), memory=memory, central=central, options=OPTIONS, feature_dim=OBS_DIM,
                  history=HISTORY, decision_s=.5, source_sha={str(p.relative_to(ROOT)):sha(p) for p in
                     [ROOT/'marl/vascular_option_rl.py', Path(__file__)]})
    (out/'config.json').write_text(json.dumps(config, indent=2))
    ctx = mp.get_context('spawn'); pipes=[]; procs=[]
    for w in range(a.workers):
        parent, child = ctx.Pipe(); p = ctx.Process(target=worker, args=(child,w,a.seed,a.horizon))
        p.start(); child.close(); pipes.append(parent); procs.append(p)
    def receive(c):
        if not c.poll(180): raise TimeoutError('Worker did not respond within 180 s')
        return c.recv()
    t0=time.monotonic(); steps=0; iteration=0
    def checkpoint():
        payload=dict(state=policy.state_dict(), config=config, steps=steps, iterations=iteration)
        tmp=out/'checkpoint.tmp'; torch.save(payload,tmp); tmp.replace(out/'latest.pt')
    try:
        ob=[receive(c) for c in pipes]; checkpoint()
        torch.save(dict(state=policy.state_dict(),config=config,steps=0,iterations=0),out/'initial.pt')
        with (out/'training.jsonl').open('a') as log, (out/'episodes.jsonl').open('a') as elog:
            while time.monotonic()-t0 < a.minutes*60 and (not a.updates or iteration < a.updates):
                hist=[]; masks=[]; actions=[]; logps=[]; values=[]; rewards=[]; dones=[]; finished=[]
                for _ in range(a.rollout):
                    x=np.stack([o[0] for o in ob]); mask=np.stack([o[1] for o in ob])
                    with torch.no_grad():
                        d,v=policy(torch.from_numpy(x),torch.from_numpy(mask)); ac=d.sample(); lp=d.log_prob(ac)
                    for c, act in zip(pipes,ac.numpy()): c.send(act)
                    replies=[receive(c) for c in pipes]
                    hist.append(x); masks.append(mask); actions.append(ac.numpy()); logps.append(lp.numpy()); values.append(v.numpy())
                    rewards.append(np.stack([r[1] for r in replies])); dones.append(np.array([r[2] for r in replies],np.float32))
                    ob=[r[0] for r in replies]
                    for r in replies:
                        if r[3] is not None:
                            finished.append(r[3]); elog.write(json.dumps(r[3])+'\n')
                    steps += int(mask.sum())*5
                with torch.no_grad():
                    _,boot=policy(torch.from_numpy(np.stack([o[0] for o in ob])),torch.from_numpy(np.stack([o[1] for o in ob])))
                adv, ret=advantages(np.array(rewards),np.array(values),np.array(dones),boot.numpy())
                T=lambda z:torch.from_numpy(np.asarray(z))
                X=T(hist).flatten(0,1); M=T(masks).flatten(0,1); A=T(actions).flatten(0,1)
                LP=T(logps).flatten(0,1); AD=T(adv).flatten(0,1); RT=T(ret).flatten(0,1)
                valid=M.bool(); AD=(AD-AD[valid].mean())/(AD[valid].std()+1e-8)
                losses=[]; kls=[]
                for _ in range(4):
                    for ids in torch.randperm(len(X)).split(128):
                        d,v=policy(X[ids],M[ids]); lp=d.log_prob(A[ids]); ratio=(lp-LP[ids]).exp(); m=M[ids]
                        pi=-torch.minimum(ratio*AD[ids],ratio.clamp(.8,1.2)*AD[ids])
                        loss=((pi+.5*(v-RT[ids]).square()-.005*d.entropy())*m).sum()/m.sum().clamp(min=1)
                        opt.zero_grad(); loss.backward(); nnorm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5); opt.step()
                        losses.append(float(loss.detach())); kls.append(float((((ratio-1)-(lp-LP[ids]))*m).sum().detach()/m.sum().clamp(min=1)))
                iteration+=1; checkpoint()
                row=dict(iteration=iteration, agent_control_steps=steps, seconds=time.monotonic()-t0,
                         episodes=len(finished), safe=float(np.mean([r['cluster_safe_success'] for r in finished])) if finished else None,
                         removal=float(np.mean([r['removal'] for r in finished])) if finished else None,
                         loss=float(np.mean(losses)), approximate_kl=float(np.mean(kls)))
                log.write(json.dumps(row)+'\n'); log.flush(); elog.flush(); print(json.dumps(row),flush=True)
            checkpoint(); torch.save(dict(state=policy.state_dict(),config=config,steps=steps,iterations=iteration),out/'final.pt')
            (out/'DONE.json').write_text(json.dumps(dict(seconds=time.monotonic()-t0,steps=steps,checkpoint_sha256=sha(out/'final.pt'))))
    finally:
        for c in pipes:
            try:c.send(None)
            except (BrokenPipeError,EOFError):pass
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():p.terminate(); p.join()


def evaluate(a):
    torch.set_num_threads(1); policy=None
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu'); policy=OptionPolicy(ck['config']['memory'],ck['config']['central'])
        policy.load_state_dict(ck['state'],strict=True); policy.eval()
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    anatomy_order=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_order']
    anatomies=a.anatomies.split(',')
    with out.open('a') as f:
        for anatomy in anatomies:
            for scene in range(a.scenes):
                seed=DEV_BASE+anatomy_order.index(anatomy)*10000+scene
                ep=Episode(anatomy,a.clusters,seed,sensing=a.sensing,tpg=not a.no_tpg,horizon=a.horizon)
                while not ep.done:
                    x,m=ep.observation()
                    if policy is None:act=np.full(3,a.fixed_option,int)
                    else:
                        with torch.no_grad():d,_=policy(torch.from_numpy(x[None]),torch.from_numpy(m[None])); act=d.logits.argmax(-1)[0].numpy()
                    ep.step(act,capture=a.capture)
                row=ep.metrics(); row.update(method=a.arm,checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None)
                f.write(json.dumps(row)+'\n'); f.flush(); print(json.dumps(row),flush=True)
                if a.capture:
                    trace=out.with_name(f'{out.stem}_{anatomy}_{seed}.trace.json')
                    geometry=dict(points=ep.env.transport.points.tolist(), ends=ep.env.transport.ends.tolist(),
                                  healthy_radius=ep.env.flow_model.healthy_radius_mm.tolist(),
                                  clot_positions=ep.env.clot_positions_mm.tolist(), initial_mass=ep.env.initial_mass.tolist())
                    trace.write_text(json.dumps(dict(metrics=row,manifest=ep.manifest,trace=ep.trace,geometry=geometry),default=str))
                ep.env.close()


def main():
    # Existing repository runtime convention: cores 6/7 have recorded instability.
    # Preserve narrower caller affinity; children inherit the admitted cores.
    if hasattr(os, 'sched_getaffinity'):
        cpus=os.sched_getaffinity(0)-{6,7}
        if not cpus: raise RuntimeError('No admitted CPU cores in current affinity')
        os.sched_setaffinity(0,cpus)
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['train','eval'])
    p.add_argument('--out',required=True); p.add_argument('--arm',default='memory_mappo')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--minutes',type=float,default=60); p.add_argument('--updates',type=int,default=0)
    p.add_argument('--rollout',type=int,default=128); p.add_argument('--horizon',type=float,default=300.)
    p.add_argument('--checkpoint'); p.add_argument('--fixed-option',type=int,default=0)
    p.add_argument('--clusters',type=int,default=3); p.add_argument('--anatomies',default='mca_m1_lvo,ica_terminus_t,renal_artery')
    p.add_argument('--scenes',type=int,default=4); p.add_argument('--sensing',choices=['noise','image'],default='noise')
    p.add_argument('--no-tpg',action='store_true'); p.add_argument('--capture',action='store_true')
    a=p.parse_args(); train(a) if a.mode=='train' else evaluate(a)


if __name__=='__main__':main()
