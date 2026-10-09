"""Benchmark v3: multi-cluster thrombus removal with microscope-visible static/dynamic obstacles.

Same scenes (paired_environment), physics (union-of-tubes), allocation A, TPG coordination (N > 1) and
spacing shield as benchmark v2; debris tracers are removed (particle_count = 0) and replaced by the
obstacle field of marl.obstacle_field (paired: the field depends only on the scene seed).
Methods
  rule_noavoid     pursuit on the pre-operative route, no obstacle avoidance        (lower reference)
  rule_apf         pursuit + artificial potential field                              (classical baseline)
  drl              TemporalPolicy checkpoint (--ckpt): mlp / gru / transformer, residual or direct
  student          deployable discrete option student checkpoint (--ckpt)
  teacher          privileged lookahead teacher (training/evaluation upper bound)
Safe Success (v3): all clots cleared, wall contact < 1 robot-s, no obstacle collision, no cluster lost,
no cluster-cluster contact, spacing compliant (N > 1).
usage: benchmark_obstacles.py --method M --clusters N --anatomy A --seeds S0:S1 --out JSONL [--ckpt C]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.deployable_sensing import DeployableConfig, DeployableSensor
from marl.multicluster import MultiClusterConfig
from marl.obstacle_control import APFPursuit, ObstacleTracker
from marl.obstacle_field import ObstacleField
from marl.teacher import TEACHER_CONFIG
import scripts.benchmark_multicluster as bm
from scripts.benchmark_deployable import packet_from
from scripts.multicluster_protocol import SpacingTracker, attach_spacing_monitor, paired_environment
from scripts.safe_metrics import WallTracker, episode_metrics


class Episode:
    """One episode of the v3 pipeline. `policy(ep, est, rule_local, hold, tgt) -> local` may replace the rule."""
    def __init__(self, n, anatomy, seed, horizon=300., d_min=2., sensing='noise', sense_cfg=DeployableConfig(),
                 avoid=True, camera=None, obstacle_seed=None, perception='detector', topo_pursuit=False):
        base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy, episode_duration_s=horizon,
                       junction_model='union', particle_count=0)
        self.env, self.manifest = paired_environment(base, n, seed)
        env = self.env; self.n = n; self.seed = seed
        if sensing == 'image':
            from marl.image_sensing import CameraConfig, ImageSensor
            self.sensor = ImageSensor(env, camera or CameraConfig(), seed=seed)
        else:
            self.sensor = DeployableSensor(env, sense_cfg, seed=seed)
        lat = self.sensor.cfg.latency_steps
        self.field = ObstacleField(env, np.random.default_rng(seed+11 if obstacle_seed is None else obstacle_seed))
        if sensing == 'image':
            self.sensor.obstacle_field = self.field      # obstacles appear in the biplane images
        # perception: 'detector' (default; noisy boxes, 3 % misses, sensor latency) or 'truth' (partial ground truth
        # as in the simulators of the NMI references: exact local obstacle positions and sizes within the field
        # of view, no misses, no latency; still no identities and no velocities)
        if perception == 'truth':
            from marl.obstacle_field import DetectorConfig
            det, lat = DetectorConfig(pos_sigma_mm=0., size_sigma_frac=0., miss_prob=0., pos_size_frac=0.), 0
        else:
            det = None
        self.perception = perception
        self.tracker = ObstacleTracker(self.field, n, np.random.default_rng(seed+13), lat, env.config.control_dt_s, det)
        self.mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n,
                                     min_spacing_mm=d_min if n > 1 else 0.)
        self.shield = bm.Shield(self.mc, robot_speed_mm_s=env.config.robot_speed_mm_s, control_dt_s=env.config.control_dt_s)
        self.plan, self.plan_info = bm.preoperative_plan(env)
        self.fallback = 'park' if n > 1 else 'help'; bm.FALLBACK['mode'] = self.fallback
        self.targets = bm.PlanTargets(self.plan)
        self.ctl = APFPursuit(env, self.sensor, avoid=avoid, topo=topo_pursuit)
        self.coord = None
        if n > 1:
            from marl.tpg_coordinator import TPGCoordinator
            _, sp = bm._station_paths(env); self.coord = TPGCoordinator(env, self.plan, sp, d_min_mm=d_min)
        self.walls = WallTracker(n)
        self.spacing = SpacingTracker(env.positions_mm[:n], env.active[:n], self.mc.min_spacing_mm if n > 1 else 0.)
        attach_spacing_monitor(env, self.spacing)
        self.initial = float(env.initial_mass.sum()); self.ms = {50: None, 90: None, 100: None}
        self.auc = 0.; self.pair = 0.; self.prev_local = np.zeros((n, 3)); self.info = None; self.t0 = time.monotonic()
        self.wall_total = 0.

    def observe(self):
        bm.FALLBACK['mode'] = self.fallback
        est = self.tracker.observe(self.sensor.observe())
        tgt = self.targets.targets(self.env, est.pos, est.clot_alive)
        rule = self.ctl.act(tgt, est)
        hold = self.coord.gate(est.pos, est.active) if self.coord is not None else np.zeros(self.n, bool)
        return est, tgt, rule, hold

    def step(self, est, local, hold):
        """Apply TPG hold + spacing shield, advance physics and obstacles. Returns (done, step outcome)."""
        env, n = self.env, self.n
        local = local.copy(); local[hold] = 0.
        F = self.ctl.frames(est)
        if n > 1:
            local = self.shield.filtered(local, packet_from(est, F, self.prev_local))
        self.prev_local = local.copy()
        before = 1-float(env.masses.sum())/self.initial; active = env.active[:n].copy()
        mass0 = float(env.masses.sum())
        self.spacing.begin_step()
        _, _, term, trunc, info = env.step(self.ctl.to_world(local, est))
        self.spacing.end_step()
        dt = float(info['step_duration_s'])
        self.field.advance(dt)
        ev, cs, near = self.field.account(env.positions_mm[:n].astype(float), env.active[:n] & active, dt)
        self.walls.update(info, active, dt); self.pair += float(info['robot_pair_contact_s'])
        self.wall_total += float(np.sum(info['wall_contact_s']))
        removal = 1-float(info['remaining_mass'])/self.initial
        self.auc += .5*(before+removal)*dt
        for k in self.ms:
            if self.ms[k] is None and removal >= k/100-1e-12:
                self.ms[k] = float(info['elapsed_s'])
        self.info = info
        out = dict(removed=(mass0-float(env.masses.sum()))/self.initial, wall=np.asarray(info['wall_contact_s'], float),
                   obs_events=ev, obs_contact=cs, obs_near=near, lost=active & ~env.active[:n], active_before=active)
        return bool(term or trunc), out

    def safe(self):
        info = self.info
        return bool(info['success'] and self.wall_total < 1. and int(self.field.events.sum()) == 0 and
                    int(info['lost_robots']) == 0 and self.pair <= 1e-12 and self.spacing.summary()['spacing_compliant'])

    def row(self, method):
        info, env = self.info, self.env
        m = episode_metrics(info, self.walls, self.initial); sp = self.spacing.summary()
        r = dict(method=method, benchmark='v3_obstacles', clusters=self.n, anatomy=env.config.anatomy, seed=self.seed,
                 scenario_hash=self.manifest['scenario_hash'], cluster_safe_success=self.safe(), **m,
                 obstacle_count=int(len(self.field.static_r)+len(self.field.dyn)), obstacle_static=int(len(self.field.static_r)),
                 obstacle_events=int(self.field.events.sum()), obstacle_events_static=self.field.events_static,
                 obstacle_events_dynamic=self.field.events_dynamic, obstacle_contact_s=float(self.field.contact_s.sum()),
                 obstacle_min_clearance_mm=float(np.min(self.field.min_clear)) if np.isfinite(self.field.min_clear).any() else None,
                 robot_pair_contact_s=self.pair, lost=int(info['lost_robots']),
                 t50_s=self.ms[50], t90_s=self.ms[90], t100_s=self.ms[100], removal_auc=self.auc/env.config.episode_duration_s,
                 path_mm=float(np.sum(info['robot_path_mm'])), spacing=sp, walltime_s=time.monotonic()-self.t0)
        return r

    def close(self):
        self.env.close()


def detected_gap(ep, est):
    """Smallest measured surface gap per cluster from the detector boxes (deployable)."""
    g = np.full(ep.n, np.inf)
    for i in range(ep.n):
        for rel, _, r in est.obstacles[i]:
            g[i] = min(g[i], float(np.linalg.norm(rel))-ep.ctl.body-r)
    return g


def run(method, n, anatomy, seed, horizon=300., d_min=2., ckpt=None, sensing='noise', camera=None, shield=None, perception='detector'):
    """shield (mm): deployable proximity shield. When a detected obstacle surface is closer than `shield`,
    the cluster executes the wall-aware APF option instead of the learned choice. rule_switch applies the same
    switch to plain pursuit, i.e. the shield without any learned policy."""
    ep = Episode(n, anatomy, seed, horizon, d_min, sensing, avoid=(method != 'rule_noavoid'), camera=camera, perception=perception)
    drl = student = teacher = None
    if method == 'drl':
        from marl.obstacle_control import DRLController
        drl = DRLController(ep.env, ep.sensor, ep.ctl, ckpt)
    elif method == 'teacher':
        from marl.lookahead_teacher import label, option_local
        teacher = (label, option_local)
    elif method == 'rule_switch':
        from marl.lookahead_teacher import option_local as switch_option
    elif method == 'teacher2':
        from marl.lookahead_teacher import label_v2, option_local as opt2
    elif method == 'student':
        from marl.lookahead_teacher import option_local
        from marl.obstacle_control import History, load_discrete_checkpoint, token, token_dim
        cfg = json.loads(Path(str(ckpt) + '.json').read_text())
        obs_vel = bool(cfg.get('obs_vel', False)); dim = token_dim(obs_vel)
        if int(cfg.get('dim', -1)) != dim:
            raise ValueError('student token dimension mismatch')
        student = (load_discrete_checkpoint(ckpt, cfg), History(ep.n, dim=dim), option_local, obs_vel)
        period = int(cfg.get('period', 1))
    period = 5 if method == 'teacher2' else (period if method == 'student' else 1)
    cur = np.zeros(ep.n, int); k_step = 0
    while True:
        est, tgt, rule, hold = ep.observe()
        if drl is not None:
            local = drl.act(est, rule, hold, tgt)
        elif teacher is not None:
            label_fn, make_option = teacher
            option, _ = label_fn(ep, est, rule, hold)
            local = make_option(ep, est, rule, int(option))
        elif method == 'teacher2':
            if k_step % period == 0:
                cur[:] = label_v2(ep, est, rule, hold, hold_steps=period)[0]
            local = opt2(ep, est, rule, int(cur[0]))
        elif student is not None:
            import torch
            net, hist, make_option, obs_vel = student
            T, _ = token(ep.env, ep.sensor, est, ep.ctl, rule, hold, tgt, ep.prev_local, obs_vel)
            seq, mask = hist.push(T)
            if k_step % period == 0:
                with torch.no_grad():
                    cur[:] = net.logits(torch.as_tensor(seq), torch.as_tensor(mask)).argmax(-1).numpy()
            options = cur.copy()
            if shield is not None:
                options = np.where(detected_gap(ep, est) < shield, 2, options)
            local = np.zeros((ep.n, 3), float)
            for k in np.unique(options):
                trial_local = make_option(ep, est, rule, int(k))
                local[options == k] = trial_local[options == k]
        elif method == 'rule_switch':
            near = detected_gap(ep, est) < (shield if shield is not None else .15)
            local = np.where(near[:, None], switch_option(ep, est, rule, 2), switch_option(ep, est, rule, 0))
        else:
            local = rule
        done, _ = ep.step(est, local, hold); k_step += 1
        if done:
            break
    r = ep.row(method if drl is None and student is None else (Path(ckpt).parent.name if ckpt else method))
    r['sensing_model'] = sensing; r['shield_mm'] = shield; r['perception'] = perception; r['ckpt'] = str(ckpt) if ckpt else None; ep.close()
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=('rule_noavoid', 'rule_apf', 'rule_switch', 'drl', 'student', 'teacher', 'teacher2'))
    p.add_argument('--clusters', type=int, required=True); p.add_argument('--anatomy', required=True)
    p.add_argument('--seeds', required=True); p.add_argument('--ckpt'); p.add_argument('--tag')
    p.add_argument('--sensing', default='noise', choices=('noise', 'image')); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); s0, s1 = map(int, a.seeds.split(':'))
    with a.out.open('a') as f:
        for seed in range(s0, s1+1):
            r = run(a.method, a.clusters, a.anatomy, seed, ckpt=a.ckpt, sensing=a.sensing)
            if a.tag:
                r['method'] = a.tag
            f.write(json.dumps(r, default=str)+'\n'); f.flush()


if __name__ == '__main__':
    main()
