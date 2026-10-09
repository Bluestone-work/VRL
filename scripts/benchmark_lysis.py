"""Benchmark v4 (2026-10-08): multi-cluster thrombolysis efficiency with wall safety.

User direction (2026-10-08): the main job is to clear more thrombi, faster, with several clusters, while avoiding
the vessel wall; collision avoidance with drifting fragments is not part of the task (moving debris avoidance during
lysis is not a reasonable requirement). Scenes, physics (union-of-tubes lumen), allocation A, sensing and spacing
safety layer as benchmark v2/v3; no obstacle field, no debris tracers. Every controller reads deployable inputs only:
the pre-operative map (centrelines, healthy radii, clot locations), biplane-image estimates of the clusters
(marl.image_sensing) and whether a clot is still visible (est.clot_alive). Simulator truth is used only for metrics.

Four metric families (all per episode, truth-based, never a controller input)
  efficiency    T50 / T90 / T100 (s; None if not reached), removal AUC over the full horizon (removal after
                completion counts as 1 until the horizon), removal rate (removal / elapsed)
  completeness  final removal fraction, all-clots-cleared (task_success), cleared-clot fraction
  interference  spacing-violation pair-seconds (d < d_min), minimum pair distance, pair contact, dipole coupling
                exposure  sum_pairs integral (d_min / d)^3 dt  over d < 2 d_min (field of a magnetic dipole ~ 1/d^3;
                an uncalibrated proxy of mutual magnetic disturbance), TPG hold seconds
  wall          total wall-contact robot-seconds, max continuous contact, wall >= 1 s / >= 5 s indicators
Composite (reported, both): Strict = cleared & wall < 1 s & interference-free; Relaxed = cleared & wall < 5 s &
interference-free (relaxed threshold pre-registered 2026-10-07).

Episode is the shared pipeline; methods plug in (allocator, low-level controller, coordinator) via `run()`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

DEV = dict(base=2600000000, stride=100000)


def dev_seeds(anatomy, count, offset=0):
    order = json.load(open('configs/evaluation_splits.json'))['anatomy_order']
    k = order.index(anatomy)
    return [DEV['base']+k*DEV['stride']+offset+i for i in range(count)]


class LysisEpisode:
    """allocation -> (TPG hold) -> low-level command -> spacing shield -> physics, deployable inputs only."""

    def __init__(self, n, anatomy, seed, horizon=300., d_min=2., sensing='image', shield=True, tpg=True,
                 fallback=None, topo=True, sense_cfg=None, actuation=None, rng=None, variation=None, flow_inlet_mm_s=None):
        from environments.mca_physical_env import DynamicsConfig
        from marl.deployable_sensing import DeployableConfig, DeployablePursuit, DeployableSensor
        from marl.multicluster import MultiClusterConfig
        from marl.teacher import TEACHER_CONFIG
        import scripts.benchmark_multicluster as bm
        from scripts.multicluster_protocol import SpacingTracker, attach_spacing_monitor, paired_environment
        from scripts.safe_metrics import WallTracker
        self.bm = bm
        base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy, episode_duration_s=horizon,
                       junction_model='union', particle_count=0)
        self.env, self.manifest = paired_environment(base, n, seed)
        env = self.env; self.n, self.seed, self.horizon, self.d_min = n, seed, horizon, d_min
        # v6 (2026-10-08): per-anatomy flow calibration. The legacy config gives every anatomy the same inlet volume flow
        # (0.01 mL/min x U[1.5,2.5]), so the mean inlet speed spans 0.0003 (pulmonary) to 0.03 mm/s (MCA). With
        # flow_inlet_mm_s = v the driving pressure is rescaled so that the HEALTHY inlet mean speed is v x (scene
        # multiplier / 2) — the scene's own U[1.5,2.5] variation is kept around v; distribution over branches, stenosis
        # acceleration and lysis feedback are unchanged (linear network).
        self.flow_scale = 1.
        if flow_inlet_mm_s:
            fm = env.flow_model; healthy = fm.solve(fm.healthy_radius_mm); root = fm.root
            v0 = float(healthy['station_inflow_mm3_s'][root])/(np.pi*float(fm.healthy_radius_mm[root])**2)
            self.flow_scale = float(flow_inlet_mm_s)*env.episode_flow_multiplier/2./max(v0, 1e-12)
            fm.driving_pressure *= self.flow_scale; fm.inlet_flow_ml_min *= self.flow_scale
            env.solution = fm.solve(env._radii(env.masses))
        if sensing == 'image':
            from marl.image_sensing import CameraConfig, ImageSensor
            self.sensor = ImageSensor(env, CameraConfig(), seed=seed,
                                      latency_steps=(sense_cfg.latency_steps if sense_cfg else 1))
        else:
            self.sensor = DeployableSensor(env, sense_cfg or DeployableConfig(), seed=seed)
        self.mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n,
                                     min_spacing_mm=d_min if n > 1 else 0.)
        self.shield = bm.Shield(self.mc, robot_speed_mm_s=env.config.robot_speed_mm_s,
                                control_dt_s=env.config.control_dt_s) if (shield and n > 1) else None
        self.plan, self.plan_info = bm.preoperative_plan(env)
        self.fallback = fallback or ('park' if (n > 1 and tpg) else 'help')
        self.plan_targets = bm.PlanTargets(self.plan)
        self.ctl = DeployablePursuit(env, self.sensor, topo=topo)
        self.coord = None
        if tpg and n > 1:
            from marl.tpg_coordinator import TPGCoordinator
            _, sp = bm._station_paths(env)
            self.coord = TPGCoordinator(env, self.plan, sp, d_min_mm=d_min)
        # actuation randomisation (training only): per-cluster gain and execution noise
        if actuation is not None and rng is None:          # evaluation stress test: paired, seeded per scene
            rng = np.random.default_rng(seed+977)
        self.actuation = actuation; self.rng = rng
        self.gain = np.ones(n) if actuation is None else rng.uniform(*actuation['gain'], n)
        self.walls = WallTracker(n)
        self.spacing = SpacingTracker(env.positions_mm[:n], env.active[:n], d_min if n > 1 else 0.)
        attach_spacing_monitor(env, self.spacing)
        self.initial = float(env.initial_mass.sum()); self.initial_each = env.initial_mass.astype(float).copy()
        self.ms = {50: None, 90: None, 100: None}
        self.auc = 0.; self.pair = 0.; self.prev_local = np.zeros((n, 3)); self.info = None
        self.sent_world = np.zeros((n, 3))  # final command before unknown plant response
        self.coupling = 0.; self.hold_s = 0.; self.t0 = time.monotonic(); self.steps = 0
        # benchmark v5: patient-specific / time-varying dynamics (marl.physio_variation); installed after every
        # controller component was built from the nominal (pre-operative) model
        self.var = None
        if variation:
            from marl.physio_variation import Variation
            vs = variation if isinstance(variation, dict) else dict(s=float(variation))
            self.var = Variation(self, vs['s'], np.random.default_rng([seed, 5150, int(round(1000*vs['s']))]),
                                 pulsatile=vs.get('pulsatile', True))

    # ---- deployable step interface
    def observe(self):
        self.bm.FALLBACK['mode'] = self.fallback
        return self.sensor.observe()

    def plan_targets_now(self, est):
        self.bm.FALLBACK['mode'] = self.fallback
        return self.plan_targets.targets(self.env, est.pos, est.clot_alive)

    def hold(self, est):
        return self.coord.gate(est.pos, est.active) if self.coord is not None else np.zeros(self.n, bool)

    def step(self, est, local, hold):
        """local: [n, 3] commands in the controller's local frames. Returns (done, outcome)."""
        from scripts.benchmark_deployable import packet_from
        env, n = self.env, self.n
        local = np.asarray(local, float).copy(); local[hold] = 0.
        if self.shield is not None:
            local = self.shield.filtered(local, packet_from(est, self.ctl.frames(est), self.prev_local))
        self.prev_local = local.copy()
        world = self.ctl.to_world(local, est)
        self.sent_world = world.copy()
        if self.actuation is not None:
            world = world*self.gain[:, None]+self.rng.normal(0., self.actuation['noise'], world.shape)*(np.linalg.norm(world, axis=1, keepdims=True) > 0)
        if self.var is not None:
            world = self.var.actuate(world)
        before = 1-float(env.masses.sum())/self.initial; active = env.active[:n].copy()
        mass0 = env.masses.astype(float).copy()
        self.spacing.begin_step()
        _, _, term, trunc, info = env.step(world)
        self.spacing.end_step()
        dt = float(info['step_duration_s'])
        self.walls.update(info, active, dt); self.pair += float(info['robot_pair_contact_s'])
        self.hold_s += float(np.sum(hold & active))*dt
        if n > 1:
            P = env.positions_mm[:n].astype(float)
            for a in range(n):
                for b in range(a+1, n):
                    if active[a] and active[b]:
                        d = float(np.linalg.norm(P[a]-P[b]))
                        if d < 2*self.d_min:
                            self.coupling += min((self.d_min/max(d, 1e-6))**3, 1e3)*dt
        removal = 1-float(info['remaining_mass'])/self.initial
        self.auc += .5*(before+removal)*dt
        for k in self.ms:
            if self.ms[k] is None and removal >= k/100-1e-12:
                self.ms[k] = float(info['elapsed_s'])
        self.info = info; self.steps += 1
        if self.var is not None:
            self.var.observe_outcome(info)
        out = dict(removed=(float(mass0.sum())-float(env.masses.sum()))/self.initial,
                   removed_each=(mass0-env.masses.astype(float))/self.initial,
                   wall=np.asarray(info['wall_contact_s'], float), lost=active & ~env.active[:n],
                   active_before=active, agent_removed=np.asarray(info.get('agent_removed_mass', np.zeros(n)), float),
                   terminated=bool(term), truncated=bool(trunc))
        return bool(term or trunc), out

    def row(self, method, **extra):
        from scripts.safe_metrics import episode_metrics
        info, env = self.info, self.env
        m = episode_metrics(info, self.walls, self.initial); sp = self.spacing.summary()
        elapsed = float(info['elapsed_s'])
        auc = (self.auc+m['removal']*max(self.horizon-elapsed, 0.))/self.horizon
        free = bool(sp['spacing_compliant'] and self.pair <= 1e-12) if self.n > 1 else True
        cleared = bool(m['task_success'] and int(info['lost_robots']) == 0)
        r = dict(method=method, benchmark='v4_lysis', clusters=self.n, anatomy=env.config.anatomy, seed=self.seed,
                 scenario_hash=self.manifest['scenario_hash'], horizon_s=self.horizon, d_min_mm=self.d_min,
                 task_success=bool(m['task_success']), removal=m['removal'],
                 clots=int(len(self.initial_each)), clots_cleared=int(np.sum(env.masses <= 0)),
                 t50_s=self.ms[50], t90_s=self.ms[90], t100_s=self.ms[100], removal_auc=auc,
                 removal_rate_per_min=60.*m['removal']/max(elapsed, 1e-9), elapsed_s=elapsed,
                 wall_contact_s=m['wall_contact_s'], max_continuous_wall_contact_s=m['max_continuous_wall_contact_s'],
                 wall_ge_1s=bool(m['wall_contact_s'] >= 1.), wall_ge_5s=bool(m['wall_contact_s'] >= 5.),
                 spacing_violation_pair_s=float(sp['spacing_violation_pair_s']), min_pair_distance_mm=sp['minimum_spacing_mm'],
                 robot_pair_contact_s=self.pair, coupling_exposure=self.coupling, interference_free=free,
                 tpg_hold_s=self.hold_s, lost=int(info['lost_robots']),
                 strict_success=bool(cleared and m['wall_contact_s'] < 1. and free),
                 relaxed_success=bool(cleared and m['wall_contact_s'] < 5. and free),
                 path_mm=float(np.sum(info['robot_path_mm'])), termination=m['termination_reason'],
                 walltime_s=time.monotonic()-self.t0, variation=self.var.summary() if self.var else None,
                 flow_scale=self.flow_scale, **extra)
        return r

    def close(self):
        self.env.close()


# ----------------------------------------------------------------------------------------- classical low levels
class WallGuard:
    """Reference wall guard (EXP0066 setting margin 0.20 mm, gain 0.4) + stall re-planning, map + estimates only."""
    def __init__(self, ep, margin=.2, gain=.4):
        from marl.hierarchical_navigation import EventReplanner, NavigationConfig, WallRecoveryExecutor
        self.cfg = NavigationConfig(recovery_margin_mm=margin, recovery_gain=gain)
        self.exec = WallRecoveryExecutor(ep.sensor, ep.n, ep.ctl.body, self.cfg)
        self.replan = EventReplanner(ep.n, self.cfg)

    def __call__(self, ep, est, targets, rule, hold):
        t = ep.env.elapsed_s
        changed = self.replan.update(t, est, targets, ep.ctl.to_world(ep.prev_local, est), hold, ep.ctl)
        if changed:
            rule = ep.ctl.act(targets, est); self.exec.trigger(t, changed)
        return self.exec.act(t, est, ep.ctl, rule, hold)




METHODS = {}


def method(name):
    def deco(f):
        METHODS[name] = f
        return f
    return deco


def rollout(ep, allocate, low, tag, **extra):
    """Generic episode loop. allocate(ep, est) -> targets; low(ep, est, targets, rule, hold) -> local."""
    while True:
        est = ep.observe()
        tgt = allocate(ep, est)
        rule = ep.ctl.act(tgt, est)
        hold = ep.hold(est)
        local = low(ep, est, tgt, rule, hold) if low is not None else rule
        done, _ = ep.step(est, local, hold)
        if done:
            break
    return ep.row(tag, **extra)


@method('ours_classical')
def m_ours_classical(n, anatomy, seed, **kw):
    """A + TPG + topology-fixed pursuit + wall guard + stall re-planning (the current deployable prior)."""
    ep = LysisEpisode(n, anatomy, seed, **kw); g = WallGuard(ep)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), g, 'ours_classical'), ep


@method('pursuit_tpg')
def m_pursuit_tpg(n, anatomy, seed, **kw):
    """A + TPG + topology-fixed pursuit (benchmark v2 'ours' + topo fix), no wall guard."""
    ep = LysisEpisode(n, anatomy, seed, **kw)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), None, 'pursuit_tpg'), ep


def rollout_alloc(ep, alloc, low, tag, **extra):
    """Episode loop with an online allocator (marl.lysis_alloc) that also owns the TPG hold."""
    while True:
        est = ep.observe()
        tgt = alloc(ep, est)
        rule = ep.ctl.act(tgt, est)
        hold = alloc.hold(ep, est)
        local = low(ep, est, tgt, rule, hold) if low is not None else rule
        alloc.observe_motion(est, np.where(hold[:, None], 0., local))
        done, _ = ep.step(est, local, hold)
        if done:
            break
    tpg = alloc.st.tpg
    return ep.row(tag, decisions=alloc.decision_count, tpg_builds=tpg.builds if tpg else 0, **extra)


@method('alloc_prior')
def m_alloc_prior(n, anatomy, seed, **kw):
    """A executed through the online allocator + dynamic TPG (sanity check of the learned allocator's start)."""
    from marl.lysis_alloc import LearnedAllocator
    ep = LysisEpisode(n, anatomy, seed, tpg=False, **kw)
    return rollout_alloc(ep, LearnedAllocator(ep, prior_only=True), WallGuard(ep), 'alloc_prior'), ep


@method('alloc_auction')
def m_alloc_auction(n, anatomy, seed, **kw):
    """Classical online re-allocation + dynamic TPG + pursuit + wall guard (heuristic ablation)."""
    from marl.lysis_alloc import AuctionAllocator
    ep = LysisEpisode(n, anatomy, seed, tpg=False, **kw)
    return rollout_alloc(ep, AuctionAllocator(ep), WallGuard(ep), 'alloc_auction'), ep


ALLOC_CKPT = {'path': None}


@method('alloc_learned')
def m_alloc_learned(n, anatomy, seed, ckpt=None, use_tpg=True, low='guard', **kw):
    """Learned MAPPO allocator (greedy) + dynamic TPG + pursuit + wall guard."""
    import torch
    from marl.lysis_alloc import AllocPolicy, LearnedAllocator
    pol = AllocPolicy(); pol.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state']); pol.eval()
    ep = LysisEpisode(n, anatomy, seed, tpg=False, **kw)
    lw = WallGuard(ep) if low == 'guard' else None
    return rollout_alloc(ep, LearnedAllocator(ep, pol, use_tpg=use_tpg), lw, 'alloc_learned', ckpt=str(ckpt)), ep


@method('pac_nmpc')
def m_pac_nmpc(n, anatomy, seed, **kw):
    """Baseline: PAC-NMPC-style distributed sampling NMPC (CASE 2025) replaces TPG + local control; allocation A."""
    from marl.lysis_baselines import PacNMPC
    ep = LysisEpisode(n, anatomy, seed, tpg=False, fallback='park' if n > 1 else 'help', **kw); ctl = PacNMPC(ep)
    row = rollout(ep, lambda e, est: e.plan_targets_now(est), ctl, 'pac_nmpc')
    row.update(pac_bound_fail_rate=ctl.bound_fail/max(ctl.calls, 1))
    return row, ep


@method('stpg')
def m_stpg(n, anatomy, seed, **kw):
    """Baseline: switchable TPG re-optimised online (AAAI 2025 STPG/IGSES objective); A + pursuit + settle guard."""
    from marl.lysis_baselines import SwitchableTPG
    ep = LysisEpisode(n, anatomy, seed, **kw); g = SettleGuard(ep)      # same low level as ours
    sw = SwitchableTPG(ep.coord) if ep.coord is not None else None
    def low(e, est, tgt, rule, hold):
        return g(e, est, tgt, rule, hold)
    def alloc(e, est):
        if sw is not None:
            sw.update(e.env.elapsed_s)
        return e.plan_targets_now(est)
    row = rollout(ep, alloc, low, 'stpg')
    if sw is not None:
        row.update(stpg_switches=sw.switches, stpg_solves=sw.solves, stpg_infeasible=sw.infeasible)
    return row, ep


@method('local_ppo')
def m_local_ppo(n, anatomy, seed, ckpt=None, post_guard=False, **kw):
    """Learned local controller (scripts/train_lysis_local.py): Turbo baseline (direct) or our residual."""
    from scripts.train_lysis_local import LocalController
    ep = LysisEpisode(n, anatomy, seed, **kw)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), LocalController(ep, ckpt, post_guard), 'local_ppo', ckpt=str(ckpt)), ep


@method('ours')
def m_ours(n, anatomy, seed, ckpt=None, low_ckpt=None, use_tpg=True, **kw):
    """Ours: learned MAPPO allocator + dynamic TPG + pursuit + wall guard (+ learned residual low level)."""
    import torch
    from marl.lysis_alloc import AllocPolicy, LearnedAllocator
    pol = AllocPolicy(); pol.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state']); pol.eval()
    ep = LysisEpisode(n, anatomy, seed, tpg=False, **kw)
    if low_ckpt:
        from scripts.train_lysis_local import LocalController
        low = LocalController(ep, low_ckpt)
    else:
        low = WallGuard(ep)
    return rollout_alloc(ep, LearnedAllocator(ep, pol, use_tpg=use_tpg), low, 'ours', ckpt=str(ckpt), low_ckpt=str(low_ckpt)), ep


class SettleGuard(WallGuard):
    """WallGuard + dwell settling: within `radius` mm of the target clot's pre-operative position the command
    magnitude is scaled by distance / radius (bounded actuator), so the cluster settles on the clot instead of
    circling it at full speed (diagnosis 2026-10-08: pursuit spent 295 steps within 1 mm, 81 in lysis contact)."""
    def __init__(self, ep, radius=.3, **kw):
        super().__init__(ep, **kw); self.radius = radius

    def __call__(self, ep, est, tgt, rule, hold):
        out = super().__call__(ep, est, tgt, rule, hold)
        for i, t in enumerate(tgt):
            if t >= 0:
                d = float(np.linalg.norm(ep.env.clot_positions_mm[t]-est.pos[i]))
                if d < self.radius:
                    out[i] *= d/self.radius
        return out


class AdaptiveSettleGuard(WallGuard):
    """Strongest adaptive classical low level (v5 baseline): settle guard whose slow-down is released when the
    measured approach stalls (estimate not closer to the clot by 0.03 mm over the last 1 s while settling: flow or a
    weak actuator is pushing it back), re-armed after 2 s; the settle radius scales with the measured response
    (ratio of measured speed along the command to the commanded speed, EMA)."""
    def __init__(self, ep, radius=.3, **kw):
        super().__init__(ep, **kw); self.radius = radius; n = ep.n
        self.hist = [[] for _ in range(n)]; self.release_until = np.zeros(n); self.resp = np.ones(n)
        self.prev_cmd = np.zeros((n, 3))

    def __call__(self, ep, est, tgt, rule, hold):
        out = super().__call__(ep, est, tgt, rule, hold)
        t = ep.env.elapsed_s; F = ep.ctl.frames(est)
        for i, c in enumerate(tgt):
            w = F[i].T@self.prev_cmd[i]; sp = float(np.linalg.norm(w))
            if sp > .5:
                self.resp[i] = .95*self.resp[i]+.05*float(np.clip(est.vel[i]@w/sp**2, 0., 2.))
            if c < 0:
                self.hist[i].clear(); continue
            d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
            r = self.radius*float(np.clip(self.resp[i], .5, 1.5))
            self.hist[i].append(d); self.hist[i] = self.hist[i][-10:]
            if d < r and t >= self.release_until[i]:
                if len(self.hist[i]) == 10 and self.hist[i][0]-d < .03 and d > .12:
                    self.release_until[i] = t+2.        # stalled while settling: push at full command
                else:
                    out[i] *= d/r
        self.prev_cmd = out.copy()
        return out


@method('classical_adaptive')
def m_classical_adaptive(n, anatomy, seed, **kw):
    """Adaptive classical (v5 baseline): A + TPG + pursuit + wall guard + adaptive settling."""
    ep = LysisEpisode(n, anatomy, seed, **kw)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), AdaptiveSettleGuard(ep), 'classical_adaptive'), ep


class FlowOracle:
    """PRIVILEGED upper bound for v5 (feasibility analysis only, never a deployable method). Uses simulator truth:
    exact position/edge, the instantaneous flow velocity at the cluster, and the cluster's true response (gain,
    drift, direction bias, shape and near-wall factors, adhesion). It inverts the response so the cluster moves with a
    desired world velocity v_des:  command = R^T (v_des - u_flow) / k,  clipped to the actuator limit.
    v_des: toward the pursuit carrot at 1 mm/s; within 0.5 mm of the target clot, toward the clot at 2 d mm/s (settle);
    when the axial flow at the cluster opposes the route by more than 0.7 k, it also aims radially toward the wall
    region where the Poiseuille speed is lower (|r|/R <= 0.85), i.e. the best case for flow-limited approach."""
    def __init__(self, ep, know_flow=True, know_response=True):
        self.ep = ep; self.know_flow, self.know_response = know_flow, know_response

    def __call__(self, ep, est, tgt, rule, hold):
        env, var, n = ep.env, ep.var, ep.n
        tp = env.transport; P = env.positions_mm[:n].astype(float); E = env.edges[:n]
        q = var.q(env.elapsed_s) if var is not None else 1.
        sol = dict(env.solution); sol['station_inflow_mm3_s'] = sol['station_inflow_mm3_s']*q
        uf = tp.velocity_mm_s(P, E, sol)*(1. if self.know_flow else 0.)
        axis, lumen, radial, _ = tp.coordinates(P, E, env.solution)
        F = ep.ctl.frames(est); out = np.zeros((n, 3))
        for i in range(n):
            t = int(tgt[i])
            if t < 0 or not est.active[i] or hold[i]:
                continue
            k = 1.
            R = np.eye(3)
            if var is not None and self.know_response:
                k = var.gain0[i]*np.exp(var.log_drift[i])
                gap = max(float(lumen[i]-env.body_radius[i]), 0.)
                k *= float(np.clip(gap/.3, .4, 1.))**var.beta
                k *= 1-var.adhesion if var.touch[i] else 1.
                R = var.rot[i]
            sgap = max(float(lumen[i]-radial[i]-env.body_radius[i]), 0.)
            lub = tp.lubrication_floor+(1-tp.lubrication_floor)*float(np.clip(sgap/(4*env.body_radius[i]), 0, 1))
            k *= lub if self.know_response else 1.
            clot = env.clot_positions_mm[t]; dc = float(np.linalg.norm(clot-P[i]))
            if dc < .5:
                v_des = (clot-P[i])*2.
            else:
                d = ep.ctl.carrot[i]-P[i]; v_des = d/max(np.linalg.norm(d), 1e-9)
                ax_dir = tp.direction[E[i]]; opp = -float(uf[i]@ax_dir)*np.sign(float(v_des@ax_dir) or 1.)
                if opp > .7*k:
                    off = P[i]-axis[i]; ro = float(np.linalg.norm(off))
                    side = off/ro if ro > 1e-6 else np.cross(ax_dir, [1., 0, 0])/max(np.linalg.norm(np.cross(ax_dir, [1., 0, 0])), 1e-9)
                    target_r = .85*max(float(lumen[i]-env.body_radius[i]), 0.)
                    v_des = v_des+1.5*(target_r-ro)*side
            c = R.T@(v_des-uf[i])/max(k, 1e-3)
            c = c/max(np.linalg.norm(c), 1.)
            out[i] = F[i]@c
        return out                     # no wall guard: the bound may use the near-wall region freely


@method('oracle_flow')
def m_oracle_flow(n, anatomy, seed, know_flow=True, know_response=True, **kw):
    """PRIVILEGED feasibility bound (FlowOracle) with exact sensing; A + TPG."""
    from marl.deployable_sensing import DeployableConfig
    kw = dict(kw); kw.update(sensing='noise', sense_cfg=DeployableConfig(position_sigma_mm=0., latency_steps=0, dropout_prob=0.))
    ep = LysisEpisode(n, anatomy, seed, **kw)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), FlowOracle(ep, know_flow, know_response), 'oracle_flow'), ep


@method('classical_settle')
def m_classical_settle(n, anatomy, seed, radius=.3, **kw):
    """ours_classical + dwell settling (hand-crafted)."""
    ep = LysisEpisode(n, anatomy, seed, **kw)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), SettleGuard(ep, radius), 'classical_settle'), ep


@method('alloc_settle')
def m_alloc_settle(n, anatomy, seed, ckpt=None, auction=False, **kw):
    """Learned (or auction) allocator + dynamic TPG + settle guard (classical low level)."""
    import torch
    from marl.lysis_alloc import AllocPolicy, AuctionAllocator, LearnedAllocator
    ep = LysisEpisode(n, anatomy, seed, tpg=False, **kw)
    if auction:
        al = AuctionAllocator(ep)
    else:
        pol = AllocPolicy(); pol.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state']); pol.eval()
        al = LearnedAllocator(ep, pol)
    return rollout_alloc(ep, al, SettleGuard(ep), 'alloc_settle'), ep

@method('nav')
def m_nav(n, anatomy, seed, ckpt=None, **kw):
    """Ours (v5): route-frame Transformer PPO navigation (scripts/train_lysis_nav.py) + shared wall guard; A + TPG."""
    from scripts.train_lysis_nav import NavController
    ep = LysisEpisode(n, anatomy, seed, **kw)
    return rollout(ep, lambda e, est: e.plan_targets_now(est), NavController(ep, ckpt), 'nav', ckpt=str(ckpt)), ep


def run_job(job):
    name, n, anatomy, seed, kw = job
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        row, ep = METHODS[name](n, anatomy, seed, **{k: v for k, v in kw.items() if k != 'tag'})
        ep.close()
        return row
    except Exception as e:
        return dict(method=kw.get('tag', name), clusters=n, anatomy=anatomy, seed=seed, error=repr(e),
                    trace=traceback.format_exc()[-2000:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True); p.add_argument('--clusters', default='1,2,3')
    p.add_argument('--anatomies', default='all'); p.add_argument('--count', type=int, default=10)
    p.add_argument('--offset', type=int, default=0); p.add_argument('--workers', type=int, default=18)
    p.add_argument('--horizon-s', type=float, default=300.); p.add_argument('--sensing', default='image')
    p.add_argument('--kw', default='{}', help='JSON keyword arguments for the method')
    p.add_argument('--tag'); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    order = json.load(open('configs/evaluation_splits.json'))['anatomy_order']
    anats = order if a.anatomies == 'all' else a.anatomies.split(',')
    kw = dict(horizon=a.horizon_s, sensing=a.sensing, **json.loads(a.kw))
    if a.tag:
        kw['tag'] = a.tag
    jobs = [(a.method, int(n), an, s, kw) for n in a.clusters.split(',') for an in anats
            for s in dev_seeds(an, a.count, a.offset)]
    done = set()
    if a.out.exists():
        for l in a.out.open():
            r = json.loads(l)
            if 'error' not in r:
                done.add((r['method'], r['clusters'], r['anatomy'], r['seed']))
    jobs = [j for j in jobs if (a.tag or a.method, j[1], j[2], j[3]) not in done]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(f'{len(jobs)} jobs', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=20) as pool, a.out.open('a') as f:
        for r in pool.imap_unordered(run_job, jobs):
            if a.tag and 'error' not in r:
                r['method'] = a.tag
            f.write(json.dumps(r, default=float)+'\n'); f.flush()
            if 'error' in r:
                print('ERROR', r['anatomy'], r['seed'], r['error'], flush=True)


if __name__ == '__main__':
    main()
