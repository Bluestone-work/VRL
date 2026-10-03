"""Roll out one controller on fixed layouts of one anatomy and record safe-navigation metrics.

Information class is part of every row and must be kept apart in reports:
  privileged  teacher                 route + avoid + wait (known map, geodesic allocation)
              pure_rl:CKPT            EXP40 GAT-MAPPO; its observation contains the known-map route bearing
              residual:CKPT           EXP43 residual RL over the route prior (routed observation)
              residual_shield:CKPT    EXP43 + post-policy shield
  fair        reactive_bearing        marl.fair_reactive, fair observation only (marl.partial_obs)
              reactive_path
Metrics (scripts.safe_metrics): raw task success, safe success (total wall contact < 1 robot-s),
collision-free, wall contact (total, ratio, longest continuous run), timeout, removal, plus the
cosine to the true route direction and wrong-half-space rate (computed from the privileged teacher
label AFTER the action is chosen; diagnostic only, never an input).
usage: evaluate_fair_policies.py --policy P --anatomy A --count N --out JSONL [--split diagnostic] [--robots 5]
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.fair_reactive import fair_reactive_action
from marl.partial_obs import PartialObsConfig, PartialObserver
from marl.teacher import execute_local, teacher_config, teacher_label
from scripts.collect_teacher_dataset import student_env_config
from scripts.mca_eval_splits import split
from scripts.safe_metrics import WallTracker, episode_metrics
from scripts.train_mca_compiled import reset_with_valid_particles

LEARNED_CONFIGS = {'pure_rl': 'configs/experiments/EXP_0040_OWN_BEARING_DYNAMICS.json',
                   'residual': 'configs/experiments/EXP_0043_GATED_PRIOR_DYNAMICS.json',
                   'residual_shield': 'configs/experiments/EXP_0045_SHIELD_DYNAMICS.json'}
INFORMATION = {'teacher': 'privileged', 'pure_rl': 'privileged', 'residual': 'privileged', 'residual_shield': 'privileged',
               'reactive_bearing': 'fair', 'reactive_path': 'fair'}


def make_controller(spec, anatomy, robots):
    """(env config, act(env, obs, observer) -> (world command, local action or None))."""
    kind, _, ckpt = spec.partition(':')
    if kind in LEARNED_CONFIGS:
        import torch
        from marl.mca_physical_policy import make_physical_agent, physical_policy_action
        torch.set_num_threads(1)
        cfg = replace(DynamicsConfig.from_json(LEARNED_CONFIGS[kind]), anatomy=anatomy, num_robots=robots)
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
        agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'], hidden_dim=128, device='cpu')
        agent.load(ckpt, load_optimizers=False)
        def act(env, obs, observer):
            _, _, _, execution, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
            world = env._apply_action_prior(np.asarray(execution, np.float64))  # what the env will execute
            return execution, world
        return cfg, act
    cfg = student_env_config(anatomy, robots)
    if kind == 'teacher':
        return cfg, None
    if kind.startswith('reactive_'):
        mode = kind.split('_', 1)[1]
        def act(env, obs, observer):
            local = fair_reactive_action(observer.observe(), mode, observer.cfg)
            observer.record_action(local)
            world = execute_local(env, local)
            return world, world
        return cfg, act
    raise ValueError(spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True); p.add_argument('--anatomy', required=True)
    p.add_argument('--split', default='diagnostic', choices=('diagnostic', 'validation'))
    p.add_argument('--count', type=int, default=20); p.add_argument('--robots', type=int, default=5)
    p.add_argument('--noise', type=float, default=.025); p.add_argument('--position-noise-mm', type=float, default=0.)
    p.add_argument('--tag', help='label for the output rows (defaults to the policy kind)')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    base, size = split(args.split, args.anatomy)
    assert args.count <= size
    kind = args.policy.split(':')[0]
    cfg, act = make_controller(args.policy, args.anatomy, args.robots)
    tcfg = teacher_config(student_env_config(args.anatomy, args.robots))
    ocfg = PartialObsConfig(noise=args.noise, position_noise_mm=args.position_noise_mm)
    with args.out.open('a') as out:
        for seed in range(base, base+args.count):
            env = CompiledMCAPhysicalEnv(cfg)
            obs, _ = reset_with_valid_particles(env, seed)
            observer = PartialObserver(env, ocfg, seed=seed)
            tracker = WallTracker(env.num_robots); agree = []
            while True:
                t_local, t_stop, _ = teacher_label(env, tcfg)
                if act is None:
                    local = np.where(t_stop[:, None], 0., t_local)
                    command = world = execute_local(env, local)
                else:
                    command, world = act(env, obs, observer)
                frame = np.stack((env.tree.tangents[env.robot_stations], env.tree.normals[env.robot_stations],
                                  env.tree.binormals[env.robot_stations]), axis=1)
                executed = np.einsum('nij,nj->ni', frame, world)
                a = env.active[:env.num_robots].copy()
                mv = a & ~t_stop & (np.linalg.norm(t_local, axis=1) > 1e-9) & (np.linalg.norm(executed, axis=1) > 1e-9)
                if mv.any():
                    agree += list((executed[mv]*t_local[mv]).sum(1)/np.linalg.norm(executed[mv], axis=1)/np.linalg.norm(t_local[mv], axis=1))
                obs, _, term, trunc, info = env.step(command)
                tracker.update(info, a, float(info['step_duration_s']))
                if term or trunc:
                    break
            agree = np.array(agree)
            row = dict(policy=args.tag or kind, information=INFORMATION[kind], checkpoint=args.policy.partition(':')[2] or None,
                       anatomy=args.anatomy, split=args.split, seed=seed, robots=args.robots,
                       noise=args.noise, position_noise_mm=args.position_noise_mm,
                       **episode_metrics(info, tracker, env.initial_mass.sum()),
                       particle_events=int(info['episode_particle_collision_events']), lost=int(info['lost_robots']),
                       route_cos=float(agree.mean()) if len(agree) else None,
                       wrong_half_space=float((agree < 0).mean()) if len(agree) else None)
            out.write(json.dumps(row)+'\n'); out.flush()
            env.close()


if __name__ == '__main__':
    main()
