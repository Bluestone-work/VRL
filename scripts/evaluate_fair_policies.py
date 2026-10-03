"""Roll out controllers under the fair partial observation on fixed layouts of one anatomy.

Policies
  teacher         privileged route + avoid + wait (known map, geodesic allocation); upper bound, NOT a fair baseline
  reactive_bearing / reactive_path   hand-written, fair observation only (marl.fair_reactive)
Every run also records how often the robot's straight-line target bearing disagrees with the true route
(diagnostic only, computed from the privileged teacher label after the action is chosen).
usage: evaluate_fair_policies.py --policy P --anatomy A --count N --out JSONL [--split diagnostic] [--robots 5]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.fair_reactive import fair_reactive_action
from marl.partial_obs import PartialObsConfig, PartialObserver
from marl.teacher import execute_local, teacher_config, teacher_label
from scripts.collect_teacher_dataset import student_env_config
from scripts.mca_eval_splits import split
from scripts.train_mca_compiled import reset_with_valid_particles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True); p.add_argument('--anatomy', required=True)
    p.add_argument('--split', default='diagnostic', choices=('diagnostic', 'validation'))
    p.add_argument('--count', type=int, default=20); p.add_argument('--robots', type=int, default=5)
    p.add_argument('--noise', type=float, default=.025); p.add_argument('--position-noise-mm', type=float, default=0.)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    base, size = split(args.split, args.anatomy)
    assert args.count <= size
    cfg = student_env_config(args.anatomy, args.robots)
    tcfg = teacher_config(cfg)
    ocfg = PartialObsConfig(noise=args.noise, position_noise_mm=args.position_noise_mm)
    with args.out.open('a') as out:
        for seed in range(base, base+args.count):
            env = CompiledMCAPhysicalEnv(cfg)
            reset_with_valid_particles(env, seed)
            observer = PartialObserver(env, ocfg, seed=seed)
            wall = 0.; agree = []
            while True:
                obs = observer.observe()
                t_local, t_stop, _ = teacher_label(env, tcfg)
                if args.policy == 'teacher':
                    local = np.where(t_stop[:, None], 0., t_local)
                else:
                    local = fair_reactive_action(obs, args.policy.split('_', 1)[1], ocfg)
                a = env.active[:env.num_robots]
                mv = a & ~t_stop & (np.linalg.norm(t_local, axis=1) > 1e-9) & (np.linalg.norm(local, axis=1) > 1e-9)
                if mv.any():
                    agree += list((local[mv]*t_local[mv]).sum(1)/np.linalg.norm(local[mv], axis=1)/np.linalg.norm(t_local[mv], axis=1))
                observer.record_action(local)
                _, _, term, trunc, info = env.step(execute_local(env, local))
                wall += float(np.sum(info['wall_contact_s']))
                if term or trunc:
                    break
            agree = np.array(agree)
            row = dict(policy=args.policy, anatomy=args.anatomy, split=args.split, seed=seed, robots=args.robots,
                       noise=args.noise, position_noise_mm=args.position_noise_mm,
                       success=bool(info['success']), collision_free=bool(info['collision_free_success']),
                       removal=float(1-info['remaining_mass']/env.initial_mass.sum()), elapsed_s=float(env.elapsed_s),
                       wall_contact_s=wall, particle_events=int(info['episode_particle_collision_events']),
                       lost=int(info['lost_robots']),
                       route_cos=float(agree.mean()) if len(agree) else None,
                       wrong_half_space=float((agree < 0).mean()) if len(agree) else None)
            out.write(json.dumps(row)+'\n'); out.flush()
            env.close()


if __name__ == '__main__':
    main()
