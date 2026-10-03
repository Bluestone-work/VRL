"""Collect a teacher-labelled scene-graph dataset (training anatomies, training seed range only).

Each step stores the student's scene graph (marl.scene_graph.extract_scene) and the
teacher's labels: local Frenet command, stop flag and subgoal clot (its allocation).
The executed action is the teacher's own (behaviour cloning) or, with --student, the
student's action while the teacher still labels every visited state (DAgger).

Schema per episode file (npz, one per episode):
  meta: anatomy, layout seed, robot count, executor, beta, outcome
  static (once): geodesic, depth, kept, vessel_adjacency, station_points, station_frames
  per step (stacked): every dynamic extract_scene array + teacher_local, teacher_stop, teacher_subgoal
usage: collect_teacher_dataset.py --anatomy A --first-seed S --episodes N --out DIR [--robots 4,5,6]
       [--student CKPT --beta 0.0]
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.scene_graph import STATIC_KEYS, extract_scene
from marl.teacher import TEACHER_CONFIG, execute_local, teacher_config, teacher_label
from scripts.train_mca_compiled import reset_with_valid_particles

DYNAMIC_SKIP = set(STATIC_KEYS) | {'anatomy', 'tree_key', 'robot_radius', 'particle_radius', 'robot_speed'}


def student_env_config(anatomy, robots):
    """Student execution config: teacher physics, no action prior (the command is the policy's own)."""
    base = DynamicsConfig.from_json(TEACHER_CONFIG)
    return replace(base, anatomy=anatomy, num_robots=robots, action_prior='none', action_residual_scale=1.,
                   action_wait_clearance=0., action_wait_horizon_s=0.)


def run_episode(env, seed, tcfg, executor=None, beta=1., rng=None, record=True):
    scene_rows, labels = [], []
    reset_with_valid_particles(env, seed)
    static = None
    while True:
        scene = extract_scene(env)
        local, stop, subgoal = teacher_label(env, tcfg)
        if record:
            if static is None:
                static = {k: scene[k] for k in STATIC_KEYS}
            scene_rows.append({k: v for k, v in scene.items() if k not in DYNAMIC_SKIP})
            labels.append((local, stop, subgoal))
        use_teacher = executor is None or (rng is not None and rng.uniform() < beta)
        if use_teacher:
            act = np.where(stop[:, None], 0., local)
        else:
            act = executor([scene])[0]
        act = act*env.active[:env.num_robots, None]
        _, _, term, trunc, info = env.step(execute_local(env, act))
        if term or trunc:
            break
    outcome = dict(success=bool(info['success']), collision_free=bool(info['collision_free_success']),
                   removal=float(1-info['remaining_mass']/env.initial_mass.sum()), elapsed_s=float(env.elapsed_s),
                   lost=int(info['lost_robots']), particle_events=int(info['episode_particle_collision_events']),
                   steps=len(scene_rows) if record else int(env.steps))
    return static, scene_rows, labels, outcome


def save_episode(path, static, rows, labels, meta):
    arrays = {f's_{k}': v for k, v in static.items()}
    for k in rows[0]:
        arrays[f'd_{k}'] = np.stack([r[k] for r in rows])
    arrays['teacher_local'] = np.stack([l[0] for l in labels])
    arrays['teacher_stop'] = np.stack([l[1] for l in labels])
    arrays['teacher_subgoal'] = np.stack([l[2] for l in labels])
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anatomy', required=True); p.add_argument('--first-seed', type=int, required=True)
    p.add_argument('--episodes', type=int, required=True); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--robots', default='4,5,6'); p.add_argument('--student'); p.add_argument('--beta', type=float, default=0.)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    from scripts.mca_eval_splits import load_registry
    registry = load_registry()
    reserved = next(r for r in registry['reserved_ranges'] if r['name'].startswith('teacher dataset'))
    if not (reserved['start'] <= args.first_seed and args.first_seed+args.episodes <= reserved['stop']):
        raise ValueError('Dataset seeds must lie in the registered teacher-dataset range')
    executor = None
    if args.student:
        import torch
        from marl.graph_transformer_student import load_student, student_local_action
        torch.set_num_threads(1)
        model = load_student(args.student, args.device)
        executor = lambda scenes: student_local_action(model, scenes, args.device)
    robots = [int(x) for x in args.robots.split(',')]
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.first_seed)
    for seed in range(args.first_seed, args.first_seed+args.episodes):
        path = args.out/f'{args.anatomy}_{seed}.npz'
        if path.exists():
            continue
        n = int(rng.choice(robots))
        cfg = student_env_config(args.anatomy, n)
        env = CompiledMCAPhysicalEnv(cfg)
        static, rows, labels, outcome = run_episode(env, seed, teacher_config(cfg), executor, args.beta, rng)
        meta = dict(anatomy=args.anatomy, seed=seed, robots=n, executor='student' if executor else 'teacher',
                    beta=args.beta if executor else 1., student=args.student, teacher=TEACHER_CONFIG, **outcome)
        save_episode(path, static, rows, labels, meta)
        print(json.dumps(meta), flush=True)
        env.close()


if __name__ == '__main__':
    main()
