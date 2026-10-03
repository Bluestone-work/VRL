"""Roll out teacher / BC student / pure RL on fixed layouts of one anatomy (no training, no test split).

Policies
  teacher            route + avoid + wait controller (training-time teacher; reference only)
  student:CKPT       Graph Transformer student, scene graph only
  pure_rl:CKPT       EXP40 pure-RL GAT-MAPPO checkpoint (MCA-trained), its own routed observation
Metrics per episode: completion, collision-free completion, removal, episode length, wall contact
robot-seconds, particle collision events, lost robots, and for learned policies the agreement with the
teacher label on the states the policy itself visits (cosine on teacher-moving steps, stop agreement).
usage: evaluate_student_policies.py --policy P --anatomy A --split diagnostic --count N --out JSONL [--robots 5]
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.scene_graph import extract_scene
from marl.teacher import execute_local, teacher_config, teacher_label
from scripts.collect_teacher_dataset import student_env_config
from scripts.mca_eval_splits import split
from scripts.train_mca_compiled import reset_with_valid_particles


def make_policy(spec, anatomy, robots, device):
    kind, _, ckpt = spec.partition(':')
    cfg = student_env_config(anatomy, robots)
    if kind == 'teacher':
        return cfg, None
    if kind == 'student':
        import torch
        from marl.graph_transformer_student import load_student, student_local_action
        torch.set_num_threads(1)
        model = load_student(ckpt, device)
        return cfg, lambda env, obs: student_local_action(model, [extract_scene(env)], device)[0]
    if kind == 'pure_rl':
        import torch
        from marl.mca_physical_policy import make_physical_agent, physical_policy_action
        torch.set_num_threads(1)
        cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0040_OWN_BEARING_DYNAMICS.json'),
                      anatomy=anatomy, num_robots=robots)
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
        agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'], hidden_dim=128, device='cpu')
        agent.load(ckpt, load_optimizers=False)
        def act(env, obs):
            proposal, *_ = physical_policy_action(agent, env, obs, deterministic=True)
            return np.asarray(proposal, np.float64).reshape(env.num_robots, 3)
        return cfg, act
    raise ValueError(spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True); p.add_argument('--anatomy', required=True)
    p.add_argument('--split', default='diagnostic', choices=('diagnostic', 'validation'))
    p.add_argument('--count', type=int, default=20); p.add_argument('--robots', type=int, default=5)
    p.add_argument('--out', required=True, type=Path); p.add_argument('--device', default='cpu')
    args = p.parse_args()
    base, size = split(args.split, args.anatomy)
    assert args.count <= size
    cfg, act = make_policy(args.policy, args.anatomy, args.robots, args.device)
    with args.out.open('a') as out:
        for seed in range(base, base+args.count):
            env = CompiledMCAPhysicalEnv(cfg)
            tcfg = teacher_config(student_env_config(args.anatomy, args.robots)) if cfg.target_observation == 'routed_assigned_own' else None
            obs, _ = reset_with_valid_particles(env, seed)
            wall = 0.; cos = []; stop_agree = []
            while True:
                if tcfg is not None:
                    t_local, t_stop, _ = teacher_label(env, tcfg)
                if act is None:
                    local = np.where(t_stop[:, None], 0., t_local)
                else:
                    local = act(env, obs)
                    if tcfg is not None:
                        a = env.active[:env.num_robots]
                        mv = a & ~t_stop & (np.linalg.norm(t_local, axis=1) > 1e-9)
                        n1 = np.linalg.norm(local, axis=1)
                        ok = mv & (n1 > 1e-9)
                        if ok.any():
                            cos += list((local[ok]*t_local[ok]).sum(1)/n1[ok]/np.linalg.norm(t_local[ok], axis=1))
                        stop_agree += list(((n1 < 1e-9) == t_stop)[a])
                obs, _, term, trunc, info = env.step(execute_local(env, local))
                wall += float(np.sum(info['wall_contact_s']))
                if term or trunc:
                    break
            row = dict(policy=args.policy, anatomy=args.anatomy, split=args.split, seed=seed, robots=args.robots,
                       success=bool(info['success']), collision_free=bool(info['collision_free_success']),
                       removal=float(1-info['remaining_mass']/env.initial_mass.sum()), elapsed_s=float(env.elapsed_s),
                       wall_contact_s=wall, particle_events=int(info['episode_particle_collision_events']),
                       lost=int(info['lost_robots']), reason=str(info.get('termination_reason')),
                       teacher_cos=float(np.mean(cos)) if cos else None,
                       teacher_stop_agreement=float(np.mean(stop_agree)) if stop_agree else None)
            out.write(json.dumps(row)+'\n'); out.flush()
            env.close()


if __name__ == '__main__':
    main()
