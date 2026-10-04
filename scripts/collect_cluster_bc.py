"""Behaviour-cloning data for the cluster benchmark: route_follow (+ spacing shield) demonstrations.

Training anatomies only (anatomy_holdout_v1.train), registered teacher-dataset seed range. For every
stored step: the scene graph (marl.scene_graph, roadmap-level input for the graph student), the fair
local observation vector (marl.partial_obs via the sensor adapter, for the local student), the plan
target slot, and the teacher's nominal local command / stop flag / planned clot (subgoal).
The executed action is the teacher's own after the shared spacing shield.
usage: collect_cluster_bc.py --anatomy A --clusters N --first-seed S --episodes E --out DIR [--stride 3]
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.edge_follower import RouteFollower
from marl.multicluster import MultiClusterConfig
from marl.multicluster_observation import ClusterSensorAdapter
from marl.scene_graph import STATIC_KEYS, extract_scene
from marl.teacher import TEACHER_CONFIG
from scripts.benchmark_multicluster import PlanTargets, Shield, preoperative_plan, slots_for
from scripts.collect_teacher_dataset import DYNAMIC_SKIP
from scripts.multicluster_protocol import paired_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anatomy', required=True); p.add_argument('--clusters', type=int, required=True)
    p.add_argument('--first-seed', type=int, required=True); p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--stride', type=int, default=3)
    p.add_argument('--horizon-s', type=float, default=300.)
    p.add_argument('--student-local', help='DAgger: this local student executes with prob 1-beta; teacher labels every state')
    p.add_argument('--beta', type=float, default=0.)
    a = p.parse_args()
    reg = json.loads(Path('configs/evaluation_splits.json').read_text())
    assert a.anatomy in reg['anatomy_holdout_v1']['train'], 'training anatomies only'
    r = next(x for x in reg['reserved_ranges'] if x['name'].startswith('teacher dataset'))
    assert r['start'] <= a.first_seed and a.first_seed+a.episodes <= r['stop']
    a.out.mkdir(parents=True, exist_ok=True)
    student = None
    if a.student_local:
        import torch
        from marl.local_student import load_local, local_student_action
        torch.set_num_threads(1); student = load_local(a.student_local)
    base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=a.anatomy, episode_duration_s=a.horizon_s)
    for seed in range(a.first_seed, a.first_seed+a.episodes):
        path = a.out/f'{a.anatomy}_N{a.clusters}_{seed}.npz'
        if path.exists():
            continue
        try:
            env, _ = paired_environment(base, a.clusters, seed)
        except (ValueError, RuntimeError):
            continue
        mc = MultiClusterConfig(method='multi_parallel' if a.clusters > 1 else 'single_sequential', clusters=a.clusters,
                                min_spacing_mm=2. if a.clusters > 1 else 0.)
        sensor = ClusterSensorAdapter(env, mc); sensor.reset(seed); rng = np.random.default_rng(seed)
        shield = Shield(mc, robot_speed_mm_s=env.config.robot_speed_mm_s, control_dt_s=env.config.control_dt_s)
        plan, _ = preoperative_plan(env); targets = PlanTargets(plan); teacher = RouteFollower(env)
        rows, nav, slot, lab_local, lab_stop, lab_goal, static = [], [], [], [], [], [], None
        while True:
            packet = sensor.observe()
            tgt = targets.targets(env, env.positions_mm[:a.clusters])
            local = teacher.act(tgt)
            if env.steps % a.stride == 0:
                scene = extract_scene(env)
                if static is None:
                    static = {k: scene[k] for k in STATIC_KEYS}
                row = {k: v for k, v in scene.items() if k not in DYNAMIC_SKIP}
                row['robot_goal'] = np.where(env.active[:a.clusters], tgt, -1).astype(np.int64)   # allocation A (input)
                rows.append(row)
                nav.append(packet.navigation.copy()); slot.append(slots_for(packet, tgt))
                lab_local.append(local.astype(np.float32))
                lab_stop.append((np.linalg.norm(local, axis=1) < 1e-12) & env.active[:a.clusters])
                lab_goal.append(np.where(env.active[:a.clusters], tgt, -1))
            if student is not None and rng.uniform() >= a.beta:
                local = local_student_action(student, packet.navigation, slots_for(packet, tgt))
            _, _, term, trunc, info = env.step(sensor.execute(shield.filtered(local, packet)))
            if term or trunc:
                break
        arrays = {f's_{k}': v for k, v in static.items()}
        for k in rows[0]:
            arrays[f'd_{k}'] = np.stack([x[k] for x in rows])
        meta = dict(anatomy=a.anatomy, clusters=a.clusters, seed=seed, teacher='route_follow+shield',
                    executor='student' if student is not None else 'teacher', beta=a.beta if student is not None else 1., student=a.student_local,
                    success=bool(info['success']), removal=float(1-info['remaining_mass']/env.initial_mass.sum()),
                    elapsed_s=float(env.elapsed_s), steps=len(rows), stride=a.stride)
        np.savez_compressed(path, meta=json.dumps(meta), nav=np.stack(nav), target_slot=np.stack(slot),
                            teacher_local=np.stack(lab_local), teacher_stop=np.stack(lab_stop),
                            teacher_subgoal=np.stack(lab_goal), **arrays)
        print(json.dumps(meta), flush=True)
        env.close()


if __name__ == '__main__':
    main()
