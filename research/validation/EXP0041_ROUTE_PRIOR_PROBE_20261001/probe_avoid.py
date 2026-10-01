"""Eval-only probe: observed own route bearing + repulsion from the observed predicted particle slots.

Uses only observation columns: 112:115 (own route bearing) and the four
predicted-clearance particle slots 36:76 (relative position/velocity, time and
clearance at closest approach). No learning. Diagnostic layouts only.
usage: probe_avoid.py SEED_FIRST SEED_LAST GAIN OUT_JSONL
"""
import json, sys
from pathlib import Path
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.geometric_control import direct_local_action
from scripts.train_mca_compiled import reset_with_valid_particles

cfg = DynamicsConfig.from_json(Path('configs/experiments/EXP_0040_OWN_BEARING_DYNAMICS.json'))
gain = float(sys.argv[3])


def avoid(nodes):
    push = np.zeros((len(nodes), 3))
    for k in range(4):
        s = 36+10*k
        valid = nodes[:, s+9] > 0
        rel = nodes[:, s:s+3]*1.5                      # mm, local frame
        relv = nodes[:, s+3:s+6]*cfg.robot_speed_mm_s  # mm/s
        t = nodes[:, s+7]*cfg.particle_prediction_horizon_s
        clear = nodes[:, s+8]                          # predicted clearance / safety margin
        closest = rel+t[:, None]*relv
        away = -closest/np.maximum(np.linalg.norm(closest, axis=1, keepdims=True), 1e-9)
        weight = np.clip(1-clear, 0, 1)**2*valid
        push += weight[:, None]*away
    return push


with open(sys.argv[4], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, seed)
        while True:
            nodes = obs['nodes'].astype(np.float64)
            local = nodes[:, 112:115]+gain*avoid(nodes)
            ex = direct_local_action(local, env)
            ex[~env.active[:env.num_robots]] = 0
            obs, r, term, trunc, info = env.step(ex)
            if term or trunc:
                break
        out.write(json.dumps(dict(seed=seed, success=bool(info['success']), elapsed=env.elapsed_s,
                                  removal=1-info['remaining_mass']/env.initial_mass.sum(), lost=info['lost_robots'],
                                  collision_free=bool(info['collision_free_success']),
                                  particle_events=info['episode_particle_collision_events']))+'\n'); out.flush()
        env.close()
