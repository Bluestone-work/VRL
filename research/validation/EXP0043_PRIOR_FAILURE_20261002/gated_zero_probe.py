"""Diagnostic only: gated prior with zero policy; how often is the prior itself inside the deadzone?
usage: gated_zero_probe.py SEED_FIRST SEED_LAST DEADZONE OUT_JSONL"""
import json, sys
from dataclasses import replace
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_compiled import reset_with_valid_particles
dz = float(sys.argv[3])
cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json'),
              action_prior='own_route_bearing_avoid_gated', action_stop_deadzone=dz, action_residual_scale=1.)
with open(sys.argv[4], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg); reset_with_valid_particles(env, seed); stops = 0; steps = 0
        while True:
            z = np.zeros((cfg.num_robots, 3)); c = env._apply_action_prior(z)
            stops += int(((np.linalg.norm(c, axis=1) == 0) & env.active[:cfg.num_robots]).sum()); steps += 1
            _, _, term, trunc, info = env.step(z)
            if term or trunc: break
        out.write(json.dumps(dict(seed=seed, dz=dz, success=bool(info['success']), cf=bool(info['collision_free_success']), elapsed=env.elapsed_s, stop_frac=stops/steps/cfg.num_robots))+'\n'); out.flush(); env.close()
