"""Diagnostic only: route+avoid prior (gain 6) through the env's own prior path, zero policy,
with command_speed=MODE. usage: bounded_probe.py SEED_FIRST SEED_LAST MODE OUT_JSONL"""
import json, sys
from dataclasses import replace
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_compiled import reset_with_valid_particles
cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json'), command_speed=sys.argv[3])
with open(sys.argv[4], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg); reset_with_valid_particles(env, seed)
        while True:
            _, _, term, trunc, info = env.step(np.zeros((cfg.num_robots, 3)))
            if term or trunc: break
        out.write(json.dumps(dict(seed=seed, mode=sys.argv[3], success=bool(info['success']), cf=bool(info['collision_free_success']), elapsed=env.elapsed_s))+'\n'); out.flush(); env.close()
