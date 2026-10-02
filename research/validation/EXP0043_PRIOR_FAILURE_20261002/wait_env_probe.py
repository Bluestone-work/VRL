"""Diagnostic only: the env's own_route_bearing_avoid_wait prior with zero policy. usage: wait_env_probe.py A B CONFIG OUT"""
import json, sys
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_compiled import reset_with_valid_particles
cfg = DynamicsConfig.from_json(sys.argv[3])
with open(sys.argv[4], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg); reset_with_valid_particles(env, seed)
        while True:
            _, _, term, trunc, info = env.step(np.zeros((cfg.num_robots, 3)))
            if term or trunc: break
        out.write(json.dumps(dict(seed=seed, success=bool(info['success']), cf=bool(info['collision_free_success']), elapsed=env.elapsed_s))+'\n'); out.flush(); env.close()
