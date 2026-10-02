"""Diagnostic only: gated prior (deadzone DZ, residual 1) with Gaussian residual noise of std SIGMA
(tanh-squashed, as the actor samples). Measures how exploration perturbs the prior.
usage: noise_probe.py SEED_FIRST SEED_LAST SIGMA DZ OUT_JSONL"""
import json, sys
from dataclasses import replace
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_compiled import reset_with_valid_particles
sigma, dz = float(sys.argv[3]), float(sys.argv[4])
cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json'),
              action_prior='own_route_bearing_avoid_gated', action_stop_deadzone=dz, action_residual_scale=1.)
with open(sys.argv[5], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        rng = np.random.default_rng(seed)
        env = CompiledMCAPhysicalEnv(cfg); reset_with_valid_particles(env, seed)
        while True:
            _, _, term, trunc, info = env.step(np.tanh(sigma*rng.standard_normal((cfg.num_robots, 3))))
            if term or trunc: break
        out.write(json.dumps(dict(seed=seed, sigma=sigma, dz=dz, success=bool(info['success']), cf=bool(info['collision_free_success']), elapsed=env.elapsed_s))+'\n'); out.flush(); env.close()
