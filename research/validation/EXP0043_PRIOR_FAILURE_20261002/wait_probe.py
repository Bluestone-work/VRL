"""Diagnostic only: does WAITING (exactly-zero command) help the route+avoid prior avoid particles?
Executes the prior manually (identical to action_prior='own_route_bearing_avoid', tested in
tests/test_mca_action_prior.py) and zeroes robot i's command if any observed slot predicts
clearance < THRESH within T_MAX seconds.
usage: wait_probe.py SEED_FIRST SEED_LAST THRESH T_MAX OUT_JSONL
"""
import json, sys
from dataclasses import replace
import numpy as np
from environments.mca_physical_env import DynamicsConfig, observed_particle_repulsion
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.geometric_control import direct_local_action
from scripts.train_mca_compiled import reset_with_valid_particles

cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json'),
              action_prior='none')
thresh, tmax = float(sys.argv[3]), float(sys.argv[4])
with open(sys.argv[5], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, seed)
        waits = 0
        while True:
            nodes = obs['nodes'].astype(np.float64)
            act = direct_local_action(nodes[:, 112:115]+6.*observed_particle_repulsion(nodes, cfg), env)
            act[~env.active[:cfg.num_robots]] = 0
            stop = np.zeros(cfg.num_robots, bool)
            for k in range(4):
                s = 36+10*k
                t = nodes[:, s+7]*cfg.particle_prediction_horizon_s
                stop |= (nodes[:, s+9] > 0) & (nodes[:, s+8] < thresh) & (t < tmax)
            act[stop] = 0; waits += int(stop.sum())
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        out.write(json.dumps(dict(seed=seed, thresh=thresh, tmax=tmax, success=bool(info['success']),
                                  cf=bool(info['collision_free_success']), elapsed=env.elapsed_s, waits=waits))+'\n'); out.flush()
        env.close()
