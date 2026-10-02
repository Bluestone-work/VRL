"""Diagnostic only: where and why the route+avoid prior (gain 6, no learning) collides or times out.
usage: trace.py SEED_FIRST SEED_LAST OUT_JSONL
"""
import json, sys
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_compiled import reset_with_valid_particles

cfg = DynamicsConfig.from_json('configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json')
with open(sys.argv[3], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, seed)
        events = []
        step = 0
        while True:
            nodes = obs['nodes'].astype(np.float64)
            pre = dict(min_clear=[float(nodes[i, 44]) if nodes[i, 45] > 0 else None for i in range(cfg.num_robots)],
                       remaining=int((env.masses > 1e-9).sum()))
            obs, r, term, trunc, info = env.step(np.zeros((cfg.num_robots, 3)))
            step += 1
            ev = np.asarray(info.get('particle_collision_events', np.zeros(cfg.num_robots)))
            for i in np.nonzero(ev)[0]:
                rel = nodes[i, 36:39]*1.5; relv = nodes[i, 39:42]*cfg.robot_speed_mm_s
                events.append(dict(step=step, robot=int(i), remaining=pre['remaining'],
                                   slot0_clear=pre['min_clear'][i], slot0_rel_speed=float(np.linalg.norm(relv)),
                                   slot0_dist=float(np.linalg.norm(rel)), own_dist=float(nodes[i, 108+1]) if nodes.shape[1] > 109 else None))
            if term or trunc:
                break
        out.write(json.dumps(dict(seed=seed, success=bool(info['success']), cf=bool(info['collision_free_success']),
                                  elapsed=env.elapsed_s, n_events=len(events), lost=int(info['lost_robots']),
                                  removal=float(1-info['remaining_mass']/env.initial_mass.sum()), events=events[:20]))+'\n'); out.flush()
        env.close()
