"""Eval-only: route+avoid prior (gain 6, no learning) on one anatomy, its registered diagnostic split.

Diagnostic layouts only (configs/evaluation_splits.json); no validation or test layouts are touched.
usage: probe.py ANATOMY COUNT OUT_JSONL
"""
import json, sys
from dataclasses import replace
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.mca_eval_splits import split
from scripts.train_mca_compiled import reset_with_valid_particles

name, count = sys.argv[1], int(sys.argv[2])
base, size = split('diagnostic', name)
assert count <= size
cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json'), anatomy=name)
with open(sys.argv[3], 'a') as out:
    for seed in range(base, base+count):
        try:
            env = CompiledMCAPhysicalEnv(cfg)
            reset_with_valid_particles(env, seed)
            while True:
                _, _, term, trunc, info = env.step(np.zeros((cfg.num_robots, 3)))
                if term or trunc:
                    break
            row = dict(anatomy=name, seed=seed, success=bool(info['success']), elapsed=env.elapsed_s,
                       removal=float(1-info['remaining_mass']/env.initial_mass.sum()), lost=int(info['lost_robots']),
                       collision_free=bool(info['collision_free_success']),
                       mm_per_unit=float(env.tree.physical_mm_per_unit))
            env.close()
        except Exception as e:
            row = dict(anatomy=name, seed=seed, error=f'{type(e).__name__}: {str(e)[:120]}')
        out.write(json.dumps(row)+'\n'); out.flush()
