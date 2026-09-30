"""Paired reference/compiled replay without touching live training."""
import argparse,copy,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import MCAPhysicalEnv,DynamicsConfig
from scripts.train_mca_physical import reset_with_valid_particles,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(seeds=[42,43,44],fractions=[.1,.05,.025],position_tolerance_mm=.001,mass_tolerance=2e-5,
                  exit_time_tolerance_s=2e-5,reward_tolerance=1e-4,require_exact_masks_edges_and_exit_nodes=True,
                  note='0.001 mm paired-backend bound is stricter than original unchanged 0.05 mm convergence gate; not bitwise equality')
    atomic_json(out/'protocol.json',protocol);records=[]
    # Compile before measuring.
    warm=CompiledMCAPhysicalEnv();reset_with_valid_particles(warm,42);warm.step(np.zeros((5,3)))
    for seed in protocol['seeds']:
        for fraction in protocol['fractions']:
            ref=MCAPhysicalEnv(replace(DynamicsConfig.from_json(),spatial_fraction=fraction))
            reset_with_valid_particles(ref,seed);fast=copy.deepcopy(ref);fast.__class__=CompiledMCAPhysicalEnv
            rng=np.random.default_rng(seed)
            rec=dict(seed=seed,fraction=fraction,max_position_mm=0.,max_mass=0.,max_reward=0.,max_exit_time_s=0.,reference_s=0.,compiled_s=0.,exact_discrete=True,steps=0)
            while True:
                action=rng.uniform(-1,1,(5,3));t=time.perf_counter();ro,rr,rt,rx,ri=ref.step(action);rec['reference_s']+=time.perf_counter()-t
                t=time.perf_counter();fo,fr,ft,fx,fi=fast.step(action);rec['compiled_s']+=time.perf_counter()-t
                rec['max_position_mm']=max(rec['max_position_mm'],float(np.linalg.norm(ref.positions_mm-fast.positions_mm,axis=1).max()))
                rec['max_mass']=max(rec['max_mass'],float(np.abs(ref.masses-fast.masses).max()))
                rec['max_reward']=max(rec['max_reward'],abs(rr-fr))
                finite=np.isfinite(ref.exit_time_s)&np.isfinite(fast.exit_time_s)
                if finite.any(): rec['max_exit_time_s']=max(rec['max_exit_time_s'],float(np.abs(ref.exit_time_s[finite]-fast.exit_time_s[finite]).max()))
                rec['exact_discrete'] &= bool(np.array_equal(ref.edges,fast.edges) and np.array_equal(ref.active,fast.active) and np.array_equal(ref.exit_node,fast.exit_node) and (rt,rx)==(ft,fx))
                rec['steps']+=1
                if rt or rx or ft or fx: break
            rec['passed']=bool(rec['exact_discrete'] and rec['max_position_mm']<=protocol['position_tolerance_mm'] and rec['max_mass']<=protocol['mass_tolerance'] and rec['max_reward']<=protocol['reward_tolerance'] and rec['max_exit_time_s']<=protocol['exit_time_tolerance_s'])
            records.append(rec);print(json.dumps(rec),flush=True)
    atomic_json(out/'summary.json',dict(passed=all(r['passed'] for r in records),cases=records,
                physics_inputs_changed=False,python_reference_preserved=True))
    if not all(r['passed'] for r in records): raise SystemExit(1)


if __name__=='__main__': main()
