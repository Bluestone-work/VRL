"""Audit causal predictions against subsequent policy rollouts; no training."""
import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from environments.mca_obstacle_forecast import forecast_particles, HORIZONS_S
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_physical import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', type=Path, required=True)
    args = p.parse_args();args.out.mkdir(parents=True, exist_ok=False);torch.set_num_threads(1)
    rows = [];sources = {};state_count = 0;mask_checks = [];changed_flow_horizons = 0
    for seed in (42,43,44):
        checkpoint = ROOT / f'research/runs/EXP_0030_PPO_REPAIR_20260930a/b_low_actor_lr/seed_{seed}/policy_500000.pt'
        sources[str(seed)] = dict(path=str(checkpoint.relative_to(ROOT)), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for index, saved in enumerate(payload['training_state']['envs']):
            env = copy.deepcopy(saved)
            if env._done:continue
            ids = np.flatnonzero(env.active[env.num_robots:]) + env.num_robots
            if not len(ids):continue
            before = env.positions_mm.copy();masses = env.masses.copy();rng = copy.deepcopy(env.np_random.bit_generator.state)
            predicted, valid, exits = forecast_particles(env, ids)
            np.testing.assert_array_equal(before, env.positions_mm);np.testing.assert_array_equal(masses, env.masses)
            assert rng == env.np_random.bit_generator.state
            linear_velocity = env.transport.velocity_mm_s(before[ids],env.edges[ids],env.solution)
            agent = make_physical_agent(env, seed=seed, hidden_dim=payload['meta']['hidden_dim'], device='cpu')
            agent.load(checkpoint, load_optimizers=False);agent.actor.eval();agent.critic.eval()
            started = env.elapsed_s;state_count += 1
            obs = env._observation()
            for k,horizon in enumerate(HORIZONS_S):
                while env.elapsed_s < started+horizon-1e-9 and not env._done:
                    _,_,_,action,_,_ = physical_policy_action(agent,env,obs,deterministic=True)
                    obs,_,_,_,_ = env.step(action)
                if env.elapsed_s < started+horizon-1e-9:break
                changed = bool(np.any(env.masses != masses));changed_flow_horizons += int(changed)
                mask_checks.extend((valid[k] == env.active[ids]).tolist())
                for j,body in enumerate(ids):
                    if not (valid[k,j] and env.active[body]):continue
                    actual = env.positions_mm[body]
                    rows.append(dict(training_seed=seed,env_index=index,particle_id=int(body-env.num_robots),
                        horizon_s=horizon,flow_changed=changed,
                        route_error_mm=float(np.linalg.norm(predicted[k,j]-actual)),
                        linear_error_mm=float(np.linalg.norm(before[body]+horizon*linear_velocity[j]-actual))))
    if not rows:raise RuntimeError('No comparable predictions')
    summary = dict(scope='24 saved training-environment states from3 parent policies; diagnostic accuracy, not collision-rate or generalization evidence',
        states=state_count,predictions=len(rows),flow_changed_state_horizons=changed_flow_horizons,
        validity_mask_agreement=float(np.mean(mask_checks)),world_model=False,sources=sources,
        assumptions='Frozen current flow/lumen; future actions and lysis are not available to predictor; comparisons below use subsequent real deterministic policy rollout',
        error_by_horizon={},observation_benchmark={})
    for horizon in HORIZONS_S:
        subset=[r for r in rows if r['horizon_s']==horizon];metrics={}
        for name in ('route_error_mm','linear_error_mm'):
            values=np.asarray([r[name] for r in subset]);metrics[name]=dict(mean=float(values.mean()),
                median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(values.max()))
        summary['error_by_horizon'][str(horizon)]=dict(count=len(subset),**metrics)
    cfg = DynamicsConfig.from_json(ROOT/'configs/experiments/EXP_0029_MCA_ALL_RANDOM_DYNAMICS.json')
    for mode in ('predictive_four','trajectory_four','trajectory_four_masked'):
        env=CompiledMCAPhysicalEnv(replace(cfg,obstacle_observation=mode));env.reset(seed=832000000)
        env._observation();cpu=[];wall=[]
        for _ in range(3):
            a=time.process_time();b=time.perf_counter()
            for _ in range(100):env._observation()
            cpu.append((time.process_time()-a)*10);wall.append((time.perf_counter()-b)*10)
        summary['observation_benchmark'][mode]=dict(cpu_ms_per_call=statistics.median(cpu),
            wall_ms_per_call=statistics.median(wall),scope='Single-process fixed-state observation only; not end-to-end training throughput')
    paths=[Path(__file__),ROOT/'environments/mca_obstacle_forecast.py',ROOT/'environments/mca_physical_env.py',ROOT/'environments/mca_compiled.py']
    summary['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (args.out/'prediction_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    atomic_json(args.out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','source_sha256')},indent=2),flush=True)


if __name__=='__main__':main()
