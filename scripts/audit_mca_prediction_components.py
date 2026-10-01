"""Separate particle transport error from unknown future robot motion."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'research/runs/EXP_0032_TRAJECTORY_20260930a/trajectory/source_snapshot'
# Keep this historical diagnostic executable after newer observation changes.
if __name__ == '__main__':
    if not FROZEN.is_dir():
        raise FileNotFoundError('The archived EXP32 source snapshot is required')
    sys.path.insert(0, str(FROZEN))

import numpy as np
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_compiled import reset_with_valid_particles
from scripts.train_mca_physical import atomic_json

HORIZONS = (.1, .5, 1., 2.)


def metrics(values):
    x = np.asarray(values, float)
    if not len(x):
        return dict(count=0)
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), maximum=float(x.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warmup-steps', type=int, default=50)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    if args.warmup_steps < 0:
        raise ValueError('Warmup steps must be nonnegative')
    torch.set_num_threads(1)
    study = ROOT/'research/runs/EXP_0032_TRAJECTORY_20260930a'
    rows = []; sources = {}; states = 0
    for arm in ('trajectory', 'masked'):
        for seed in (42, 43, 44):
            path = study/arm/f'seed_{seed}/policy_500000.pt'
            payload = torch.load(path, map_location='cpu', weights_only=False)
            for name, digest in payload['meta']['source_sha256'].items():
                if hashlib.sha256((study/arm/'source_snapshot'/name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Changed diagnostic dependency: {name}')
            sources[f'{arm}/{seed}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            for index, saved in enumerate(payload['training_state']['envs']):
                if saved._done:
                    continue
                env = copy.deepcopy(saved); n = env.num_robots
                agent = make_physical_agent(env, seed=seed, hidden_dim=payload['meta']['hidden_dim'])
                agent.load(path, load_optimizers=False); agent.actor.eval(); agent.critic.eval()
                obs = env._observation()
                # Saved training velocities follow stochastic actions. Settle
                # under the evaluated deterministic policy before auditing CV.
                for _ in range(args.warmup_steps):
                    if env._done:
                        break
                    _, _, _, action, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
                    obs, _, _, _, _ = env.step(action)
                if env._done:
                    continue
                ids = np.flatnonzero(env.active[n:])+n
                if not len(ids):
                    continue
                states += 1
                initial = env.positions_mm.copy(); rv = env.velocity_mm_s.copy()
                pv = env.transport.velocity_mm_s(initial[ids], env.edges[ids], env.solution)
                current = initial[ids][None]-initial[:n, None]
                dv = pv[None]-rv[:, None]
                t = np.clip(-np.sum(current*dv, -1)/np.maximum(np.sum(dv*dv, -1), 1e-12), 0, 2.)
                selected = np.argsort(np.linalg.norm(current+t[:, :, None]*dv, axis=-1), axis=1)[:, :4]
                positions = initial[ids].copy(); edges = env.edges[ids].copy()
                active = env.active[ids].copy(); previous = 0.; forecasts = []
                for horizon in HORIZONS:
                    prediction = env._advance_particle_prediction(positions, edges, env.body_radius[ids], active, horizon-previous)
                    positions, edges, active = prediction.positions_mm, prediction.edge, prediction.active
                    forecasts.append((positions.copy(), active.copy())); previous = horizon
                start = env.elapsed_s; obs = env._observation()
                for horizon, (predicted, valid) in zip(HORIZONS, forecasts):
                    while env.elapsed_s < start+horizon-1e-9 and not env._done:
                        _, _, _, action, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
                        obs, _, _, _, _ = env.step(action)
                    if env.elapsed_s < start+horizon-1e-9:
                        break
                    common = valid & env.active[ids]
                    robot_pred = initial[:n]+rv*horizon
                    linear = initial[ids]+pv*horizon
                    base = dict(arm=arm, seed=seed, state=index, horizon_s=horizon)
                    for j in np.flatnonzero(common):
                        rows.append(dict(**base, kind='particle', near=False,
                            route_error_mm=float(np.linalg.norm(predicted[j]-env.positions_mm[ids[j]])),
                            linear_error_mm=float(np.linalg.norm(linear[j]-env.positions_mm[ids[j]]))))
                    for i in np.flatnonzero(env.agent_mask):
                        rows.append(dict(**base, kind='robot', near=False,
                            route_error_mm=float(np.linalg.norm(robot_pred[i]-env.positions_mm[i])),
                            linear_error_mm=float(np.linalg.norm(initial[i]-env.positions_mm[i]))))
                        for j in selected[i]:
                            if not common[j]:
                                continue
                            actual = env.positions_mm[ids[j]]-env.positions_mm[i]
                            rows.append(dict(**base, kind='relative',
                                near=bool(min(np.linalg.norm(current[i,j]), np.linalg.norm(actual)) < .5),
                                route_error_mm=float(np.linalg.norm(predicted[j]-robot_pred[i]-actual)),
                                linear_error_mm=float(np.linalg.norm(linear[j]-robot_pred[i]-actual))))
            print(f'Completed {arm} seed {seed}', flush=True)
    summary = dict(states=states, checkpoint_sha256=sources, world_model=False,
        deterministic_warmup_steps=args.warmup_steps,
        scope='Saved states of six EXP32 final models, future deterministic policy execution; correlated diagnostic samples, not independent evaluation episodes.',
        definitions={'particle': 'route transport vs constant particle velocity',
                     'robot': 'constant robot velocity error (route column) vs stationary robot error (linear column)',
                     'relative': 'four linear-risk-ranked particle-robot pairs; route or linear particles minus constant-velocity robot',
                     'near': 'current or actual future center distance <0.5 mm; diagnostic selection uses future only after rollout, never policy observation'},
        errors={})
    for horizon in HORIZONS:
        summary['errors'][str(horizon)] = {}
        for kind in ('particle', 'robot', 'relative', 'relative_near'):
            subset = [r for r in rows if r['horizon_s']==horizon and r['kind']==kind.replace('_near', '')
                      and (not kind.endswith('_near') or r['near'])]
            summary['errors'][str(horizon)][kind] = {key:metrics([r[key] for r in subset])
                for key in ('route_error_mm', 'linear_error_mm')}
    # Difficulty strata use initial geometry only and the existing paired evaluations.
    protocol = json.loads((study/'trajectory/protocol.json').read_text())
    env = CompiledMCAPhysicalEnv(DynamicsConfig.from_json(FROZEN/protocol['physics_config']))
    difficulty = []
    for index in range(100):
        reset_with_valid_particles(env, protocol['validation_seed_base']+index)
        distance = env._target_distances()
        difficulty.append((float(distance.min(axis=0).max()), index))
    summary['initial_distance_strata'] = {}
    for label, indices in zip(('near_third','middle_third','far_third'), np.array_split(np.argsort([x[0] for x in difficulty]), 3)):
        groups = {}
        for arm in ('masked','trajectory'):
            records = [json.loads((study/arm/f'seed_{seed}/evaluation_500000.json').read_text())['episodes'][i]
                       for seed in (42,43,44) for i in indices]
            groups[arm] = dict(success_rate=float(np.mean([r['success'] for r in records])),
                              collision_free_success_rate=float(np.mean([r['collision_free_success'] for r in records])))
        summary['initial_distance_strata'][label] = dict(layout_count=len(indices),
            maximum_nearest_robot_geodesic_distance_mm=metrics([difficulty[i][0] for i in indices]), results=groups)
    summary['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),FROZEN/'environments/mca_obstacle_forecast.py',FROZEN/'environments/mca_physical_env.py']}
    with (args.out/'errors.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    atomic_json(args.out/'summary.json', summary)
    print(json.dumps(summary['errors'], indent=2), flush=True)


if __name__ == '__main__':
    main()
