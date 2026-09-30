"""EXP_0010-0013 prospective validation on the sealed EXP_0002 manifest.

Follows the EXP_0005 protocol: deterministic policy, same 140-record
validation split, identity asserted per episode. The evaluator never opens
the test split.

Per-arm deviations from the plain EXP_0005 path, matching each arm's
training conditions:
  * EXP_0011_SEPARATED_INIT: initialization_mode="separated" at reset
  * EXP_0012_PARTICLES_*: dynamic particles with the arm's count
  * EXP_0013_CONNECTIVITY_ALLOC: allocator re-plans every step
  * EXP_0010_VARIABLE_N: 8-slot agent on a 5-robot env (GAT is per-node)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import resolve_pool
from marl.connectivity_allocator import allocate
from marl.mappo_advanced import MAPPOAdvanced
from scripts.research_protocol import geometry_hash, initial_identity
from scripts.train_vector_mappo import build_context

ROOT = Path('/home/wj/桌面/vascular_marl_local.tar.')
OUT = ROOT / 'research/runs/EXP_0010_0013_prospective'
RUNS = ROOT / 'research/runs'
STUDY = json.loads((ROOT / 'configs/experiments/EXP_0005_DIRECT_LOCAL.json').read_text())

ARMS = [
    ('EXP_0010_CONTROL', dict()),
    ('EXP_0010_VARIABLE_N', dict()),
    ('EXP_0011_CONTROL', dict()),
    ('EXP_0011_SEPARATED_INIT', dict(init='separated')),
    ('EXP_0012_CONTROL', dict()),
    ('EXP_0012_PARTICLES_SPARSE', dict(particles=8)),
    ('EXP_0012_PARTICLES_MODERATE', dict(particles=16)),
    ('EXP_0012_PARTICLES_DENSE', dict(particles=32)),
    ('EXP_0013_CONTROL', dict()),
    ('EXP_0013_CONNECTIVITY_ALLOC', dict(allocator='connectivity_aware')),
]

METRICS = ('success', 'removal_rate', 'steps', 'wall_contact_rate', 'collision_rate')


def env_settings(cfg, opts, particle_seed, episode_seed):
    kw = dict(
        num_robots=cfg['robots'], num_clots=cfg['clots'], horizon=cfg['horizon'],
        robot_radius=cfg['robot_radius'], obs_mode=cfg['obs_mode'],
        reward_mode=cfg['reward_mode'], contact_mode=cfg['contact_mode'],
        coverage_bonus=cfg['coverage_bonus'], step_cost=cfg['step_cost'],
        approach_scale=cfg['approach_scale'], reward_double_count=cfg['reward_double_count'],
        control_margin=not cfg['no_control_margin'], randomize_scenario=False,
    )
    if opts.get('init') == 'separated':
        kw['initialization_mode'] = 'separated'
    if 'particles' in opts:
        kw['dynamic_intravascular_particles'] = True
        kw['particle_count'] = opts['particles']
        kw['particle_seed'] = particle_seed
    return kw


def build_agent(checkpoint: Path, cfg, device: str):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta = saved['meta']
    assert meta.get('control_mode') == 'local'
    assert meta.get('action_semantics') == 'direct_local_frenet'
    agent = MAPPOAdvanced(
        n_agents=meta['n_agents'], obs_dim=meta['obs_dim'], action_dim=meta['action_dim'],
        architecture=meta['architecture'], state_dim=meta.get('state_dim', 0),
        hidden_dim=meta.get('hidden_dim', 128), num_layers=meta.get('num_layers', 2),
        num_heads=meta.get('num_heads', 4), control_mode='local',
        residual_scale=meta.get('residual_scale', 0.2),
        guidance_speed=meta.get('guidance_speed', 0.65),
        critic_value_mode=meta.get('critic_value_mode', 'v'),
        dropout=meta.get('dropout', 0.0), device=device,
    )
    agent.load(str(checkpoint), load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    return agent


def episode(agent, env, target, opts, trace_path: Path):
    identity = initial_identity(env, target['episode_seed'])
    # Geometry must match the manifest. The initial-state hash is only
    # asserted for arms whose reset is identical to the manifest's legacy
    # reset: separated init redefines the spawn rule (state hash legitimately
    # differs), particles add obstacle state, and the allocator only touches
    # task assignments after reset.
    strict_state = not (opts.get('init') == 'separated' or 'particles' in opts)
    for key in ('complete_geometry_sha256', 'initial_state_sha256', 'active_clots', 'stations', 'branches'):
        if identity[key] != target[key]:
            if key == 'initial_state_sha256' and not strict_state:
                continue
            raise AssertionError(
                f'prospective identity mismatch for {target["scenario"]}/{target["episode_seed"]}: {key}')
    obs, _ = env.reset(seed=target['episode_seed'])
    n = env.num_robots
    wall = pair = 0
    for step in range(env.horizon):
        if opts.get('allocator'):
            res = allocate(env, opts['allocator'])
            env.set_task_assignments(res.assignments)
        ctx = build_context(env, obs)
        state = obs['clot_state'].reshape(-1)
        local_action, _, _ = agent.act(obs['nodes'], ctx, state, deterministic=True)
        executed = agent.env_action(local_action, obs, env)
        obs, reward, terminated, truncated, info = env.step(executed)
        wall += int(info['wall_collisions'])
        pair += int(info['robot_collisions'])
        if terminated or truncated:
            break
    steps = int(info['steps']) if 'steps' in info else step + 1
    return {
        'scenario': target['scenario'], 'episode_seed': int(target['episode_seed']),
        'success': float(bool(info['success'])),
        'removal_rate': float(info['removal_rate']),
        'steps': steps,
        'completion_steps': steps if info['success'] else None,
        'wall_contact_rate': float(wall / (n * steps)),
        'collision_rate': float(pair / (max(n * (n - 1) // 2, 1) * steps)),
    }


def mean_sd(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return {'mean': None, 'sample_sd': None, 'n': 0}
    arr = np.asarray(values, dtype=float)
    return {'mean': float(arr.mean()),
            'sample_sd': float(arr.std(ddof=1)) if len(values) > 1 else 0.0,
            'n': int(len(values))}


def main():
    manifest = json.loads((ROOT / STUDY['prospective_validation']['manifest']).read_text())
    assert manifest.get('policy_evaluations') == 0, 'manifest already consumed'
    records_by_scenario = {}
    for row in manifest['records']:
        records_by_scenario.setdefault(row['scenario'], []).append(row)
    cfg = STUDY['training']
    device = 'cuda:0'
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm, opts in ARMS:
        arm_dir = OUT / arm
        arm_dir.mkdir(exist_ok=True)
        per_seed = []
        summaries = [arm_dir / f'seed_{s}_summary.json' for s in STUDY['seeds']]
        if all(p.exists() for p in summaries):
            per_seed = [json.loads(p.read_text()) for p in summaries]
            results[arm] = {
                key: {
                    'mean': float(np.mean([s[key]['mean'] for s in per_seed])),
                    'sample_sd': float(np.std([s[key]['mean'] for s in per_seed], ddof=1)),
                } for key in METRICS
            }
            print(f'== {arm} (cached): success {results[arm]["success"]["mean"]:.3f} ± {results[arm]["success"]["sample_sd"]:.3f}', flush=True)
            continue
        for seed in STUDY['seeds']:
            checkpoint = RUNS / arm / f'seed_{seed}' / 'final_policy.pt'
            agent = build_agent(checkpoint, cfg, device)
            rows = []
            for scenario, targets in sorted(records_by_scenario.items()):
                for target in targets:
                    particle_seed = 82000000 + (target['episode_seed'] % 100000)
                    env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario],
                                            seed=target['episode_seed'],
                                            **env_settings(cfg, opts, particle_seed, target['episode_seed']))
                    try:
                        rows.append(episode(agent, env, target, opts, None))
                    finally:
                        env.close()
            summary = {'training_seed': seed, **{key: mean_sd(rows, key) for key in METRICS}}
            (arm_dir / f'seed_{seed}_summary.json').write_text(json.dumps(summary, indent=1))
            (arm_dir / f'seed_{seed}_episodes.json').write_text(
                json.dumps(rows, indent=None))
            per_seed.append(summary)
            print(f'{arm} seed {seed}: success={summary["success"]["mean"]:.3f}', flush=True)
        aggregate = {
            key: {
                'mean': float(np.mean([s[key]['mean'] for s in per_seed])),
                'sample_sd': float(np.std([s[key]['mean'] for s in per_seed], ddof=1)),
            } for key in METRICS
        }
        results[arm] = aggregate
        print(f'== {arm}: success {aggregate["success"]["mean"]:.3f} ± {aggregate["success"]["sample_sd"]:.3f}', flush=True)
    (OUT / 'aggregate.json').write_text(json.dumps(results, indent=1))
    print('DONE')


if __name__ == '__main__':
    main()
