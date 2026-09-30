"""Paired deterministic/sampled execution diagnostics; never changes training scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_physical import atomic_json, reset_with_valid_particles

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--action-seed-base', type=int, default=710000000)
    args = p.parse_args()
    source = args.protocol.resolve().parents[2]
    if source.name == 'source_snapshot' and source != ROOT:
        os.chdir(source)
        os.execv(sys.executable, [sys.executable, '-m', 'scripts.diagnose_mca_policy_execution', *sys.argv[1:]])
    if args.out.exists() or args.count < 1:
        raise ValueError('Require a new output directory and positive count')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for name, digest in payload['meta']['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source mismatch: {name}')
    protocol = json.loads(args.protocol.read_text())
    cfg = DynamicsConfig.from_json(ROOT / protocol['physics_config'])
    env = CompiledMCAPhysicalEnv(cfg)
    agent = make_physical_agent(env, seed=payload['meta']['training_seed'],
                               hidden_dim=protocol['hidden_dim'], device=args.device,
                               ppo=protocol.get('ppo'))
    agent.load(args.checkpoint, load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    args.out.mkdir(parents=True)
    atomic_json(args.out / 'provenance.json', dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        count=args.count, action_seed_base=args.action_seed_base, training_seed=payload['meta']['training_seed'],
        world_model=False, scope='Paired execution diagnostic; does not replace canonical deterministic validation'))
    summary = {}
    for mode in ('deterministic', 'sampled'):
        episodes = []
        for index in range(args.count):
            seed = protocol['validation_seed_base'] + index
            torch.manual_seed(args.action_seed_base + index)
            env = CompiledMCAPhysicalEnv(cfg)
            obs, reset_info = reset_with_valid_particles(env, seed)
            minimum = np.full((env.num_robots, env.num_clots), np.inf)
            contact = np.zeros(env.num_robots)
            action_norm = 0.; steps = 0; trace = []; heartbeat = 0.
            while True:
                positions = env.positions_mm[:env.num_robots].copy()
                distance = np.linalg.norm(positions[:, None] - env.clot_positions_mm[None], axis=-1)
                minimum = np.minimum(minimum, distance)
                proposal, _, _, execution, _, _ = physical_policy_action(
                    agent, env, obs, deterministic=(mode == 'deterministic'))
                action_norm += float(np.linalg.norm(proposal, axis=-1).mean())
                if steps % 50 == 0:
                    trace.append(dict(elapsed_s=float(env.elapsed_s), masses=env.masses.tolist(),
                                      positions_mm=positions.tolist(), actions=proposal.tolist(),
                                      target_distance_mm=env._target_distances().tolist()))
                obs, reward, term, trunc, info = env.step(execution)
                # Integrator contact time is the physical quantity, not display overlap.
                contact += np.asarray(info['contact_s'])
                steps += 1
                if time.monotonic() - heartbeat > 10:
                    atomic_json(args.out / 'status.json', dict(phase='running', mode=mode, episode=index,
                                count=args.count, elapsed_s=float(env.elapsed_s), timestamp=time.time(), pid=os.getpid()))
                    heartbeat = time.monotonic()
                if term or trunc:
                    break
            record = dict(seed=seed, reset_info=reset_info, success=bool(info['success']),
                          collision_free_success=bool(info['collision_free_success']),
                          particle_collision_events=int(info['episode_particle_collision_events']),
                          removal_fraction=float(1-info['remaining_mass']/env.initial_mass.sum()),
                          lost_robots=int(info['lost_robots']), elapsed_s=float(info['elapsed_s']),
                          reason=info['termination_reason'], final_masses=env.masses.tolist(),
                          minimum_euclidean_distance_mm=minimum.tolist(), contact_s=contact.tolist(),
                          mean_action_norm=action_norm/steps, steps=steps)
            episodes.append(record)
            # Infinity denotes an already cleared target; use JSON null in diagnostic traces.
            for frame in trace:
                frame['target_distance_mm'] = [[x if np.isfinite(x) else None for x in row]
                                                for row in frame['target_distance_mm']]
            atomic_json(args.out / f'{mode}_{seed}_trace.json', dict(record=record, trace=trace))
            env.close()
        metrics = {k:float(np.mean([e[k] for e in episodes]))
                   for k in ('success', 'collision_free_success', 'removal_fraction')}
        metrics['particle_collision_episode_rate'] = float(np.mean([e['particle_collision_events'] > 0 for e in episodes]))
        atomic_json(args.out / f'{mode}.json', dict(episodes=episodes, metrics=metrics))
        summary[mode] = metrics
        print(json.dumps(dict(mode=mode, **metrics)), flush=True)
    atomic_json(args.out / 'summary.json', summary)
    atomic_json(args.out / 'status.json', dict(phase='completed', timestamp=time.time(), pid=os.getpid()))


if __name__ == '__main__':
    main()
