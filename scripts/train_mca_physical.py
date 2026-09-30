"""EXP23: fresh pure MAPPO training on the frozen EXP22B engineering task.

Atomic restart checkpoints include simulator and RNG state. Original physical
readiness flags remain unchanged: this is an authorized optimization baseline,
not a claim of calibration or a reachable all-clot-success task.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent, physical_policy_action, store_physical_transition
from scripts.mca_training_gate import require_all_clear_capacity, require_feasibility_certificate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / 'configs/experiments/EXP_0023_MCA_PURE_RL.json'


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes(protocol_path, physics_path):
    paths = [protocol_path, physics_path, Path(__file__), ROOT/'scripts/mca_training_gate.py']
    for folder in ('environments', 'marl'):
        paths.extend(sorted((ROOT / folder).rglob('*.py')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def reset_with_valid_particles(env, seed):
    """Reject only sampled tracer placements that snap to an incompatible edge.

    Use the configured robot initializer; record every accepted/rejected seed.
    The task RNG key remains independent from the policy RNG.
    """
    rejected = []
    for attempt in range(100):
        actual = seed if attempt == 0 else int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
        try:
            obs, info = env.reset(seed=actual)
            return obs, dict(requested_seed=seed, accepted_seed=actual, rejected_particle_seeds=rejected,
                             **env.initialization_record())
        except ValueError as exc:
            if str(exc) != 'Initial body does not fit the obstructed lumen':
                raise
            _, radius, distance, _ = env.transport.coordinates(env.positions_mm, env.edges, env.solution)
            bad = np.flatnonzero(distance + env.body_radius > radius + 1e-7)
            if np.any(bad < env.num_robots):
                raise
            rejected.append(actual)
    raise RuntimeError('Could not sample valid passive particles after 100 attempts')


def episode_accumulator():
    return dict(reward=0., removed_mass=0., contact_s=0., wall_contact_s=0., particle_contact_s=0., steps=0)


def evaluate(agent, cfg, protocol, count):
    # Deterministic act still computes critic; restore modes and RNG so evaluation
    # never changes the future training trajectory.
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = agent.actor.training, agent.critic.training
    agent.actor.eval(); agent.critic.eval()
    records = []
    try:
        for index in range(count):
            env = MCAPhysicalEnv(cfg)
            seed = protocol['validation_seed_base'] + index
            obs, reset_info = reset_with_valid_particles(env, seed)
            reward = 0.
            while True:
                _, _, _, execution, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
                obs, r, term, trunc, info = env.step(execution)
                reward += r
                if term or trunc:
                    break
            records.append(dict(seed=seed, reset_info=reset_info, reward=reward, success=info['success'],
                                removal_fraction=1-info['remaining_mass']/env.initial_mass.sum(),
                                lost_robots=info['lost_robots'], elapsed_s=info['elapsed_s'],
                                reason=info['termination_reason']))
            env.close()
    finally:
        agent.actor.train(modes[0]); agent.critic.train(modes[1])
        np.random.set_state(np_state); torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return dict(episodes=records, success_rate=float(np.mean([r['success'] for r in records])),
                mean_removal_fraction=float(np.mean([r['removal_fraction'] for r in records])))


def train(args):
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    physics_path = ROOT / protocol['physics_config']
    cfg = DynamicsConfig.from_json(physics_path)
    gate_env = MCAPhysicalEnv(cfg)
    reset_with_valid_particles(gate_env, args.seed)
    feasibility = require_all_clear_capacity(gate_env,
        allow_unreachable_baseline=args.allow_unreachable_baseline)
    hashes = source_hashes(protocol_path, physics_path)
    if cfg.contact_model in ('stenosis_surface', 'localized_point'):
        feasibility.update(require_feasibility_certificate(protocol, ROOT))
    out = Path(args.out).resolve()
    if args.resume:
        if not out.is_dir():
            raise ValueError('Resume directory does not exist')
    else:
        out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    env = MCAPhysicalEnv(cfg)
    agent = make_physical_agent(env, seed=args.seed, hidden_dim=protocol['hidden_dim'], device=args.device)
    agent.meta.update(training_experiment=protocol['experiment'], training_seed=args.seed,
                      source_sha256=hashes, physical_calibration_ready=False)
    target = args.timesteps or protocol['timesteps_per_seed']
    rollout = args.rollout_steps or protocol['rollout_steps']
    transitions = episodes = updates = 0
    accum = episode_accumulator()
    wall_before = 0.
    def reset_seed():
        return protocol['training_seed_base'] + args.seed * 10000000 + episodes
    if args.resume:
        payload = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        if payload['meta'].get('source_sha256') != hashes or payload['meta'].get('training_seed') != args.seed:
            raise ValueError('Resume source/protocol/seed mismatch; create a new experiment instead')
        state = agent.load(out/'latest.pt')
        env = state['env']
        transitions, episodes, updates = state['transitions'], state['episodes'], state['updates']
        accum, wall_before = state['accum'], state['wall_seconds']
        reset_info = state['reset_info']
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        obs = env._observation()
    else:
        obs, reset_info = reset_with_valid_particles(env, reset_seed())
        atomic_json(out/'manifest.json', dict(protocol=protocol, source_sha256=hashes,
                    seed=args.seed, device=args.device, target=target, rollout_steps=rollout,
                    feasibility=feasibility, allow_unreachable_baseline=args.allow_unreachable_baseline,
                    preflight=args.preflight, pid=os.getpid()))
    started = time.monotonic()
    next_save = ((transitions // protocol['checkpoint_interval'])+1)*protocol['checkpoint_interval']
    milestones = [m for m in protocol['milestones'] if m > transitions]
    def status(phase, **extra):
        elapsed = wall_before + time.monotonic()-started
        atomic_json(out/'status.json', dict(phase=phase, pid=os.getpid(), seed=args.seed,
                    transitions=transitions, target=target, episodes=episodes, updates=updates,
                    wall_seconds=elapsed, fps=transitions/max(elapsed, 1e-9),
                    timestamp=time.time(), preflight=args.preflight, **extra))
    def checkpoint(name):
        state = dict(transitions=transitions, episodes=episodes, updates=updates,
                     accum=accum, env=env, reset_info=reset_info, wall_seconds=wall_before+time.monotonic()-started,
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        tmp = out/(name+'.tmp')
        agent.save(tmp, training_state=state)
        tmp.replace(out/name)
    status('running')
    heartbeat = time.monotonic()
    # Append logs on resume. A restart record explicitly identifies the restored
    # transition; any tail beyond that checkpoint is superseded, not hidden.
    with (out/'updates.jsonl').open('a' if args.resume else 'x', buffering=1) as logs, \
         (out/'episodes.jsonl').open('a' if args.resume else 'x', buffering=1) as eps:
        if args.resume:
            logs.write(json.dumps(dict(event='resume', restored_transition=transitions))+'\n')
        try:
            while transitions < target:
                remaining = min(rollout, target-transitions)
                if milestones:
                    remaining = min(remaining, milestones[0]-transitions)
                for _ in range(remaining):
                    proposal, lp, value, execution, context, state = physical_policy_action(agent, env, obs)
                    returned, reward, terminated, truncated, info = env.step(execution)
                    if not all(np.isfinite(x).all() for x in (proposal, lp, value, returned['nodes'], info['agent_rewards'])):
                        raise FloatingPointError('Nonfinite rollout values')
                    store_physical_transition(agent, obs, proposal, lp, value, context, state,
                                              env, returned, info, terminated, truncated)
                    transitions += 1
                    accum['steps'] += 1; accum['reward'] += reward
                    for key in ('removed_mass', 'contact_s', 'wall_contact_s', 'particle_contact_s'):
                        accum[key] = accum.get(key, 0.) + float(np.asarray(info[key]).sum())
                    obs = returned
                    if terminated or truncated:
                        record = dict(accum, transitions=transitions, episode=episodes, reset_info=reset_info,
                                      success=info['success'], remaining_mass=info['remaining_mass'],
                                      lost_robots=info['lost_robots'], elapsed_s=info['elapsed_s'],
                                      reason=info['termination_reason'])
                        eps.write(json.dumps(record, allow_nan=False)+'\n')
                        episodes += 1; accum = episode_accumulator()
                        obs, reset_info = reset_with_valid_particles(env, reset_seed())
                    if time.monotonic()-heartbeat >= 10:
                        status('running'); heartbeat = time.monotonic()
                metrics = agent.update(n_epochs=protocol['epochs'], batch_size=protocol['batch_size'])
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError('Nonfinite PPO metrics')
                if not all(torch.isfinite(p).all() for net in (agent.actor, agent.critic) for p in net.parameters()):
                    raise FloatingPointError('Nonfinite model parameters')
                updates += 1
                record = dict(transitions=transitions, updates=updates, episodes=episodes, **metrics)
                logs.write(json.dumps(record, allow_nan=False)+'\n')
                print(json.dumps(record), flush=True)
                is_milestone = bool(milestones and transitions == milestones[0])
                if updates == 1 or transitions >= next_save or transitions == target or is_milestone:
                    checkpoint('latest.pt')
                    next_save = ((transitions//protocol['checkpoint_interval'])+1)*protocol['checkpoint_interval']
                if is_milestone or transitions == target:
                    checkpoint(f'policy_{transitions}.pt')
                    if not args.preflight:
                        status('evaluating')
                        atomic_json(out/f'evaluation_{transitions}.json', evaluate(agent, cfg, protocol, protocol['eval_episodes']))
                    if is_milestone:
                        milestones.pop(0)
                status('running', metrics=metrics)
            status('completed')
            atomic_json(out/'summary.json', dict(seed=args.seed, transitions=transitions, episodes=episodes,
                        updates=updates, preflight=args.preflight, source_sha256=hashes))
        except BaseException as exc:
            status('failed', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--allow-unreachable-baseline', action='store_true',
                        help='Explicit engineering diagnostic; does not make all-clot success reachable')
    args = parser.parse_args()
    if args.seed not in (42,43,44) or (args.timesteps is not None and args.timesteps <= 0) or (args.rollout_steps is not None and args.rollout_steps <= 0):
        parser.error('Invalid seed or budget')
    train(args)


if __name__ == '__main__':
    main()
