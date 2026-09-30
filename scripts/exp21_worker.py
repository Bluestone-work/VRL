"""EXP_0021 worker: train/evaluate the tail-repair experiment.

Two arms, matched in every base component:
  base    — variable-N padding + particles-16 training + GRU prediction
            (arm 17 observation stack), NO stall truncation (stall_limit=inf)
  repair  — identical base + no-progress early truncation (K=50)

Budget/protocol per the preregistration in
research/experiments/EXP_0021_TAIL_REPAIR.md: 3M transitions, seeds 42/43/44,
checkpoints at 1M/2M/3M, dev-validation-v2 evaluation (140 episodes per
checkpoint per seed), deterministic policy, cuda:0, sealed test untouched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from environments.exp21_tail_env import make_eval_env, make_training_env
from environments.research_predictive_env import replan
from marl.geometric_control import direct_local_action
from marl.mappo_advanced import MAPPOAdvanced
from marl.predictive_research import bounded
from scripts.research16_20_worker import digest, load_motion


def agent_for(device):
    # 8 slots / 5 active, padded interaction models.
    return MAPPOAdvanced(n_agents=8, obs_dim=52, action_dim=3, state_dim=24,
                         hidden_dim=128, architecture='adaptive_edge_gat',
                         control_mode='local', critic_value_mode='v', dropout=0,
                         device=device, lr_actor=3e-4, lr_critic=1e-3,
                         max_agents=10)


def context(env, obs):
    ctx = dict(positions=env.robot_positions.copy(),
               velocities=env.robot_velocities.copy(),
               adjacency=obs['adjacency'].copy())
    if 'agent_mask' in obs:
        ctx['agent_mask'] = obs['agent_mask'].copy()
    return ctx


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictor = load_motion(args.predictor, args.device)
    stall_limit = 10**9 if args.arm == 'base' else int(args.stall_limit)
    env = make_training_env(args.n_envs, args.seed, predictor, args.device,
                            stall_limit=stall_limit)
    agent = agent_for(args.device)
    agent.meta.update(research_protocol='exp0021_tail_repair', experiment_arm=args.arm,
                      stall_limit=(None if args.arm == 'base' else int(args.stall_limit)),
                      predictor_sha256=digest(args.predictor) if args.predictor else None,
                      base='variable_n_8slot_5active + particles16 + gru_predictive',
                      executed_action_semantics='direct_local_frenet')
    log = (out / 'updates.jsonl').open('x')
    epfile = (out / 'episodes.jsonl').open('x')
    obs, _ = replan(env)
    transitions, next_ckpt = 0, args.checkpoint_interval
    started = time.monotonic()
    accum = {k: np.zeros(args.n_envs) for k in
             ('return', 'wall', 'particle', 'pair', 'steps', 'stall_cuts')}
    rollout = 128
    try:
        while transitions < args.timesteps:
            for _ in range(min(rollout, int(np.ceil((args.timesteps - transitions) / args.n_envs)))):
                obs, plan = replan(env)
                ctx = context(env, obs)
                states = obs['clot_state'].reshape(args.n_envs, -1)
                actions, lp, values = agent.act_batch(obs['nodes'], ctx, states)
                execution = bounded(actions)
                returned, reward, term, trunc, info = env.step(
                    direct_local_action(execution, env))
                done = term | trunc
                # Final observation handling: the shell exposes final_observation
                # via the same info contract as the v2 envs.
                final = {k: v.copy() for k, v in obs.items()} if 'final_observation' not in info else \
                    {k: v.copy() for k, v in info['final_observation'].items()}
                if 'final_observation' not in info:
                    final = {k: v.copy() for k, v in returned.items()}
                next_ctx = context(env, obs)
                if 'final_context' in info:
                    d = np.flatnonzero(done)
                    if 'agent_mask' in next_ctx:
                        next_ctx['agent_mask'][d] = info['final_observation'].get(
                            'agent_mask', next_ctx['agent_mask'])[d]
                ar = info['agent_rewards'] + info['team_reward'][:, None] / 5
                # PPO likelihood belongs to the sampled proposal action.
                agent.buffer.store(obs['nodes'], actions, ar,
                    np.repeat(done[:, None], 8, 1).astype(np.float32), lp, values,
                    ctx, states,
                    next_obs=final['nodes'], next_ctx=next_ctx,
                    next_state=final['clot_state'].reshape(args.n_envs, -1),
                    terminals=np.repeat(term[:, None], 8, 1).astype(np.float32))
                transitions += args.n_envs
                accum['return'] += reward
                accum['wall'] += info['wall_collisions']
                accum['particle'] += info['particle_collisions'].sum(-1)
                accum['pair'] += info['robot_collisions']
                accum['steps'] += 1
                accum['stall_cuts'] += np.asarray(info.get('stall_truncated', np.zeros(args.n_envs, bool))).astype(float)
                for i in np.flatnonzero(done):
                    record = {k: float(v[i]) for k, v in accum.items()}
                    record.update(transitions=transitions,
                                  scenario=str(info['scenario'][i]),
                                  geometry_id=int(info['geometry_id'][i]),
                                  success=bool(info['success'][i]),
                                  removal=float(info['removal_rate'][i]),
                                  stall_cut=bool(info['stall_truncated'][i]))
                    epfile.write(json.dumps(record) + '\n')
                    for v in accum.values():
                        v[i] = 0
                obs = returned
            metrics = agent.update(n_epochs=args.epochs, batch_size=2048)
            record = dict(transitions=transitions,
                          fps=transitions / (time.monotonic() - started), **metrics)
            log.write(json.dumps(record) + '\n')
            log.flush(); epfile.flush()
            print(json.dumps(record), flush=True)
            if transitions >= next_ckpt or transitions >= args.timesteps:
                name = out / f'policy_{transitions}.pt'
                agent.save(str(name))
                next_ckpt += args.checkpoint_interval
        (out / 'summary.json').write_text(json.dumps(
            dict(seed=args.seed, arm=args.arm, timesteps=transitions,
                 stall_limit=(None if args.arm == 'base' else int(args.stall_limit)))))
    finally:
        log.close(); epfile.close()


def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictor = load_motion(args.predictor, args.device)
    agent = agent_for(args.device)
    agent.load(args.checkpoint, load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    stall_limit = 10**9 if args.arm == 'base' else int(args.stall_limit)
    from environments.vessel_geometry import resolve_pool
    records = []
    for si, scenario in enumerate(resolve_pool('anatomical')):
        for ep in range(args.eval_episodes):
            episode_seed = 72000000 + si * 1000 + ep
            particle_seed = 82000000 + si * 1000 + ep
            env = make_eval_env(scenario, episode_seed, particle_seed,
                                predictor, args.device, stall_limit=stall_limit)
            # 5-real-robot policy acts on 5 rows (checkpoint reloads for any
            # n_agents <= 8 via padding): use the 8-slot act path with a mask.
            obs, _ = env.reset(seed=episode_seed)
            wall = pair = 0
            steps = 0
            while True:
                obs, plan = replan(env)
                n = env.num_robots
                nodes8 = np.zeros((1, 8, 52), np.float32)
                nodes8[0, :n] = obs['nodes']
                ctx = dict(positions=np.zeros((1, 8, 3), np.float32),
                           velocities=np.zeros((1, 8, 3), np.float32),
                           adjacency=np.zeros((1, 8, 8), np.float32))
                ctx['positions'][0, :n] = env.robot_positions
                ctx['velocities'][0, :n] = env.robot_velocities
                ctx['adjacency'][0, :n, :n] = obs['adjacency']
                ctx['agent_mask'] = np.zeros((1, 8), bool)
                ctx['agent_mask'][0, :n] = True
                state = np.zeros((1, 24), np.float32)
                state[0] = obs['clot_state'].reshape(-1)
                action, _, _ = agent.act_batch(nodes8, ctx, state, deterministic=True)
                # Same execution semantics as training and the v2 evaluations:
                # bound the Frenet proposal, then transform to world frame.
                # Feeding the raw proposal to env.step (as the first run did)
                # makes the env read it as a world-frame velocity command —
                # a different controller entirely.
                a5 = bounded(action[0, :n])
                obs, r, term, trunc, info = env.step(direct_local_action(a5, env))
                wall += int(info['wall_collisions'])
                pair += int(info['robot_collisions'])
                steps = int(env.steps)
                if term or trunc:
                    break
            records.append(dict(scenario=scenario, episode_seed=episode_seed,
                                success=bool(info['success']),
                                removal=float(info['removal_rate']),
                                steps=steps,
                                stall_cut=bool(info.get('stall_truncated', False)),
                                stall_steps=int(info.get('stall_steps', 0))))
            env.close()
    summary = dict(arm=args.arm, checkpoint=args.checkpoint, seed=args.seed,
                   episodes=len(records),
                   success=float(np.mean([r['success'] for r in records])),
                   removal=float(np.mean([r['removal'] for r in records])),
                   steps=float(np.mean([r['steps'] for r in records])),
                   stall_cuts=int(sum(r['stall_cut'] for r in records)),
                   wall_rate=float(wall / max(sum(r['steps'] for r in records) * 5, 1)))
    (out / 'episodes.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in records))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job', choices=['train', 'evaluate'])
    p.add_argument('--arm', choices=['base', 'repair'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', required=True)
    p.add_argument('--n-envs', type=int, default=56)
    p.add_argument('--timesteps', type=int, default=3_000_000)
    p.add_argument('--checkpoint-interval', type=int, default=1_000_000)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--stall-limit', type=int, default=50)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--predictor', default='')
    p.add_argument('--checkpoint', default='')
    args = p.parse_args()
    if args.job == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
