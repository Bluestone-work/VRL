"""Train/evaluate frozen EXP16--20 v2 jobs; outputs are append-only per attempt."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from environments.research_predictive_env import ResearchBalanced, ResearchSingle, replan
from environments.vessel_geometry import resolve_pool
from marl.predictive_research import (MotionGRU, LocalWorldEnsemble, ActionController,
                                      SPEED, RANGE, HISTORY, HORIZONS, frame, local, bounded)
from marl.geometric_control import direct_local_action, route_guidance
from marl.mappo_advanced import MAPPOAdvanced
from scripts.research_protocol import geometry_hash, content_hash


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_data_cost(path):
    if not path:
        return 0
    metadata = json.loads((Path(path).parent/'metrics.json').read_text())
    return metadata.get('real_data_transitions')


def load_motion(path, device):
    if not path:
        return None
    record = torch.load(path, map_location=device, weights_only=False)
    model = MotionGRU().to(device)
    model.load_state_dict(record['model'])
    return model.eval()


def load_world(path, device):
    record = torch.load(path, map_location=device, weights_only=False)
    if not record['gate']['passed']:
        raise ValueError('World model validation gate failed; no force override')
    model = LocalWorldEnsemble().to(device)
    model.load_state_dict(record['model'])
    return model.eval()


def agent_for(device, checkpoint=None):
    agent = MAPPOAdvanced(n_agents=5, obs_dim=52, action_dim=3, state_dim=24,
                         hidden_dim=128, architecture='adaptive_edge_gat',
                         control_mode='local', critic_value_mode='v', dropout=0,
                         device=device, lr_actor=3e-4, lr_critic=1e-3)
    if checkpoint:
        agent.load(checkpoint, load_optimizers=False)
    return agent


def context(env, obs):
    ctx = dict(positions=env.robot_positions.copy(), velocities=env.robot_velocities.copy(),
               adjacency=obs['adjacency'].copy())
    if 'agent_mask' in obs:
        ctx['agent_mask'] = obs['agent_mask'].copy()
    return ctx


def environment(args, predictor=None, n_envs=None):
    return ResearchBalanced(n_envs=n_envs or args.n_envs, seed=args.seed,
        num_robots=5, num_clots=3, horizon=300, robot_radius=.0011,
        obs_mode='geometric_predictive', dynamic_intravascular_particles=True,
        particle_count=24, particle_radius_ratio=1.6, particle_lateral_drift=.15,
        contact_mode='geodesic', tree_resample_interval=900,
        predictor=predictor, predictor_device=args.device)


def terminal_observation(env, obs, done, info):
    final = {k: v.copy() for k, v in obs.items()}
    ctx = context(env, obs)
    if 'final_observation' in info:
        for key in final:
            final[key][done] = info['final_observation'][key][done]
        for key in ('positions', 'velocities'):
            ctx[key][done] = info['final_context'][key][done]
        ctx['adjacency'] = final['adjacency'].copy()
    return final, ctx


def train_policy(args):
    out = Path(args.out)
    predictor = load_motion(args.predictor, args.device)
    env = environment(args, predictor)
    agent = agent_for(args.device)
    world = load_world(args.world, args.device) if args.arm == 20 else None
    controller = ActionController(world, args.device) if args.arm in (18, 20) else None
    agent.meta.update(research_protocol='exp16_20_v2', experiment_arm=args.arm,
                      predictor_sha256=digest(args.predictor) if args.predictor else None,
                      world_sha256=digest(args.world) if args.world else None,
                      executed_action_semantics='hybrid_predictive_filter' if controller else 'direct_local_frenet')
    obs = env.observe()
    transitions, next_checkpoint = 0, args.checkpoint_interval
    started = time.monotonic()
    accum = {k: np.zeros(args.n_envs) for k in ('return', 'wall', 'particle', 'pair', 'steps', 'interventions')}
    epfile = (out/'episodes.jsonl').open('x')
    updatefile = (out/'updates.jsonl').open('x')
    rollout_steps = 128
    try:
        while transitions < args.timesteps:
            planner_time, switch_count, interventions, trusted = 0., 0, 0, 0
            for _ in range(min(rollout_steps, int(np.ceil((args.timesteps-transitions)/args.n_envs)))):
                obs, plan = replan(env)
                planner_time += plan['planner_seconds']
                switch_count += plan['switches']
                ctx = context(env, obs)
                states = obs['clot_state'].reshape(args.n_envs, -1)
                actions, lp, values = agent.act_batch(obs['nodes'], ctx, states)
                execution = bounded(actions)
                if controller:
                    execution, diagnostics = controller.choose(obs['nodes'], actions)
                    interventions += diagnostics['interventions']
                    trusted += diagnostics['model_trusted']
                changed = (np.linalg.norm(execution-bounded(actions), axis=-1)>1e-6).sum(-1)
                returned, reward, term, trunc, info = env.step(direct_local_action(execution, env))
                done = term | trunc
                final, next_ctx = terminal_observation(env, returned, done, info)
                ar = info['agent_rewards'] + info['team_reward'][:, None]/5
                # PPO likelihood belongs to sampled proposal, never the filtered action.
                # The deterministic filter is part of the effective environment.
                agent.buffer.store(obs['nodes'], actions, ar,
                    np.repeat(done[:, None], 5, 1).astype(np.float32), lp, values, ctx, states,
                    next_obs=final['nodes'], next_ctx=next_ctx,
                    next_state=final['clot_state'].reshape(args.n_envs, -1),
                    terminals=np.repeat(term[:, None], 5, 1).astype(np.float32))
                transitions += args.n_envs
                accum['return'] += reward
                accum['wall'] += info['wall_collisions']
                accum['particle'] += info['particle_collisions'].sum(-1)
                accum['pair'] += info['robot_collisions']
                accum['steps'] += 1
                accum['interventions'] += changed
                for i in np.flatnonzero(done):
                    record = {k: float(v[i]) for k, v in accum.items()}
                    record.update(transitions=transitions, scenario=str(info['scenario'][i]),
                        geometry_id=int(info['geometry_id'][i]), success=bool(info['success'][i]),
                        removal=float(info['removal_rate'][i]), path_length=float(info['path_length'][i]),
                        robot_path_length=info['robot_path_length'][i].tolist(),
                        tree_truncation=bool(info['tree_resampled'][i]))
                    epfile.write(json.dumps(record)+'\n')
                    for v in accum.values():
                        v[i] = 0
                obs = returned
            metrics = agent.update(n_epochs=args.epochs, batch_size=2048)
            record = dict(transitions=transitions, fps=transitions/(time.monotonic()-started),
                          planner_seconds=planner_time, switches=switch_count,
                          interventions=interventions, model_trusted=trusted, **metrics)
            updatefile.write(json.dumps(record)+'\n')
            updatefile.flush()
            epfile.flush()
            print(json.dumps(record), flush=True)
            if transitions >= next_checkpoint or transitions >= args.timesteps:
                name = out/f'policy_{transitions}.pt'
                agent.save(name)
                write_new(out/f'checkpoint_{transitions}.ready.json', dict(
                    checkpoint=str(name), transitions=transitions, arm=args.arm,
                    predictor=args.predictor, world=args.world,
                    sha256=digest(name), seed=args.seed,
                    prior_motion_data_transitions=model_data_cost(args.predictor),
                    prior_world_data_transitions=model_data_cost(args.world)))
                next_checkpoint += args.checkpoint_interval
        write_new(out/'complete.json', dict(transitions=transitions, seconds=time.monotonic()-started))
    finally:
        epfile.close()
        updatefile.close()
        env.close()


def collect(args):
    """Fresh simulator transitions; future states only as supervised labels."""
    out = Path(args.out)
    rng = np.random.default_rng(args.seed+91)
    predictor = load_motion(args.predictor, args.device)
    env = environment(args, predictor, n_envs=14)
    actor = agent_for(args.device, args.checkpoint) if args.checkpoint else None
    if actor:
        actor.actor.eval()
        actor.critic.eval()
    pending, motion_x, motion_y, motion_g = [], [], [], []
    records = {k: [] for k in ('obs', 'action', 'target', 'group')}
    sequences = {k: [] for k in ('seq_obs', 'seq_action', 'seq_target', 'seq_group')}
    window = []
    steps = int(np.ceil(args.model_data_steps/14))
    # Entire geometry generation groups, not adjacent steps, define the holdout.
    for step in range(steps):
        obs, _ = replan(env)
        if args.kind == 'motion':
            for child_id, child in enumerate(env.envs):
                hist = child.features.history
                ready = np.flatnonzero(hist.valid[0] >= HISTORY)
                if len(ready) and step % 3 == 0:
                    r = int(rng.choice(ready))
                    particle = int(hist.ids[0, r])
                    x = hist.values[0, r].copy()
                    p = child.particles.positions[0, particle].copy()
                    x[:, :3] = (x[:, :3]-p)/RANGE
                    pending.append(dict(child=child_id, particle=particle, position=p,
                        x=x, labels=[], age=0, episode=int(child.steps[0]),
                        generation=child.active_geometry_id,
                        group=child_id*1000000+child.active_geometry_id))
        if actor:
            action, _, _ = actor.act_batch(obs['nodes'], context(env, obs),
                                          obs['clot_state'].reshape(14, -1), deterministic=True)
        else:
            action = route_guidance(obs['nodes'][..., :36])
        # Mixed behaviour covers on-policy, noisy and stationary dynamics.
        noisy = bounded(action+rng.normal(0, .25, action.shape)).astype(np.float32)
        if step % 10 == 0:
            noisy[:] = rng.uniform(-1, 1, noisy.shape)
        elif step % 10 == 1:
            noisy[:] = 0
        before = env.robot_positions.copy()
        bases = np.concatenate([frame(c) for c in env.envs])
        masses = np.concatenate([c.clot_masses.sum(-1) for c in env.envs])
        groups = env.geometry_ids.copy()
        returned, _, term, trunc, info = env.step(direct_local_action(noisy, env))
        done = term | trunc
        final, ctx = terminal_observation(env, returned, done, info)
        if args.kind == 'motion':
            keep = []
            for item in pending:
                child = env.envs[item['child']]
                item['age'] += 1
                if (child.active_geometry_id != item['generation'] or
                    child.steps[0] != item['episode']+item['age']):
                    continue
                if item['age'] in HORIZONS:
                    item['labels'].append((child.particles.positions[0, item['particle']]-item['position'])/SPEED)
                if item['age'] == max(HORIZONS):
                    motion_x.append(item['x'])
                    motion_y.append(np.stack(item['labels']))
                    motion_g.append(item['group'])
                else:
                    keep.append(item)
            pending = keep
        else:
            # Drop reset transitions for mass targets; terminal success is retained
            # with full pre-step mass. Geometry truncation is excluded explicitly.
            after_mass = np.concatenate([c.clot_masses.sum(-1) for c in env.envs])
            removed = np.maximum(masses-after_mass, 0)
            removed[term] = masses[term]
            valid = ~trunc
            target = np.concatenate((
                (final['nodes']-obs['nodes'])/.1,
                local(ctx['positions']-before, bases)/SPEED,
                np.broadcast_to((removed/(5*.018))[:, None, None], (14, 5, 1)),
                np.broadcast_to((info['wall_collisions']/5)[:, None, None], (14, 5, 1))), -1)
            for key, value in (('obs', obs['nodes']), ('action', noisy), ('target', target)):
                records[key].append(value[valid].reshape(-1, value.shape[-1]))
            records['group'].append(np.repeat(groups[valid], 5))
            window.append((obs['nodes'].copy(), noisy.copy(), final['nodes'].copy(),
                           done.copy(), groups.copy()))
            window = window[-3:]
            if len(window) == 3:
                valid_seq = ~(window[0][3] | window[1][3] | trunc)
                sequences['seq_obs'].append(window[0][0][valid_seq].reshape(-1, 52))
                sequences['seq_action'].append(np.stack([w[1] for w in window], -2)[valid_seq].reshape(-1, 3, 3))
                sequences['seq_target'].append(final['nodes'][valid_seq].reshape(-1, 52))
                sequences['seq_group'].append(np.repeat(window[0][4][valid_seq], 5))
        if step % 200 == 0:
            print(f'collect {args.kind} transitions={(step+1)*14}', flush=True)
    if args.kind == 'motion':
        data = dict(x=np.asarray(motion_x, np.float32), y=np.asarray(motion_y, np.float32),
                    group=np.asarray(motion_g, np.int64))
    else:
        data = {k: np.concatenate(v) for k, v in records.items()}
        data.update({k: np.concatenate(v) for k, v in sequences.items()})
    np.savez_compressed(out/'data.npz', **data)
    write_new(out/'data_manifest.json', dict(real_transitions=steps*14,
        samples=len(data['group']), geometry_groups=int(len(np.unique(data['group']))),
        seed=args.seed, kind=args.kind, action_source='executed_local_frenet',
        sha256=digest(out/'data.npz')))


def fit(args):
    out = Path(args.out)
    data = np.load(args.data)
    groups = np.unique(data['group'])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    valid_groups = groups[:max(1, len(groups)//5)]
    validation = np.flatnonzero(np.isin(data['group'], valid_groups))
    training = np.flatnonzero(~np.isin(data['group'], valid_groups))
    if len(training) < 32 or len(validation) < 16:
        raise ValueError('Insufficient independent geometry groups')
    model = MotionGRU() if args.kind == 'motion' else LocalWorldEnsemble()
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best, best_state = float('inf'), None
    log = (out/'fit.jsonl').open('x')
    arrays = {k: torch.as_tensor(data[k], device=args.device, dtype=torch.float32)
              for k in data.files if k != 'group' and not k.startswith('seq_')}
    def loss(indices, bootstrap=False):
        if args.kind == 'motion':
            mu, lv = model(arrays['x'][indices])
            error = arrays['y'][indices]-mu
            return (.5*(error.square()*torch.exp(-lv)+lv)).mean()
        losses = []
        for member in model.members:
            idx = rng.choice(indices, len(indices), replace=True) if bootstrap else indices
            target = arrays['target'][idx]
            pred = member(torch.cat((arrays['obs'][idx], arrays['action'][idx]), -1))
            losses.append(nn.functional.smooth_l1_loss(pred[:, :56], target[:, :56])+
                          nn.functional.binary_cross_entropy_with_logits(pred[:, 56], target[:, 56]))
        return torch.stack(losses).mean()
    try:
        for epoch in range(args.fit_epochs):
            model.train()
            order = rng.permutation(training)
            for start in range(0, len(order), 1024):
                optimizer.zero_grad()
                value = loss(order[start:start+1024], bootstrap=True)
                value.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                vals = [float(loss(validation[i:i+2048])) for i in range(0, len(validation), 2048)]
            score = float(np.mean(vals))
            log.write(json.dumps(dict(epoch=epoch, validation_loss=score))+'\n')
            log.flush()
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
            print(f'fit {args.kind} epoch={epoch} validation={score:.6f}', flush=True)
        model.load_state_dict(best_state)
        # Metrics on held-out geometry groups; no test-set model selection.
        with torch.no_grad():
            idx = validation[:20000]
            if args.kind == 'motion':
                mu, lv = model(arrays['x'][idx])
                target = arrays['y'][idx]
                cv = arrays['x'][idx, -1, 3:6][:, None]*torch.tensor(HORIZONS, device=args.device)[None, :, None]
                per_h = (mu-target).square().mean((0, 2)).sqrt()*SPEED
                cv_h = (cv-target).square().mean((0, 2)).sqrt()*SPEED
                coverage = ((mu-target).abs() <= 1.96*torch.exp(.5*lv)).float().mean()
                metrics = dict(rmse_by_horizon=per_h.tolist(), cv_rmse_by_horizon=cv_h.tolist(),
                               marginal_95_coverage=float(coverage))
                gate = dict(passed=bool(per_h[1] < cv_h[1]), criterion='3-step RMSE below CV')
            else:
                pred = model(arrays['obs'][idx], arrays['action'][idx]).mean(0)
                target = arrays['target'][idx]
                obs_error = float((pred[:, :52]-target[:, :52]).square().mean().sqrt())
                persist = float(target[:, :52].square().mean().sqrt())
                pos_error = float((pred[:, 52:55]-target[:, 52:55]).square().mean().sqrt())
                baseline = arrays['action'][idx]*arrays['obs'][idx, 24:25]+arrays['obs'][idx, 21:24]
                cv_error = float((baseline-target[:, 52:55]).square().mean().sqrt())
                metrics = dict(obs_rmse=obs_error*.1, persistence_rmse=persist*.1,
                               position_rmse=pos_error*SPEED, kinematic_rmse=cv_error*SPEED)
                seq_idx = np.flatnonzero(np.isin(data['seq_group'], valid_groups))[:20000]
                initial = torch.as_tensor(data['seq_obs'][seq_idx], device=args.device)
                seq_action = torch.as_tensor(data['seq_action'][seq_idx], device=args.device)
                seq_target = torch.as_tensor(data['seq_target'][seq_idx], device=args.device)
                state = initial.clone()
                for step in range(3):
                    state = (state+.1*model(state, seq_action[:, step]).mean(0)[:, :52]).clamp(-1, 1)
                rollout_error = float((state-seq_target).square().mean().sqrt())
                rollout_persistence = float((initial-seq_target).square().mean().sqrt())
                metrics.update(three_step_obs_rmse=rollout_error,
                               three_step_persistence_rmse=rollout_persistence,
                               validation_sequences=len(seq_idx))
                gate = dict(passed=bool(obs_error < persist and pos_error < cv_error and
                                       rollout_error < rollout_persistence),
                            criterion='1/3-step obs RMSE below persistence AND displacement RMSE below kinematics')
        record = dict(model=model.state_dict(), gate=gate, metrics=metrics,
                      data_sha256=digest(args.data), selected_on='geometry_holdout_validation')
        torch.save(record, out/'model.pt')
        write_new(out/'metrics.json', dict(gate=gate, metrics=metrics,
            train_samples=len(training), validation_samples=len(validation),
            real_data_transitions=json.loads((Path(args.data).parent/'data_manifest.json').read_text())['real_transitions'],
            validation_geometry_groups=valid_groups.tolist(), data_sha256=digest(args.data)))
    finally:
        log.close()


def evaluate(args):
    out = Path(args.out)
    predictor = load_motion(args.predictor, args.device)
    agent = agent_for(args.device, args.checkpoint)
    agent.actor.eval()
    agent.critic.eval()
    controller = ActionController(load_world(args.world, args.device) if args.arm == 20 else None,
                                  args.device) if args.arm in (18, 20) else None
    records = []
    output = (out/'episodes.jsonl').open('x')
    try:
        for si, scenario in enumerate(resolve_pool('anatomical')):
            for ep in range(args.eval_episodes):
                # Fixed new validation only; never consume the historical sealed test.
                episode_seed = 72000000+si*1000+ep
                particle_seed = 82000000+si*1000+ep
                env = ResearchSingle(scenario=scenario, scenario_pool=[scenario],
                    randomize_scenario=False, seed=episode_seed, num_robots=5, num_clots=3,
                    horizon=300, robot_radius=.0011, obs_mode='geometric_predictive',
                    contact_mode='geodesic', dynamic_intravascular_particles=True,
                    particle_count=24, particle_seed=particle_seed,
                    predictor=predictor, predictor_device=args.device)
                obs, _ = env.reset(seed=episode_seed)
                initial_mass = float(env.clot_masses.sum())
                geom_hash = geometry_hash(env.tree)
                initial_arrays = {k: getattr(env, k) for k in ('robot_positions', 'robot_velocities',
                    'robot_stations', 'clot_positions', 'clot_stations', 'clot_masses', 'clot_initial_mass')}
                initial_arrays.update(particles=env.particles.positions, particle_velocities=env.particles.velocities)
                initial_hash = content_hash(initial_arrays, dict(geometry=geom_hash,
                    rng=env._rng.bit_generator.state, particle_rng=env.particles._rng.bit_generator.state))
                totals = dict(wall=0, pair=0, particle=0, interventions=0, model_trusted=0,
                              switches=0, planner_seconds=0., inference_seconds=0.)
                positions, particle_positions, proposals, executed, masses = [], [], [], [], []
                positions.append(env.robot_positions.copy())
                particle_positions.append(env.particles.positions[0].copy())
                while True:
                    obs, plan = replan(env)
                    totals['switches'] += plan['switches']
                    totals['planner_seconds'] += plan['planner_seconds']
                    started = time.perf_counter()
                    action, _, _ = agent.act(obs['nodes'], context(env, obs),
                                            obs['clot_state'].reshape(-1), deterministic=True)
                    execution = bounded(action)
                    if controller:
                        execution, diag = controller.choose(obs['nodes'], action)
                        for k, v in diag.items():
                            totals[k] += v
                    world_action = direct_local_action(execution, env)
                    totals['inference_seconds'] += time.perf_counter()-started
                    obs, _, term, trunc, info = env.step(world_action)
                    totals['wall'] += int(info['wall_collisions'])
                    totals['pair'] += int(info['robot_collisions'])
                    totals['particle'] += int(info['particle_collisions'])
                    positions.append(env.robot_positions.copy())
                    particle_positions.append(env.particles.positions[0].copy())
                    proposals.append(action)
                    executed.append(execution)
                    masses.append(env.clot_masses.copy())
                    if term or trunc:
                        break
                removed = initial_mass-float(env.clot_masses.sum())
                record = dict(seed=args.seed, scenario=scenario, episode_seed=episode_seed,
                    particle_seed=particle_seed, initial_hash=initial_hash, geometry_hash=geom_hash, steps=env.steps,
                    success=bool(info['success']), removal=float(info['removal_rate']),
                    successful_steps=int(env.steps) if info['success'] else None,
                    path_length=float(info['path_length']),
                    successful_path_length=float(info['path_length']) if info['success'] else None,
                    removed_mass=removed, path_per_mass=float(info['path_length'])/removed if removed>1e-8 else None,
                    robot_path_length=info['robot_path_length'].tolist(),
                    wall_rate=totals['wall']/(5*env.steps), pair_rate=totals['pair']/(10*env.steps),
                    particle_events_per_robot_step=totals['particle']/(5*env.steps), **totals)
                records.append(record)
                output.write(json.dumps(record)+'\n')
                output.flush()
                # All episodes retained, including failures; no trajectory selection.
                np.savez_compressed(out/f'trace_{si}_{ep}.npz', positions=positions,
                                    particles=particle_positions, proposal=proposals,
                                    executed=executed, masses=masses)
                env.close()
        per_scenario = {}
        keys = ('success', 'removal', 'steps', 'path_length', 'wall_rate', 'pair_rate',
                'particle_events_per_robot_step', 'interventions', 'switches', 'model_trusted',
                'planner_seconds', 'inference_seconds')
        for scenario in resolve_pool('anatomical'):
            rows = [r for r in records if r['scenario'] == scenario]
            per_scenario[scenario] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        macro = {k: float(np.mean([s[k] for s in per_scenario.values()])) for k in keys}
        summary = dict(arm=args.arm, seed=args.seed, checkpoint_sha256=digest(args.checkpoint),
                       device=args.device, episodes=len(records), per_scenario=per_scenario, macro=macro,
                       split='development_validation_v2', sealed_test_accessed=False,
                       zero_mass_episodes=sum(r['removed_mass']<=1e-8 for r in records))
        summary['success_conditioned'] = {k: float(np.mean([r[k] for r in records if r[k] is not None]))
            if any(r[k] is not None for r in records) else None
            for k in ('successful_steps', 'successful_path_length')}
        summary['path_per_removed_mass'] = dict(
            episode_mean=float(np.mean([r['path_per_mass'] for r in records if r['path_per_mass'] is not None]))
                if any(r['path_per_mass'] is not None for r in records) else None,
            aggregate_ratio=sum(r['path_length'] for r in records)/sum(r['removed_mass'] for r in records)
                if sum(r['removed_mass'] for r in records)>1e-8 else None)
        write_new(out/'summary.json', summary)
        print(json.dumps(summary), flush=True)
    finally:
        output.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job', choices=['train', 'collect', 'fit', 'evaluate'])
    p.add_argument('--kind', choices=['motion', 'world'], default='motion')
    p.add_argument('--arm', type=int, choices=[16, 17, 18, 19, 20], default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', required=True)
    p.add_argument('--predictor', default='')
    p.add_argument('--world', default='')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--data', default='')
    p.add_argument('--n-envs', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=3000000)
    p.add_argument('--checkpoint-interval', type=int, default=1000000)
    p.add_argument('--model-data-steps', type=int, default=50000)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--fit-epochs', type=int, default=30)
    p.add_argument('--eval-episodes', type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'config.json', vars(args))
    if args.job == 'train' and args.arm >= 17 and not args.predictor:
        raise ValueError('Learned prediction arm requires frozen predictor')
    {'train': train_policy, 'collect': collect, 'fit': fit, 'evaluate': evaluate}[args.job](args)


if __name__ == '__main__':
    main()
