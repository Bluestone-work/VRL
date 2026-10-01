"""Compiled multi-environment MAPPO continuation with batched GPU inference.

128 environment transitions per update, matching EXP23. Across eight environments
this is 16 time steps each. The changed rollout layout is explicitly recorded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.mca_physical_policy import make_physical_agent, physical_context, initialize_expanded_obstacle_policy
from marl.mca_exploration import apply_exploration_schedule, exploration_cap
from marl.geometric_control import direct_local_action
from scripts.train_mca_physical import ROOT, DEFAULT_PROTOCOL, atomic_json, reset_with_valid_particles, episode_accumulator
from scripts.train_mca_compiled import evaluate
from scripts.mca_training_gate import require_all_clear_capacity, require_feasibility_certificate


def hashes(protocol_path=DEFAULT_PROTOCOL, physics_path=None):
    paths=[]
    for folder in ('environments','marl'): paths+=list((ROOT/folder).rglob('*.py'))
    paths += [Path(__file__),ROOT/'scripts/train_mca_physical.py',ROOT/'scripts/train_mca_compiled.py',ROOT/'scripts/mca_training_gate.py',protocol_path,
              physics_path or ROOT/'configs/experiments/EXP_0022B_MCA_DYNAMICS.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def context_batch(envs,obs):
    contexts=[physical_context(e,o) for e,o in zip(envs,obs)]
    return {k:np.stack([c[k] for c in contexts]) for k in contexts[0]}


def train(args):
    protocol_path=Path(args.protocol).resolve()
    protocol=json.loads(protocol_path.read_text())
    exploration_cap(protocol.get('exploration_schedule'), 0)
    physics_path=ROOT/protocol['physics_config'];source=hashes(protocol_path,physics_path)
    cfg=DynamicsConfig.from_json(physics_path)
    gate_env=CompiledMCAPhysicalEnv(cfg);reset_with_valid_particles(gate_env,args.seed)
    feasibility=require_all_clear_capacity(gate_env,allow_unreachable_baseline=args.allow_unreachable_baseline)
    if cfg.contact_model in ('stenosis_surface', 'localized_point'):
        feasibility.update(require_feasibility_certificate(protocol, ROOT))
    out=Path(args.out).resolve()
    if args.resume:
        if not out.is_dir(): raise ValueError('Missing resume directory')
    else: out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed)
    envs=[CompiledMCAPhysicalEnv(cfg) for _ in range(args.n_envs)]
    agent=make_physical_agent(envs[0],seed=args.seed,hidden_dim=protocol['hidden_dim'],device=args.device,
                              ppo=protocol.get('ppo'))
    target=args.timesteps or protocol['timesteps_per_seed'];rollout=protocol['rollout_steps']
    if rollout%args.n_envs or target%args.n_envs: raise ValueError('Budget and rollout must be divisible by n_envs')
    transitions=episodes=updates=next_episode=0;wall_before=0.
    accum=[episode_accumulator() for _ in envs];reset_infos=[None]*args.n_envs;episode_ids=[None]*args.n_envs;obs=[None]*args.n_envs
    def reset(i):
        nonlocal next_episode
        episode_ids[i]=next_episode
        seed=protocol['training_seed_base']+args.seed*10000000+next_episode
        next_episode+=1
        obs[i],reset_infos[i]=reset_with_valid_particles(envs[i],seed)
    payload=None
    initialization=None
    if args.initialize_from:
        initial_path=Path(args.initialize_from).resolve()
        parent=torch.load(initial_path,map_location='cpu',weights_only=False)
        expansion=protocol.get('weight_initialization') in ('expand_point36_to_obstacles76', 'expand_obstacles76_to_trajectories172',
            'expand_obstacles76_to_bounded172', 'expand_obstacles76_to_anchored172',
            'expand_obstacles76_to_routed112', 'expand_bounded172_to_routed208')
        for key in (('action_semantics','physical_action_semantics') if expansion else
                    ('observation_schema','action_semantics','physical_action_semantics')):
            if parent['meta'].get(key)!=agent.meta.get(key):
                raise ValueError(f'Warm-start {key} mismatch')
        if parent['meta'].get('training_seed')!=args.seed:
            raise ValueError('Warm-start seed mismatch')
        expanded=initialize_expanded_obstacle_policy(agent,parent) if expansion else []
        if not expansion:agent.load(initial_path,load_optimizers=False)
        initialization=dict(checkpoint=str(initial_path),sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                            experiment=parent['meta'].get('training_experiment'),
                            parent_transitions=parent['training_state']['transitions'],
                            expanded_input_weights=expanded,
                            reused='actor and critic weights only; fresh optimizer, environments, RNG and counters')
        del parent
    if args.resume or args.migrate_from:
        path=out/'latest.pt' if args.resume else Path(args.migrate_from).resolve()
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['meta'].get('training_seed')!=args.seed: raise ValueError('Seed mismatch')
        if args.resume:
            if payload['meta'].get('source_sha256')!=source: raise ValueError('Source mismatch')
            if payload['meta'].get('n_envs')!=args.n_envs: raise ValueError('Environment count mismatch')
        else:
            for name,digest in payload['meta']['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError(f'Original source mismatch: {name}')
        state=agent.load(path)
        transitions,episodes,updates=state['transitions'],state['episodes'],state['updates'];wall_before=state['wall_seconds']
        if transitions%args.n_envs: raise ValueError('Migration checkpoint must align with n_envs')
        if args.resume:
            envs=state['envs'];accum=state['accum'];reset_infos=state['reset_infos'];episode_ids=state['episode_ids'];next_episode=state['next_episode']
            obs=[e._observation() for e in envs]
        else:
            envs[0]=state['env'];envs[0].__class__=CompiledMCAPhysicalEnv
            accum[0]=state['accum'];reset_infos[0]=state['reset_info'];episode_ids[0]=episodes;next_episode=episodes+1;obs[0]=envs[0]._observation()
            for i in range(1,len(envs)): reset(i)
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    else:
        for i in range(len(envs)): reset(i)
    # Warm compilation once outside training and restore policy RNG afterwards.
    import copy
    warm=copy.deepcopy(envs[0]);warm.step(np.zeros((cfg.num_robots,3)))
    agent.meta.update(training_experiment=protocol['experiment'],training_seed=args.seed,source_sha256=source,
                      backend='numba_float64_parallel',n_envs=args.n_envs,workers=args.workers,
                      robot_initialization=cfg.robot_initialization,
                      initialization=initialization or (payload['meta'].get('initialization') if payload else None),
                      entropy_estimator='current_policy_reparameterized_tanh_v1',
                      rollout_environment_transitions=rollout,rollout_time_steps=rollout//args.n_envs)
    if not args.resume:
        atomic_json(out/'manifest.json',dict(protocol=protocol,source_sha256=source,seed=args.seed,device=args.device,
                    target=target,feasibility=feasibility,allow_unreachable_baseline=args.allow_unreachable_baseline,
                    n_envs=args.n_envs,workers=args.workers,rollout_transitions=rollout,
                    rollout_time_steps=rollout//args.n_envs,pid=os.getpid(),preflight=args.preflight,
                    migrated_from=str(path) if payload else None,restored_transitions=transitions,
                    migration_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if payload else None))
        if initialization: atomic_json(out/'initialization.json',initialization)
    start_steps=transitions;start_updates=updates;started=time.monotonic();stop=False;pending_rollout=None
    def stop_handler(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    def status(phase,**kwargs):
        now=time.monotonic();dt=now-started
        atomic_json(out/'status.json',dict(phase=phase,pid=os.getpid(),seed=args.seed,transitions=transitions,target=target,
                    episodes=episodes,updates=updates,wall_seconds=wall_before+dt,fps=(transitions-start_steps)/max(dt,1e-9),
                    timestamp=time.time(),preflight=args.preflight,backend='compiled_parallel',n_envs=args.n_envs,
                    start_transitions=start_steps,**kwargs))
    def checkpoint(name='latest.pt'):
        state=dict(transitions=transitions,episodes=episodes,updates=updates,envs=envs,accum=accum,
                   reset_infos=reset_infos,episode_ids=episode_ids,next_episode=next_episode,
                   wall_seconds=wall_before+time.monotonic()-started,numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        tmp=out/(name+'.tmp');agent.save(tmp,training_state=state);tmp.replace(out/name)
    checkpoint();status('running');heartbeat=time.monotonic()
    next_save=(transitions//protocol['checkpoint_interval']+1)*protocol['checkpoint_interval']
    milestones=[m for m in protocol['milestones'] if m>transitions]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
         (out/'updates.jsonl').open('a' if args.resume else 'x',buffering=1) as logs, \
         (out/'episodes.jsonl').open('a' if args.resume else 'x',buffering=1) as eps:
        if args.resume: logs.write(json.dumps(dict(event='resume',restored_transition=transitions))+'\n')
        try:
            while transitions<target:
                std_cap=apply_exploration_schedule(agent, protocol.get('exploration_schedule'), transitions)
                count=min(rollout,target-transitions,milestones[0]-transitions if milestones else target)
                for _ in range(count//args.n_envs):
                    ctx=context_batch(envs,obs);nodes=np.stack([o['nodes'] for o in obs]);states=np.stack([o['clot_state'].reshape(-1) for o in obs])
                    actions,lp,values=agent.act_batch(nodes,ctx,states)
                    execution=[direct_local_action(a,e) for a,e in zip(actions,envs)]
                    futures=[pool.submit(e.step,a) for e,a in zip(envs,execution)]
                    results=[f.result() for f in futures]
                    final=[r[0] for r in results];next_ctx=context_batch(envs,final)
                    terminal=np.array([r[2] for r in results]);trunc=np.array([r[3] for r in results]);done=terminal|trunc
                    lost=~next_ctx['agent_mask']
                    rewards=np.stack([r[-1]['agent_rewards']+r[-1]['team_reward']*ctx['agent_mask'][i]/max(ctx['agent_mask'][i].sum(),1) for i,r in enumerate(results)])
                    if not all(np.isfinite(x).all() for x in (nodes,actions,lp,values,rewards,np.stack([o['nodes'] for o in final]))): raise FloatingPointError('Nonfinite rollout')
                    agent.buffer.store(nodes,actions,rewards,(lost|done[:,None]).astype(np.float32),lp,values,ctx,states,
                            next_obs=np.stack([o['nodes'] for o in final]),next_ctx=next_ctx,
                            next_state=np.stack([o['clot_state'].reshape(-1) for o in final]),terminals=(lost|terminal[:,None]).astype(np.float32))
                    transitions+=args.n_envs;obs=final
                    for i,(_,r,term,tr,info) in enumerate(results):
                        acc=accum[i];acc['steps']+=1;acc['reward']+=r
                        for key in ('removed_mass','contact_s','wall_contact_s','particle_contact_s'): acc[key]=acc.get(key,0.)+float(np.asarray(info[key]).sum())
                        if term or tr:
                            eps.write(json.dumps(dict(acc,transitions=transitions,episode=episode_ids[i],env_index=i,
                                      reset_info=reset_infos[i],success=info['success'],remaining_mass=info['remaining_mass'],
                                      collision_free_success=info['collision_free_success'],
                                      particle_collision_events=info['episode_particle_collision_events'],
                                      lost_robots=info['lost_robots'],elapsed_s=info['elapsed_s'],reason=info['termination_reason']),allow_nan=False)+'\n')
                            episodes+=1;accum[i]=episode_accumulator();reset(i)
                    if time.monotonic()-heartbeat>=10: status('running');heartbeat=time.monotonic()
                pending_rollout=agent.buffer._d
                metrics=agent.update(n_epochs=protocol['epochs'],batch_size=protocol['batch_size']);updates+=1
                if std_cap is not None:
                    metrics.update(exploration_std_cap=std_cap,
                                   policy_raw_std_max=float(agent.actor.policy_head.log_std.detach().clamp(-5.,2.).exp().max()))
                if not all(np.isfinite(v) for v in metrics.values()): raise FloatingPointError('Nonfinite PPO metrics')
                if not all(torch.isfinite(p).all() for net in (agent.actor,agent.critic) for p in net.parameters()): raise FloatingPointError('Nonfinite weights')
                record=dict(transitions=transitions,updates=updates,episodes=episodes,**metrics);logs.write(json.dumps(record,allow_nan=False)+'\n');print(json.dumps(record),flush=True)
                milestone=bool(milestones and transitions==milestones[0])
                if updates==start_updates+1 or transitions>=next_save or milestone or transitions==target:
                    checkpoint();next_save=(transitions//protocol['checkpoint_interval']+1)*protocol['checkpoint_interval']
                if milestone or transitions==target:
                    checkpoint(f'policy_{transitions}.pt')
                    if not args.preflight:
                        status('evaluating');atomic_json(out/f'evaluation_{transitions}.json',evaluate(
                            agent,cfg,protocol,protocol['eval_episodes'],heartbeat=lambda:status('evaluating')))
                    if milestone: milestones.pop(0)
                status('running',metrics=metrics)
                if stop or (out/'STOP').exists(): checkpoint();status('paused');return
            status('completed');atomic_json(out/'summary.json',dict(transitions=transitions,episodes=episodes,updates=updates,seed=args.seed,source_sha256=source))
        except BaseException as exc:
            # Keep the last good checkpoint intact; preserve the failing batch
            # separately for diagnosis. It must never be used as latest.pt.
            torch.save(dict(buffer=agent.buffer._d if len(agent.buffer) else pending_rollout, actor=agent.actor.state_dict(),
                            critic=agent.critic.state_dict(), transitions=transitions,
                            error=repr(exc)), out/'failure_diagnostic.pt')
            status('failed',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--protocol',default=str(DEFAULT_PROTOCOL))
    p.add_argument('--allow-unreachable-baseline',action='store_true',
                   help='Explicitly run an engineering baseline that cannot achieve all-clot success')
    p.add_argument('--device',default='cuda:0');p.add_argument('--n-envs',type=int,default=8);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--timesteps',type=int);p.add_argument('--preflight',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--migrate-from')
    p.add_argument('--initialize-from',help='Compatible actor/critic weights only; a separate new experiment')
    a=p.parse_args()
    if a.seed not in (42,43,44) or a.n_envs<1 or a.workers<1 or sum(bool(v) for v in (a.resume,a.migrate_from,a.initialize_from))>1: p.error('Invalid arguments')
    train(a)


if __name__=='__main__': main()
