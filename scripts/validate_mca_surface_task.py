"""Constructive feasibility witness, NEVER training/evaluation policy data.

Uses bounded diagnostic actions via the exact step() API, unchanged distributed
starts, open exits, finite bodies and real time-dependent mass/flow feedback.
The navigation/flow compensation here is NOT imported by RL training.
"""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from scripts.train_mca_physical import atomic_json, reset_with_valid_particles

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/experiments/EXP_0026_MCA_IN_VITRO_DYNAMICS.json'


def witness_action(env, assignments, lookahead_mm=None):
    """Known-map path following and flow compensation, diagnostic only."""
    n=env.num_robots;t=env.transport
    actions=np.zeros((n,3));distance=env._target_distances()
    active_clots=np.flatnonzero(env.masses>0)
    if not len(active_clots): return actions
    for i in range(n):
        if not env.active[i]: continue
        j=assignments[i]
        if env.masses[j]<=0:
            j=int(np.argmin(distance[i]));assignments[i]=j
        p=env.positions_mm[i];e=env.edges[i]
        axis,_,_,fraction=t.coordinates(p[None],np.array([e]),env.solution)
        u,v=t.ends[e];f=fraction[0];route=env.routes[j]
        station=int(env.clot_stations[j])
        next_hop=env.tree.route_to(station)[1]
        endpoint=int(u if route[u]+f*t.length[e] <= route[v]+(1-f)*t.length[e] else v)
        remaining=.06 if lookahead_mm is None else float(lookahead_mm[i])
        previous=axis[0];node=endpoint;goal=t.points[node]
        for _ in range(env.tree.n_stations+1):
            segment=t.points[node]-previous;length=np.linalg.norm(segment)
            if length>=remaining:
                goal=previous+segment*remaining/max(length,1e-12);break
            remaining-=length;previous=t.points[node];goal=previous
            if node==station: break
            node=int(next_hop[node])
        if distance[i,j]<.7 and env.config.contact_model != 'stenosis_surface':
            goal=env.clot_positions_mm[j]
        elif distance[i,j]<.7:
            # Approach the finite-body contact surface at the clot section.
            ce=int(t.nearest_edges(env.clot_positions_mm[j:j+1])[0])
            tangent=t.direction[ce]
            normal=np.cross(tangent,np.eye(3)[np.argmin(abs(tangent))]);normal/=np.linalg.norm(normal)
            binormal=np.cross(tangent,normal)
            angle=2*np.pi*i/n
            radial=np.cos(angle)*normal+np.sin(angle)*binormal
            radius=env.solution['radius_mm'][station]
            goal=env.clot_positions_mm[j]+radial*(radius-env.config.robot_radius_mm-.5*env.config.contact_distance_mm)
        delta=goal-p
        desired=delta*min(12.,.75/max(np.linalg.norm(delta),1e-12))
        flow=t.velocity_mm_s(p[None],np.array([e]),env.solution)[0]
        command=(desired-flow)/env.config.robot_speed_mm_s
        actions[i]=command/max(1.,np.linalg.norm(command))
    return actions


def run_witness(config,seed,*,backend='compiled',save_trace=False):
    env=(CompiledMCAPhysicalEnv if backend=='compiled' else MCAPhysicalEnv)(config)
    _,reset=reset_with_valid_particles(env,seed)
    initial=env.positions_mm[:env.num_robots].copy()
    distance=env._target_distances()
    assignments=np.argmin(distance,axis=1)
    rows,cols=linear_sum_assignment(distance)
    assignments[rows]=cols
    contact=np.zeros(env.num_robots);path=[];times=[];masses=[];active=[];max_action=0.
    particle_contact=0.;initial_particles=env.positions_mm[env.num_robots:].copy()
    last_progress_pos=initial.copy();last_progress_check=0.
    short_until=np.zeros(env.num_robots);recoveries=0
    started=time.monotonic()
    while not env._done:
        if save_trace:
            path.append(env.positions_mm.copy());times.append(env.elapsed_s)
            masses.append(env.masses.copy());active.append(env.active.copy())
        if env.elapsed_s-last_progress_check>=2.-1e-9:
            stalled=np.linalg.norm(env.positions_mm[:env.num_robots]-last_progress_pos,axis=1)<.01
            stalled &= env._target_distances().min(axis=1)>.7
            stalled &= env.active[:env.num_robots] & (short_until<=env.elapsed_s)
            # Recover from short-connector projection stalls with a brief,
            # smaller lookahead; never change physics, positions or targets.
            short_until[stalled]=env.elapsed_s+2.
            recoveries+=int(stalled.sum())
            last_progress_pos=env.positions_mm[:env.num_robots].copy();last_progress_check=env.elapsed_s
        action=witness_action(env,assignments,np.where(short_until>env.elapsed_s,.02,.06))
        max_action=max(max_action,float(np.linalg.norm(action,axis=1).max()))
        _,_,_,_,info=env.step(action);contact+=info['contact_s']
        particle_contact+=float(info['particle_contact_s'].sum())
    if save_trace:
        path.append(env.positions_mm.copy());times.append(env.elapsed_s)
        masses.append(env.masses.copy());active.append(env.active.copy())
    record=dict(seed=seed,reset=reset,backend=backend,success=info['success'],elapsed_s=env.elapsed_s,
                remaining_masses=env.masses.tolist(),lost_robots=info['lost_robots'],
                initial_positions_mm=initial.tolist(),final_positions_mm=env.positions_mm[:env.num_robots].tolist(),
                path_mm=env.path_mm[:env.num_robots].tolist(),contact_robot_seconds=contact.tolist(),
                max_action_norm=max_action,termination_reason=info['termination_reason'],wall_seconds=time.monotonic()-started,
                initial_particle_count=config.particle_count,active_particles=info['active_particles'],
                diagnostic_stagnation_recoveries=recoveries,
                particle_contact_seconds=particle_contact,
                maximum_particle_displacement_mm=float(np.linalg.norm(env.positions_mm[env.num_robots:]-initial_particles,axis=1).max())
                if len(initial_particles) else 0.)
    trace=dict(time_s=np.array(times),positions_mm=np.array(path),masses=np.array(masses),active=np.array(active))
    return record,trace


def source_hashes(config_path=CONFIG):
    paths=[Path(config_path),Path(__file__)]
    paths+=list((ROOT/'environments').rglob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--config',default=str(CONFIG))
    p.add_argument('--start-seed',type=int,default=820000000);p.add_argument('--reference',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    config_path=Path(args.config).resolve()
    config=DynamicsConfig.from_json(config_path)
    # Feasibility is a property of the task physics. An action prior is part of
    # the policy's action mapping, so the witness commands the actuator directly.
    if config.action_prior!='none':
        from dataclasses import replace
        config=replace(config,action_prior='none')
    source=source_hashes(config_path)
    env=CompiledMCAPhysicalEnv(config);reset_with_valid_particles(env,args.start_seed)
    samples=[]
    for fractions in itertools.product((0.,.5,1.),repeat=env.num_clots):
        solution=env.flow_model.solve(env._radii(np.asarray(fractions)))
        peak=float(np.max(2*solution['station_inflow_mm3_s']/(np.pi*solution['radius_mm']**2)))
        samples.append(peak*config.inlet_flow_multiplier_max/env.episode_flow_multiplier)
    records=[]
    for index,seed in enumerate(range(args.start_seed,args.start_seed+args.seeds)):
        record,trace=run_witness(config,seed,backend='reference' if args.reference else 'compiled',save_trace=index==0)
        records.append(record);print(json.dumps({k:record[k] for k in ('seed','success','elapsed_s','lost_robots','remaining_masses','wall_seconds')}),flush=True)
        atomic_json(out/'progress.json',dict(completed=len(records),episodes=records))
        if index==0:np.savez_compressed(out/'witness_trace.npz',**trace)
    boundaries=[]
    if config.inlet_flow_multiplier_min != config.inlet_flow_multiplier_max:
        for multiplier in (config.inlet_flow_multiplier_min,config.inlet_flow_multiplier_max):
            fixed=replace(config,inlet_flow_multiplier_min=multiplier,inlet_flow_multiplier_max=multiplier)
            for seed in range(args.start_seed+10000,args.start_seed+10000+(1 if args.reference else 5)):
                record,_=run_witness(fixed,seed,backend='reference' if args.reference else 'compiled')
                record['boundary_multiplier']=multiplier;boundaries.append(record)
                print(json.dumps(dict(boundary=multiplier,seed=seed,success=record['success'],elapsed_s=record['elapsed_s'])),flush=True)
    passed=all(r['success'] and r['max_action_norm']<=1+1e-12 and r['lost_robots']==0 for r in records+boundaries)
    if source!=source_hashes(config_path):raise RuntimeError('Validation source changed while running; rerun before certification')
    atomic_json(out/'certificate.json',dict(passed=passed,scope='constructive in-vitro simulation feasibility, NOT RL or hardware performance',
        policy_success_rate=None,controller='bounded diagnostic known-map path follower; excluded from training',
        config=str(config_path.relative_to(ROOT)),source_sha256=source,episodes=records,boundary_episodes=boundaries,
        sampled_flow_states=len(samples),maximum_sampled_centerline_flow_mm_s=max(samples),
        robot_speed_mm_s=config.robot_speed_mm_s,all_four_masses_zero_required=True,
        calibration_ready=False))
    if not passed:raise SystemExit('Feasibility witness failed; training is blocked')


if __name__=='__main__':main()
