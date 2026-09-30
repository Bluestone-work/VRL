"""Read-only task feasibility diagnostics; oracle cases are NOT RL results.

Keep physical flow, propulsion, clot mass and success definition fixed. Compare
the one-second task to a 15-second diagnostic window and deliberately favourable
starts on the clot centres. Training and evaluation seed namespaces are distinct.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from scripts.train_mca_physical import atomic_json, reset_with_valid_particles

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/experiments/EXP_0024_MCA_DISTRIBUTED_DYNAMICS.json'


def local_contact_margins(env):
    result = []
    for index, station in enumerate(env.clot_stations):
        point = env.clot_positions_mm[index]
        edge = int(env.transport.nearest_edges(point[None])[0])
        _, radius, _, _ = env.transport.coordinates(point[None], np.array([edge]), env.solution)
        r = float(radius[0]); a = env.config.robot_radius_mm
        flux = env.solution['station_inflow_mm3_s'][env.transport.ends[edge, 1]]
        mean = float(flux/(np.pi*r*r))
        # Best possible cross-sectional position for opposing the axial flow.
        near_wall = 2*mean*(1-(1-a/r)**2) if r >= a else None
        result.append(dict(clot=index, branch=int(env.tree.branch_ids[station]),
                           radius_mm=r, mean_flow_mm_s=mean,
                           minimum_accessible_flow_mm_s=near_wall,
                           optimistic_upstream_margin_mm_s=env.config.robot_speed_mm_s-near_wall if near_wall is not None else None))
    return result


def run_case(config, seed, start_mode, controller):
    env = CompiledMCAPhysicalEnv(config)
    reset_with_valid_particles(env, seed)
    if start_mode == 'at_clots_oracle':
        # Give four robots perfect initial centre contact; fifth starts safely
        # off the first centre. This is a labelled diagnostic, never training.
        positions = env.clot_positions_mm.copy()
        edge = env.transport.nearest_edges(positions[:1])[0]
        extra = positions[0] + env.transport.direction[edge]*(3*config.robot_radius_mm)
        positions = np.concatenate((positions, extra[None]))
        env.reset(seed=seed, options={'robot_positions_mm':positions,
                                     'particle_positions_mm':env.positions_mm[config.num_robots:].copy()})
    initial = env.positions_mm[:config.num_robots].copy()
    rng = np.random.default_rng(seed+123456)
    contact = np.zeros(config.num_robots)
    removed = 0.; actual_last_exit = None
    while True:
        if controller == 'zero':
            action = np.zeros((config.num_robots, 3))
        elif controller == 'maximum_brake_oracle':
            flow = env.transport.velocity_mm_s(env.positions_mm[:config.num_robots],
                                              env.edges[:config.num_robots], env.solution)
            action = -flow/np.maximum(np.linalg.norm(flow, axis=1, keepdims=True), 1e-12)
        else:
            action = rng.uniform(-1, 1, (config.num_robots, 3))
        _, _, term, trunc, info = env.step(action)
        contact += info['contact_s'];removed += float(np.asarray(info['removed_mass']).sum())
        if term or trunc:
            exits = env.exit_time_s[:config.num_robots]
            if np.isfinite(exits).all():actual_last_exit=float(exits.max())
            break
    return dict(seed=seed, start_mode=start_mode, controller=controller,
                horizon_s=config.episode_duration_s, success=info['success'],
                removal_fraction=removed/env.initial_mass.sum(), contact_robot_seconds=float(contact.sum()),
                elapsed_s=info['elapsed_s'], actual_last_exit_s=actual_last_exit,
                lost_robots=info['lost_robots'], reason=info['termination_reason'],
                initial_positions_mm=initial.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=10)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    config=DynamicsConfig.from_json(CONFIG)
    protocol=dict(scope='engineering feasibility diagnostic, not policy validation',
                  seeds=list(range(810000000,810000000+args.seeds)),
                  horizons_s=[1.,15.], starts=['distributed_branches','at_clots_oracle'],
                  controllers=['zero','random','maximum_brake_oracle'],
                  config_path=str(CONFIG), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  physical_parameters_changed=False, diagnostic_time_window_changed=True,
                  oracle_results_excluded_from_policy_success=True)
    atomic_json(out/'protocol.json',protocol)
    env=CompiledMCAPhysicalEnv(config);reset_with_valid_particles(env,810000000)
    positions=env.clot_positions_mm
    distance=np.linalg.norm(positions[:,None]-positions[None],axis=-1)
    np.fill_diagonal(distance,np.inf)
    distinct=bool(distance.min()>2*config.contact_distance_mm)
    # Under disjoint contact balls a robot cannot dissolve two targets at once.
    upper=config.num_robots*config.lysis_mass_per_s*config.episode_duration_s
    gate=dict(disjoint_contact_regions=distinct, minimum_clot_separation_mm=float(distance.min()),
              initial_mass=float(env.initial_mass.sum()), one_second_optimistic_removed_mass=upper,
              one_second_all_clear_possible=not(distinct and upper<env.initial_mass.sum()),
              optimistic_min_time_s=float(env.initial_mass.sum()/(config.num_robots*config.lysis_mass_per_s)),
              local_flow_margins=local_contact_margins(env))
    atomic_json(out/'analytic_gate.json',gate)
    records=[];started=time.monotonic()
    for horizon in protocol['horizons_s']:
        cfg=replace(config,episode_duration_s=horizon)
        for start in protocol['starts']:
            for controller in protocol['controllers']:
                batch=[run_case(cfg,seed,start,controller) for seed in protocol['seeds']]
                records.extend(batch)
                summary=dict(horizon_s=horizon,start=start,controller=controller,episodes=len(batch),
                             success_rate=float(np.mean([r['success'] for r in batch])),
                             removal_fraction=float(np.mean([r['removal_fraction'] for r in batch])),
                             elapsed_s=float(np.mean([r['elapsed_s'] for r in batch])))
                print(json.dumps(summary),flush=True)
                atomic_json(out/'progress.json',dict(cases=len(records),last_batch=summary))
    atomic_json(out/'summary.json',dict(analytic_gate=gate,episodes=records,wall_seconds=time.monotonic()-started,
                                      policy_success_rate=None,physical_task_unchanged=True))


if __name__=='__main__':main()
