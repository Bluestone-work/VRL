"""Full-window, fixed-flow transport audit, including bodies after robot loss.

The Gym episode can end when all robots exit. This separate engineering check
continues passive tracer transport to the requested time. Clot masses and flow
remain fixed, so it is NOT a lysis, task-success or trained-policy evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from environments.mca_physical_dynamics import PhysicalTubeTransport
from environments.mca_physical_env import CONFIG_PATH, DynamicsConfig, MCAPhysicalEnv
from scripts.validate_mca_dynamics import ROOT, SEEDS, hashfile, jsonable, state_digest


FRACTIONS = (.1, .05, .025, .0125)
MAX_ERROR_MM = .05


def run(config, seed, fraction, duration, out):
    env = MCAPhysicalEnv(config)
    env.reset(seed=seed)
    transport = PhysicalTubeTransport(env.flow_model, spatial_fraction=fraction,
                                     max_substeps=config.max_substeps_per_control,
                                     lubrication_floor=config.lubrication_floor)
    initial_hash = state_digest(env)
    positions, edge, active = env.positions_mm.copy(), env.edges.copy(), env.active.copy()
    rng = np.random.default_rng(seed + 10000)
    elapsed, total_substeps = 0., 0
    path = np.zeros(len(active))
    exits = np.full(len(active), -1, np.int32)
    exit_time = np.full(len(active), np.nan)
    trace = dict(time_s=[0.], positions_mm=[positions.copy()], edge=[edge.copy()],
                 active=[active.copy()], path_mm=[path.copy()], exit_node=[exits.copy()],
                 exit_time_s=[exit_time.copy()])
    started = time.monotonic()
    while elapsed < duration - 1e-12:
        dt = min(config.control_dt_s, duration-elapsed)
        commands = np.zeros_like(positions)
        commands[:config.num_robots] = rng.uniform(-.25, .25, (config.num_robots, 3)) * config.robot_speed_mm_s
        result = transport.advance(positions, edge, env.body_radius, commands, active,
                                   env.solution, dt)
        new_exits = result.exit_node >= 0
        exits[new_exits] = result.exit_node[new_exits]
        exit_time[new_exits] = elapsed + result.exit_time_s[new_exits]
        positions, edge, active = result.positions_mm, result.edge, result.active
        path += result.path_mm
        elapsed += dt
        total_substeps += result.substeps
        if not np.isfinite(positions).all() or not np.isfinite(path).all():
            raise AssertionError('Nonfinite transport state')
        for key, value in dict(time_s=elapsed, positions_mm=positions, edge=edge, active=active,
                               path_mm=path, exit_node=exits, exit_time_s=exit_time).items():
            trace[key].append(np.array(value, copy=True))
    trace = {key: np.asarray(value) for key, value in trace.items()}
    name = f'seed_{seed}_fraction_{fraction:g}.npz'
    np.savez_compressed(out/name, **trace)
    record = dict(seed=seed, fraction=fraction, trace=name, initial_state_sha256=initial_hash,
                  elapsed_s=elapsed, total_substeps=total_substeps,
                  active_robots=int(active[:config.num_robots].sum()),
                  active_particles=int(active[config.num_robots:].sum()),
                  wall_time_s=time.monotonic()-started)
    print(json.dumps(record), flush=True)
    return record, trace


def compare(trace, reference):
    grid = np.array_equal(trace['time_s'], reference['time_s'])
    if not grid:
        raise AssertionError('Full-window transport grids differ')
    error = np.linalg.norm(trace['positions_mm']-reference['positions_mm'], axis=-1)
    masks = np.array_equal(trace['active'], reference['active'])
    exits = np.array_equal(trace['exit_node'], reference['exit_node'])
    both = np.isfinite(trace['exit_time_s'][-1]) & np.isfinite(reference['exit_time_s'][-1])
    return dict(max_position_error_mm=float(error.max()),
                max_path_error_mm=float(np.abs(trace['path_mm']-reference['path_mm']).max()),
                max_exit_time_error_s=float(np.abs(trace['exit_time_s'][-1, both] -
                                                    reference['exit_time_s'][-1, both]).max()) if both.any() else None,
                active_masks_match=masks, exit_nodes_match=exits,
                passed=bool(error.max() <= MAX_ERROR_MM and masks and exits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error('duration must be finite and positive')
    args.out.mkdir(parents=True, exist_ok=False)
    config = DynamicsConfig.from_json()
    (args.out/'config.json').write_bytes(CONFIG_PATH.read_bytes())
    source_files = ['scripts/audit_mca_transport.py', 'scripts/validate_mca_dynamics.py',
                    'environments/mca_physical_dynamics.py', 'environments/mca_physical_env.py',
                    'environments/mca_physiology.py', 'environments/vessel_anatomy.py',
                    'environments/vessel_tree_generator.py', 'environments/vessel_geometry.py']
    protocol = dict(seeds=SEEDS, spatial_fractions=FRACTIONS, duration_s=args.duration_s,
                    threshold_mm=MAX_ERROR_MM, reference_fraction=FRACTIONS[-1],
                    evaluation='all control-time samples, active masks and exit-node identity',
                    scope='fixed-pressure/fixed-clot-mass transport only; no lysis or policy performance',
                    after_all_robots_exit='continue passive tracer transport for entire physical window',
                    config_sha256=hashfile(CONFIG_PATH),
                    source_sha256={f: hashfile(ROOT/f) for f in source_files})
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records, comparisons = [], []
    for seed in SEEDS:
        batch = [run(config, seed, f, args.duration_s, args.out) for f in FRACTIONS]
        records.extend(record for record, _ in batch)
        if len({record['initial_state_sha256'] for record, _ in batch}) != 1:
            raise AssertionError('Initial-state mismatch across refinements')
        for (record, trace), fraction in zip(batch[:-1], FRACTIONS[:-1]):
            comparisons.append(dict(seed=seed, fraction=fraction,
                                    reference_fraction=FRACTIONS[-1], **compare(trace, batch[-1][1])))
    by_resolution = {str(f): all(r['passed'] for r in comparisons if r['fraction'] == f)
                     for f in FRACTIONS[:-1]}
    passed = by_resolution[str(FRACTIONS[-2])]
    if any(hashfile(ROOT/f) != digest for f, digest in protocol['source_sha256'].items()):
        raise AssertionError('Source changed during numerical validation')
    summary = dict(protocol=protocol, trajectories=records, comparisons=comparisons,
                   all_seeds_pass_by_fraction=by_resolution,
                   fine_resolution_passed=passed, formal_training_ready=False,
                   policy_success_rate=None, created_at_utc=datetime.now(timezone.utc).isoformat())
    (args.out/'summary.json').write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False)+'\n')
    lines = ['# EXP22B — 完整时间窗固定流场输运验证', '',
             f'全部轨迹实际积分至 {args.duration_s:g} 秒；机器人离场后继续追踪被动粒子。',
             '固定血栓质量及流场，不是溶解/任务成功率/训练评估。阈值保持 0.05 mm。', '',
             '|seed|空间步长比例|参考比例|全轨迹最大误差 mm|活跃掩码一致|出口一致|通过|',
             '|---:|---:|---:|---:|---|---|---|']
    for row in comparisons:
        lines.append(f"|{row['seed']}|{row['fraction']}|{row['reference_fraction']}|"
                     f"{row['max_position_error_mm']:.6f}|{row['active_masks_match']}|"
                     f"{row['exit_nodes_match']}|{row['passed']}|")
    lines.extend(['', f'各分辨率三seed gate：`{json.dumps(by_resolution)}`。',
                  '数值通过只覆盖该固定流场、动作和时间窗；不代表生理/硬件标定或正式训练就绪。', ''])
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(
        status='complete' if passed else 'numerical_gate_failed',
        artifact_sha256={p.name: hashfile(p) for p in sorted(args.out.iterdir()) if p.is_file()}), indent=2)+'\n')
    print(json.dumps(dict(all_seeds_pass_by_fraction=by_resolution, fine_resolution_passed=passed)), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
