"""EXP22B: three-seed trajectory/convergence audit + one tiny PPO API smoke.

This is software/numerical validation, not a trained-policy success estimate.
An existing output directory is never overwritten. All trajectories are kept.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from environments.mca_physical_env import CONFIG_PATH, DynamicsConfig, MCAPhysicalEnv
from marl.mca_physical_policy import (
    make_physical_agent, physical_policy_action, store_physical_transition,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
FRACTIONS = (.1, .05, .025)
MAX_POSITION_ERROR_MM = .05  # engineering acceptance tolerance, chosen before runs


def hashfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(env):
    digest = hashlib.sha256()
    for array in (env.tree.points, env.tree.radii, env.positions_mm, env.masses,
                  env.clot_stations, env.solution['radius_mm']):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None  # specifically supports not-yet-exited event timestamps
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def run_trajectory(config, seed, fraction, out, *, duration_s=.1):
    effective = replace(config, spatial_fraction=fraction, episode_duration_s=duration_s)
    env = MCAPhysicalEnv(effective)
    obs, _ = env.reset(seed=seed)
    initial_hash = state_digest(env)
    action_rng = np.random.default_rng(seed + 10000)
    trace = dict(positions_mm=[env.positions_mm.copy()], active=[env.active.copy()],
                 time_s=[0.], masses=[env.masses.copy()], path_mm=[env.path_mm.copy()],
                 edge=[env.edges.copy()], exit_node=[env.exit_node.copy()],
                 exit_time_s=[env.exit_time_s.copy()],
                 inlet_flow_ml_min=[env.solution['inlet_flow_ml_min']])
    intervals = []
    started = time.monotonic()
    while True:
        action = action_rng.uniform(-.25, .25, (env.num_robots, 3))
        obs, reward, term, trunc, info = env.step(action)
        if not env.observation_space.contains(obs) or not all(
                np.isfinite(x).all() for x in (obs['nodes'], env.positions_mm, env.masses, env.path_mm)):
            raise AssertionError('Nonfinite or invalid physical rollout')
        intervals.append(info)
        for key, value in dict(positions_mm=env.positions_mm, active=env.active,
                              time_s=env.elapsed_s, masses=env.masses, path_mm=env.path_mm,
                              edge=env.edges, exit_node=env.exit_node, exit_time_s=env.exit_time_s,
                              inlet_flow_ml_min=env.solution['inlet_flow_ml_min']).items():
            trace[key].append(np.array(value, copy=True))
        if term or trunc:
            break
    name = f'seed_{seed}_fraction_{fraction:g}'
    np.savez_compressed(out / f'{name}.npz', **{k: np.asarray(v) for k, v in trace.items()})
    record = dict(seed=seed, spatial_fraction=fraction, initial_state_sha256=initial_hash,
                  effective_config=asdict(effective), intervals=intervals,
                  total_substeps=sum(i['substeps'] for i in intervals),
                  wall_time_s=time.monotonic()-started, remaining_mass=float(env.masses.sum()),
                  lost_robots=int((~env.agent_mask).sum()), success=info['success'],
                  termination_reason=info['termination_reason'],
                  final_positions_mm=env.positions_mm.copy(), final_active=env.active.copy())
    return record


def ppo_smoke(config, out):
    # Six transitions / one optimizer update on fresh weights, no checkpoint
    # selection and no reuse of legacy observation semantics.
    config = replace(config, episode_duration_s=.15, particle_count=2)
    env = MCAPhysicalEnv(config)
    obs, _ = env.reset(seed=42)
    agent = make_physical_agent(env, seed=42)
    torch.set_num_threads(1)
    np.random.seed(42)
    transitions, episodes = 0, 0
    while transitions < 6:
        raw, lp, value, execution, context, state = physical_policy_action(agent, env, obs)
        returned, _, term, trunc, info = env.step(execution)
        store_physical_transition(agent, obs, raw, lp, value, context, state, env,
                                   returned, info, term, trunc)
        transitions += 1
        obs = returned
        if term or trunc:
            episodes += 1
            if transitions < 6:
                obs, _ = env.reset(seed=42+episodes)
    metrics = agent.update(n_epochs=1, batch_size=6)
    if not all(np.isfinite(v) for v in metrics.values()):
        raise AssertionError('PPO interface produced nonfinite metrics')
    record = dict(transitions=transitions, completed_episodes=episodes,
                  optimizer_updates=1, metrics=metrics, checkpoint_written=False,
                  policy_meta=agent.meta, performance_estimate=None)
    (out / 'ppo_smoke.json').write_text(json.dumps(jsonable(record), indent=2, allow_nan=False)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, default=.1,
                        help='Physical trajectory duration; original audit is 0.1 seconds')
    args = parser.parse_args()
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error('--duration-s must be finite and positive')
    args.out.mkdir(parents=True, exist_ok=False)
    config = DynamicsConfig.from_json()
    (args.out/'config.json').write_bytes(CONFIG_PATH.read_bytes())
    protocol = dict(seeds=SEEDS, spatial_fractions=FRACTIONS,
                    trajectory_duration_s=args.duration_s, maximum_position_error_mm=MAX_POSITION_ERROR_MM,
                    convergence_compares='all control-time samples and final state against .025 reference; identical initial states/actions',
                    required_event_checks='same sampling grid, active masks and exit-node identity',
                    formal_training=False, policy_performance_estimate=None)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records, convergence = [], []
    for seed in SEEDS:
        batch = []
        for fraction in FRACTIONS:
            record = run_trajectory(config, seed, fraction, args.out, duration_s=args.duration_s)
            batch.append(record)
            records.append(record)
            print(json.dumps(dict(seed=seed, fraction=fraction, substeps=record['total_substeps'],
                                  seconds=round(record['wall_time_s'], 3))), flush=True)
        if len({r['initial_state_sha256'] for r in batch}) != 1:
            raise AssertionError('Convergence runs do not share the same initial state')
        reference = batch[-1]
        traces = []
        for fraction in FRACTIONS:
            with np.load(args.out/f'seed_{seed}_fraction_{fraction:g}.npz') as data:
                traces.append({key: data[key] for key in data.files})
        ref_trace = traces[-1]
        grid_match = all(np.array_equal(t['time_s'], ref_trace['time_s']) for t in traces[:-1])
        differences = []
        for trace in traces[:-1]:
            count = min(len(trace['time_s']), len(ref_trace['time_s']))
            differences.append(np.linalg.norm(trace['positions_mm'][:count] -
                                               ref_trace['positions_mm'][:count], axis=-1))
        masks_match = all(np.array_equal(t['active'], ref_trace['active']) for t in traces[:-1])
        exits_match = all(np.array_equal(t['exit_node'], ref_trace['exit_node']) for t in traces[:-1])
        coarse, fine = float(differences[0].max()), float(differences[1].max())
        # Compare every body, including passive tracers and exited bodies.
        convergence.append(dict(seed=seed, coarse_error_mm=coarse, fine_error_mm=fine,
                                robot_fine_error_mm=float(differences[1][:, :config.num_robots].max()),
                                final_fine_error_mm=float(np.linalg.norm(batch[1]['final_positions_mm'] -
                                                            reference['final_positions_mm'], axis=-1).max()),
                                sampling_grid_match=grid_match, active_masks_match=masks_match,
                                exit_nodes_match=exits_match,
                                passed=grid_match and masks_match and exits_match and
                                    fine <= MAX_POSITION_ERROR_MM and fine <= coarse + 1e-10))
    optimizer = ppo_smoke(config, args.out)
    files = ['environments/mca_physical_dynamics.py', 'environments/mca_physical_env.py',
             'environments/mca_physiology.py', 'environments/vessel_geometry.py',
             'environments/vessel_anatomy.py', 'environments/vessel_tree_generator.py',
             'marl/mca_physical_policy.py', 'marl/mappo_advanced.py', 'marl/geometric_control.py',
             'scripts/validate_mca_dynamics.py', 'tests/test_mca_physical_dynamics.py']
    summary = dict(protocol=protocol, convergence=convergence, trajectories=records,
                   convergence_passed=all(r['passed'] for r in convergence),
                   ppo_interface_passed=True, formal_training_ready=False,
                   policy_success_rate=None, optimizer_smoke=optimizer,
                   created_at_utc=datetime.now(timezone.utc).isoformat(),
                   source_sha256={f: hashfile(ROOT/f) for f in files},
                   config_sha256=hashfile(CONFIG_PATH))
    (args.out/'summary.json').write_text(json.dumps(jsonable(summary), ensure_ascii=False,
                                                   indent=2, allow_nan=False)+'\n')
    lines = ['# EXP22B — 原尺寸MCA动态环境验证', '',
             f"数值收敛检查：{'PASS' if summary['convergence_passed'] else 'FAIL'}。新策略接口：PASS。", '',
             '已实现：毫米/秒积分、局部子步、闭塞尺寸约束、入口/末端离场、永久失活掩码、同流场被动粒子、按秒接触溶解、时间上限截断。',
             '动作由策略直接输出Frenet单位向量，转换到世界系后保持一个控制周期，无导航控制器或残差项。', '',
             '|seed|0.1相对0.025最大位置误差 mm|0.05相对0.025误差 mm|机器人误差 mm|通过|',
             '|---:|---:|---:|---:|---|']
    for r in convergence:
        lines.append(f"|{r['seed']}|{r['coarse_error_mm']:.6f}|{r['fine_error_mm']:.6f}|"
                     f"{r['robot_fine_error_mm']:.6f}|{r['passed']}|")
    lines.extend(['', '验收阈值保持0.05 mm，比较所有控制时刻而非只看终点；要求细分误差不超过粗分、采样时间/活跃掩码/出口编号一致；含全部机器人和粒子。',
                  f'轨迹长度{args.duration_s:g}秒。三seed是同一名义几何上的粒子/动作随机化，不能当三个独立临床样本；结果仅覆盖该时间窗和动作/几何。',
                  '', 'PPO冒烟：新权重、6条transition、1次优化器更新，未保存checkpoint，不能用于成功率结论。',
                  '**没有正式训练或85%成功率结果。** 36维观测语义已更换；策略桥拒绝缺少物理schema标签的旧权重。',
                  '', '限制：分叉为局部管段几何投影而非CFD；理想独立速度执行器，不是共享磁场；粒子是被动示踪物，仅报告重叠时间，不施加碰撞力。',
                  '溶解率0.36归一化质量/秒、接触距离0.12mm、理想推进无润滑损失均为显式未测工程假设；没有添加未标定Brownian噪声。',
                  '正式训练前仍需生理/硬件参数标定、长期与分叉数值检查及任务协议。', ''])
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    artifacts = sorted(p for p in args.out.iterdir() if p.is_file())
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(
        status='complete' if summary['convergence_passed'] else 'numerical_gate_failed',
        artifact_sha256={p.name: hashfile(p) for p in artifacts}), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.out.resolve()), convergence=convergence,
                          ppo_interface_passed=True, formal_training_ready=False)), flush=True)
    return 0 if summary['convergence_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
