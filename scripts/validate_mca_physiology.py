"""Reproducible EXP_0022 unit/flow/reachability audit; does not train a policy.

Run from the project root with python -m scripts.validate_mca_physiology.
All configured cases are reported, including mechanically unfavorable ones.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from environments.mca_physiology import (
    PhysicalUnits, PressureDrivenTreeFlow, mean_speed_mm_s,
    optimistic_wall_upstream_margin,
)
from environments.vessel_anatomy import build_territory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/experiments/EXP_0022_MCA_PHYSIOLOGY.json'
SOURCE_FILES = (
    'environments/mca_physiology.py', 'environments/vessel_anatomy.py',
    'environments/vessel_tree_generator.py', 'scripts/validate_mca_physiology.py',
    'tests/test_mca_physiology.py',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def site_stations(tree):
    """Use every declared site, with no selection by success or flow speed."""
    result = []
    for i, site in enumerate(tree.territory.clot_sites):
        branch = tree.branches[tree.segment_index[site.segment]]
        index = branch.start + int(round(site.position * (branch.size - 1)))
        result.append((f'{site.segment}_site_{i}', index))
    return result


def narrowed_radii_mm(tree, healthy_radius_mm, site, fraction, width_mm):
    """Explicit radius sensitivity, NOT a calibrated clot mass/porosity law."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Radius fraction must be finite and between 0 and 1')
    if not np.isfinite(width_mm) or width_mm <= 0:
        raise ValueError('Occlusion width must be positive and finite')
    arc = (tree.arclength - tree.arclength[site]) * tree.physical_mm_per_unit
    same_branch = tree.branch_ids == tree.branch_ids[site]
    bump = np.exp(-.5 * (arc / width_mm) ** 2) * same_branch
    radii = healthy_radius_mm * (1 - (1 - fraction) * bump)
    radii[site] = healthy_radius_mm[site] * fraction
    return radii


def margin_or_none(mean_speed, radius_mm, robot_radius_mm, robot_speed_mm_s):
    if radius_mm <= robot_radius_mm:
        return None
    return optimistic_wall_upstream_margin(
        mean_speed, radius_mm, robot_radius_mm, robot_speed_mm_s)


def run_audit(config):
    geom = config['geometry_validation']
    occ = config['occlusion_sensitivity']
    reference = config['robot_reference']
    numerical = config['numerics']
    physical = PhysicalUnits(1, config['control_dt_s'], config['robot_speed_mm_s'],
                             config['robot_and_control_parameter_status'])
    if config['physical_geometry_scale'] != 1 or config['primary_scenario'] != 'mca_m1_lvo':
        raise ValueError('This preregistered audit requires original-scale MCA')
    if geom['radius_floor'] != 0 or numerical['flow_speed_cap'] is not None:
        raise ValueError('This audit disallows lumen inflation and flow-speed caps')
    if geom['length_scale_field'] != 'physical_mm_per_unit':
        raise ValueError('Physical geometry requires the actual inverse normalization')
    if config['training_gate']['ready']:
        raise ValueError('Engineering audit cannot declare the RL training gate ready')
    tolerance = numerical['conservation_tolerance_mm3_s']
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Conservation tolerance must be positive and finite')
    checks = dict(healthy_inlet=True, conservation=True, finite_speeds=True,
                  reopening_monotonic=True, closed_site_zero_flow=True,
                  pressure_held_fixed=True)
    rows, margins, numerics, geometries = [], [], [], []
    configurations = itertools.product(geom['variation_levels'], geom['seeds'])
    for variation, seed in configurations:
        tree = build_territory(config['primary_scenario'], rng=np.random.default_rng(seed),
                               variation=variation, min_radius=0)
        scale = tree.physical_mm_per_unit
        units = PhysicalUnits(scale, physical.control_dt_s, physical.robot_speed_mm_s,
                              physical.provenance)
        geometries.append(dict(seed=seed, variation=variation, stations=tree.n_stations,
                               mm_per_unit=scale, legacy_nominal_mm_per_unit=tree.mm_per_unit,
                               inlet_diameter_mm=float(2 * tree.radii[tree.inlet_station] * scale),
                               minimum_radius_mm=float(tree.radii.min() * scale),
                               robot_displacement_per_control_step=units.robot_action_scale,
                               old_0_018_step_as_mm_s=float(units.speed_from_step_displacement(.018)),
                               old_contact_radius_as_mm=.035 * scale,
                               old_300_step_horizon_as_s=300 * physical.control_dt_s,
                               old_lysis_0_018_per_step_as_mass_per_s=.018 / physical.control_dt_s))
        for inlet, distal_ratio in itertools.product(
                config['reference_inlet']['sensitivity_values_ml_min'], occ['distal_resistance_ratios']):
            model = PressureDrivenTreeFlow(tree, scale, inlet, distal_ratio)
            healthy = model.solve()
            checks['healthy_inlet'] &= bool(np.isclose(healthy['inlet_flow_ml_min'], inlet,
                                                       rtol=1e-12, atol=1e-10))
            healthy_pressure = model.driving_pressure
            for site_name, site in site_stations(tree):
                sequence = []
                for fraction in sorted(occ['radius_fractions']):
                    radii = narrowed_radii_mm(tree, model.healthy_radius_mm, site,
                                              fraction, occ['gaussian_width_mm'])
                    solved = model.solve(radii)
                    sequence.append(solved['inlet_flow_ml_min'])
                    checks['conservation'] &= solved['maximum_conservation_residual_mm3_s'] <= tolerance
                    checks['finite_speeds'] &= bool(np.isfinite(solved['mean_speed_mm_s']).all())
                    checks['pressure_held_fixed'] &= model.driving_pressure == healthy_pressure
                    if fraction == 0:
                        checks['closed_site_zero_flow'] &= solved['station_inflow_mm3_s'][site] <= tolerance
                    mean = float(solved['mean_speed_mm_s'][site])
                    radius = float(radii[site])
                    key = dict(seed=seed, variation=variation, healthy_inlet_ml_min=inlet,
                               distal_resistance_ratio=distal_ratio, site=site_name,
                               site_station=int(site), radius_fraction=fraction)
                    maximum_speed = float(2 * solved['mean_speed_mm_s'].max())
                    nominal_margin = margin_or_none(mean, radius, config['robot_radius_mm'],
                                                    config['robot_speed_mm_s'])
                    row = dict(**key, inlet_flow_ml_min=solved['inlet_flow_ml_min'],
                               flow_fraction_of_healthy=solved['inlet_flow_ml_min'] / inlet,
                               root_mean_speed_mm_s=float(solved['mean_speed_mm_s'][model.root]),
                               site_mean_speed_mm_s=mean, site_radius_mm=radius,
                               maximum_centerline_speed_mm_s=maximum_speed,
                               conservation_residual_mm3_s=solved['maximum_conservation_residual_mm3_s'],
                               optimistic_upstream_margin_mm_s=nominal_margin,
                               nominal_robot_fits=nominal_margin is not None)
                    rows.append(row)
                    for rr, speed in itertools.product(reference['radius_sensitivity_mm'],
                                                       reference['speed_sensitivity_mm_s']):
                        margin = margin_or_none(mean, radius, rr, speed)
                        margins.append(dict(**key, robot_radius_mm=rr, robot_speed_mm_s=speed,
                                            optimistic_upstream_margin_mm_s=margin,
                                            status=('DOES_NOT_FIT' if margin is None else
                                                    'UPSTREAM_EXCLUDED_IN_LOCAL_MODEL' if margin < 0 else
                                                    'NOT_EXCLUDED_NOT_PROVEN_FEASIBLE')))
                    positive = radii[radii > 0]
                    if not len(positive):
                        raise ValueError('No positive-radius stations for numerical audit')
                    for period in reference['control_dt_sensitivity_s']:
                        u = PhysicalUnits(scale, period, physical.robot_speed_mm_s, physical.provenance)
                        count = u.required_substeps(maximum_speed + physical.robot_speed_mm_s,
                                                    float(positive.min()),
                                                    numerical['maximum_displacement_fraction_of_radius'])
                        numerics.append(dict(**key, control_dt_s=period, required_substeps=count,
                                             maximum_integration_dt_s=period / count))
                checks['reopening_monotonic'] &= bool(np.all(np.diff(sequence) >= -1e-10))
    if not rows:
        raise ValueError('Audit grid must not be empty')
    inlet = config['reference_inlet']
    nominal_mean = mean_speed_mm_s(inlet['healthy_flow_ml_min'], inlet['diameter_nominal_mm'])
    nominal_margin = optimistic_wall_upstream_margin(
        nominal_mean, inlet['diameter_nominal_mm'] / 2,
        config['robot_radius_mm'], config['robot_speed_mm_s'])
    # Seed repetitions at variation=0 are determinism checks, not independent anatomy.
    healthy_margins = [r for r in margins if r['radius_fraction'] == 1]
    healthy_excluded = sum(r['status'] == 'UPSTREAM_EXCLUDED_IN_LOCAL_MODEL'
                           for r in healthy_margins)
    summary = dict(
        engineering_checks={key: bool(value) for key, value in checks.items()},
        engineering_checks_passed=all(checks.values()),
        geometry_cases=len(geometries), flow_cases=len(rows),
        reachability_cases=len(margins), numerical_cases=len(numerics),
        maximum_conservation_residual_mm3_s=max(r['conservation_residual_mm3_s'] for r in rows),
        nominal_mean_inlet_speed_mm_s=nominal_mean,
        nominal_centerline_inlet_speed_mm_s=2 * nominal_mean,
        nominal_optimistic_upstream_margin_mm_s=nominal_margin,
        healthy_site_upstream_excluded_count=healthy_excluded,
        healthy_site_cases=len(healthy_margins),
        numerical_substeps_range=[min(r['required_substeps'] for r in numerics),
                                  max(r['required_substeps'] for r in numerics)],
        geometries=geometries, training_ready=False, success_rate=None,
        training_gate=config['training_gate'],
        interpretation='Local steady straight-lumen bounds, not a navigation success estimate or proof about arbitrary 3D flow.',
    )
    return summary, rows, margins, numerics


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(summary, config, rows):
    s = summary
    text = [
        '# EXP_0022 — 原尺寸 MCA 生理参数工程验证', '',
        f"工程检查：**{'PASS' if s['engineering_checks_passed'] else 'FAIL'}**。",
        '**RL 训练门槛未通过；未训练新策略；成功率为未评估，不能写成 0% 或 ≥85%。**', '',
        '## 参数含义', '',
        '- 主血管：原尺寸 `mca_m1_lvo`，现有程序化 M1/M2 解剖，不是导入的患者网格。',
        '- 流量参考：PC-MRI 每侧 MCA 146±31 mL/min；115/146/177 是均值±SD敏感性值，不是所有人的正常区间。',
        f"- 3 mm 名义直径下，146 mL/min 换算截面平均 {s['nominal_mean_inlet_speed_mm_s']:.2f} mm/s；抛物线模型中心线 {s['nominal_centerline_inlet_speed_mm_s']:.2f} mm/s。两者不可与多普勒谱速度混用。",
        '- An 2026 补充表4的 0.25–1.00 mm/s 是群体训练随机化范围，不是本设备或单机器人在血液中的实测极限。',
        '- 文献总延迟 <30 ms 不等于动作周期。50 ms（20 Hz）是本轮工程假设，另列30/100 ms数值敏感性。',
        '- 机器人半径0.025/0.05/0.08 mm为未测量尺寸敏感性；不能与论文1.25–1.75 mm宽的群体混同。',
        '- 论文共享磁场不支持直接声称当前每机器人独立速度指令已在硬件实现。', '',
        '## 检查结果', '',
        f"- {s['geometry_cases']} 个几何配置，{s['flow_cases']} 个流场，{s['reachability_cases']} 个尺寸/速度组合，{s['numerical_cases']} 个周期组合。",
        '- variation=0 的三seed是复现检查，不能当三个独立解剖样本；variation=1 是工程扰动，不是已验证临床分布。',
        f"- 最大节点流量守恒误差 {s['maximum_conservation_residual_mm3_s']:.3e} mm³/s。",
        '- 健康流量校准、固定压力、逐步再通流量单调性、完全闭塞零穿越流量、有限速度均逐项检查。',
        '- 单位换算使用新增 physical_mm_per_unit；旧 mm_per_unit 保留。新增元数据不会改变旧几何坐标。',
        f"- 所有预登记组合的积分子步需求为 {s['numerical_substeps_range'][0]}–{s['numerical_substeps_range'][1]}；这是数值需求报告，尚未接入RL积分器。", '',
        '## 可达性约束', '',
        f"名义健康入口，半径0.08 mm、推进1 mm/s时，即使忽略润滑损失，最贴壁可达位置的逆流速度余量仍为 **{s['nominal_optimistic_upstream_margin_mm_s']:.2f} mm/s**。",
        f"健康目标站点检查中，{s['healthy_site_upstream_excluded_count']}/{s['healthy_site_cases']} 个组合在局部稳态圆管模型下排除逆流。该比例是参数网格计数，不是临床或策略成功率。",
        '该结果不能推出顺流到达一定失败，也不能证明任意三维流场都无法逆行；它说明低速机器人在再通后返回上游不能默认可行。余量非负仅表示未被此界排除，不代表路径可达。', '',
        '## 名义 M1 近端血栓的闭塞敏感性', '',
        '|剩余半径比|远端阻力比|总入口流量 mL/min|靶点平均速度 mm/s|机器人可通过|',
        '|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        if (row['seed'] == 42 and row['variation'] == 0 and
                row['healthy_inlet_ml_min'] == config['reference_inlet']['healthy_flow_ml_min'] and
                row['site'] == 'm1_site_0'):
            text.append(f"|{row['radius_fraction']:.2f}|{row['distal_resistance_ratio']:.0f}|"
                        f"{row['inlet_flow_ml_min']:.3f}|{row['site_mean_speed_mm_s']:.3f}|"
                        f"{'是（只检查尺寸）' if row['nominal_robot_fits'] else '否'}|")
    text.extend([
        '', '局部半径变小仍可能使靶点速度上升；固定压力并不保证每处局部速度单调下降。',
        '远端阻力0/1/9、半径收缩宽度2.5 mm均为敏感性假设。只知道健康流量不能唯一反推闭塞流量；不按结果挑低阻力条件。', '',
        '## 进入训练前尚缺', '',
    ])
    text.extend('- ' + item for item in config['training_gate']['blocking_items'])
    text.extend([
        '', '旧300步若硬解释成50 ms/步只有15 s，旧0.018/步的溶解速率也不因此变成测得的生理速率。不能只改血流常数而沿用其它无量纲量。',
        '下一实现阶段必须显式处理顺流出口/离场、物理子步、粒子同流场、接触与溶解每秒速率、以及场执行约束，才可做独立协议的策略训练。',
        '', '## 产物与复现', '',
        '`summary.json` 含代码/配置/资料快照SHA256与软件版本；`flow_cases.csv`、`reachability_cases.csv`、`numerical_cases.csv` 保留全部组合。',
        '复现命令：`python -m scripts.validate_mca_physiology --out <一个新的目录>`。输出目录已存在时拒绝覆盖。',
        '来源核对与适用范围见 `research/experiments/EXP_0022_MCA_PHYSIOLOGY.md`。',
        '',
    ])
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    out = args.out or ROOT / 'research/validation' / (
        'EXP0022_MCA_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_bytes(config_path.read_bytes())
    summary, rows, margins, numerics = run_audit(config)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                         capture_output=True, check=True).stdout.strip()
    source_hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    manifest_path = ROOT / config['source_manifest']
    references = json.loads(manifest_path.read_text())
    for source in references['sources']:
        path = ROOT / source['archive_path']
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source snapshot changed: {path}')
    source_hashes[config['source_manifest']] = sha256(manifest_path)
    document = Path(config['user_document'])
    source_hashes[str(document)] = sha256(document) if document.exists() else None
    summary.update(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   protocol=config['experiment'], git_commit=git,
                   source_sha256=source_hashes, config_sha256=sha256(config_path),
                   references=references,
                   runtime=dict(python=platform.python_version(), numpy=np.__version__),
                   config=config)
    summary['user_document_matches_recorded_hash'] = (
        source_hashes[str(document)] == config['user_document_sha256'])
    summary['engineering_checks']['source_snapshots_verified'] = True
    summary['engineering_checks']['user_document_hash_verified'] = summary['user_document_matches_recorded_hash']
    summary['engineering_checks_passed'] = all(summary['engineering_checks'].values())
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                 allow_nan=False) + '\n')
    write_csv(out / 'flow_cases.csv', rows)
    write_csv(out / 'reachability_cases.csv', margins)
    write_csv(out / 'numerical_cases.csv', numerics)
    (out / 'REPORT.md').write_text(report_markdown(summary, config, rows))
    artifacts = ['config.json', 'summary.json', 'flow_cases.csv',
                 'reachability_cases.csv', 'numerical_cases.csv', 'REPORT.md']
    (out / 'COMPLETE.json').write_text(json.dumps(dict(
        status='complete' if summary['engineering_checks_passed'] else 'checks_failed',
        artifact_sha256={name: sha256(out / name) for name in artifacts}), indent=2) + '\n')
    print(json.dumps(dict(output=str(out.resolve()),
                          engineering_checks=summary['engineering_checks'],
                          flow_cases=summary['flow_cases'],
                          nominal_mean_mm_s=summary['nominal_mean_inlet_speed_mm_s'],
                          upstream_margin_mm_s=summary['nominal_optimistic_upstream_margin_mm_s'],
                          training_ready=False, success_rate=None), ensure_ascii=False))
    return 0 if summary['engineering_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
