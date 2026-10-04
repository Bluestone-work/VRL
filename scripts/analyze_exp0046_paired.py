"""Audit and summarize frozen EXP0046 v2 diagnostics without rerunning episodes.

Inference units are scenes, with controller repeats averaged within scenes.
All comparisons are exploratory; confidence intervals are not multiplicity adjusted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 46062026
BOOTSTRAP_REPEATS = 10000


def cell_label(c):
    return (f"{c['method']}_n{c['clusters']}_d{c['d_min_mm']:g}"
            f"_s{c['control_seed']}_{c['budget']}")


def arm_label(c):
    return f"{c['method']}_n{c['clusters']}_d{c['d_min_mm']:g}_{c['budget']}"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    issues, rows, files = [], [], []
    hashes, accepted, base_configs = defaultdict(set), defaultdict(set), set()
    expected_labels = {cell_label(c) for c in manifest['cells']}
    actual_labels = {p.stem for p in directory.glob('*.jsonl')}
    if actual_labels != expected_labels:
        issues.append('Missing or unexpected cell JSONL files')
    for cell in manifest['cells']:
        label = cell_label(cell)
        path = directory/(label+'.jsonl')
        if not path.exists():
            issues.append(f'{label}: missing output')
            continue
        files.append(dict(path=path.name, sha256=sha(path)))
        attempts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if sorted(r['seed'] for r in attempts) != sorted(manifest['scenarios']):
            issues.append(f'{label}: missing, duplicate, or unexpected scene identity')
        for row in attempts:
            identity = (row['method'], row['clusters'], row['min_spacing_threshold_mm'],
                        row['control_seed'], row['resource_budget'])
            expected = (cell['method'], cell['clusters'], cell['d_min_mm'],
                        cell['control_seed'], cell['budget'])
            if identity != expected:
                issues.append(f'{label}: mismatched row identity')
            row['_arm'] = arm_label(cell)
            rows.append(row)
            if row['status'] != 'completed':
                issues.append(f'{label}, seed {row["seed"]}: {row["status"]}')
                continue
            if row['source_hashes'] != manifest['source_hashes']:
                issues.append(f'{label}: source hash mismatch')
            if row['revision'] != manifest['revision'] or row['sealed_test_used']:
                issues.append(f'{label}: wrong revision or sealed use')
            if row['development_guard_hit']:
                issues.append(f'{label}: development step guard hit')
            reset = row['reset_info']
            computed = hashlib.sha256(json.dumps(reset['shared_scene'], sort_keys=True,
                                                allow_nan=False).encode()).hexdigest()
            if computed != reset['scenario_hash']:
                issues.append(f'{label}: shared scene checksum mismatch')
            hashes[row['seed']].add(reset['scenario_hash'])
            accepted[row['seed']].add(reset['accepted_seed'])
            cfg = row['actual_config']
            common = {k: v for k, v in cfg.items() if k not in ('num_robots', 'lysis_mass_per_s')}
            base_configs.add(json.dumps(common, sort_keys=True))
            rate = .36/row['clusters'] if row['resource_budget'] == 'fixed_total' else .36
            if not np.isclose(cfg['lysis_mass_per_s'], rate) or cfg['action_prior'] != 'none':
                issues.append(f'{label}: resource/actuator convention mismatch')
            if cfg['episode_duration_s'] != manifest['duration_s']:
                issues.append(f'{label}: wrong horizon')
            if row['local_observation'] != (row['method'] != 'single_route'):
                issues.append(f'{label}: observation class mismatch')
    if any(len(v) != 1 for v in hashes.values()):
        issues.append('Shared scenes differ across arms')
    if any(len(v) != 1 for v in accepted.values()):
        issues.append('Accepted seeds differ across arms')
    all_accepted = [next(iter(v)) for v in accepted.values() if len(v) == 1]
    if len(set(all_accepted)) != len(all_accepted):
        issues.append('Distinct requested scenes alias the same accepted scene')
    if len(base_configs) != 1:
        issues.append('Shared dynamics differ across arms')
    for file, digest in manifest['source_hashes'].items():
        if sha(directory/'source_snapshot'/file) != digest:
            issues.append(f'Frozen snapshot mismatch: {file}')
        if sha(ROOT/file) != digest:
            issues.append(f'Working source changed since freeze: {file}')
    status_path = directory/'matrix_status.json'
    if not status_path.exists():
        issues.append('Matrix runner has not finished')
    else:
        status = json.loads(status_path.read_text())
        if (not status['complete'] or len(status['results']) != len(manifest['cells'])
                or any(r.get('error') or r.get('failed', 0) for r in status['results'])):
            issues.append('Runner reports an incomplete or failed cell')
    if len(rows) != manifest['requested_episodes']:
        issues.append('Requested episode count mismatch')
    return manifest, rows, dict(passed=not issues, issues=issues,
        requested=manifest['requested_episodes'], recorded=len(rows),
        completed=sum(r['status'] == 'completed' for r in rows),
        failed=sum(r['status'] != 'completed' for r in rows),
        unique_scenes=len(hashes), accepted_seeds=sorted(all_accepted), raw_files=files)


def metrics(row, horizon):
    """Common-horizon AUC and failure-penalized times use every completed attempt."""
    remaining = max(0., horizon-row['elapsed_s'])
    result = {k: float(row[k]) for k in (
        'task_success', 'safe_success', 'cluster_safe_success', 'removal',
        'wall_contact_s', 'particle_contact_s', 'robot_pair_contact_s',
        'spacing_violation_pair_s', 'lost_clusters', 'path_mm', 'command_squared_s',
        'mass_per_cluster_s', 'active_cluster_s', 'timeout')}
    result['auc_180_fraction'] = (row['removal_auc_s']+remaining*row['removal'])/horizon
    result['spacing_violation_rate'] = float(not row['spacing_compliant'])
    result['wall_unsafe_rate'] = float(row['wall_contact_s'] >= 1.)
    result['particle_contact_rate'] = float(row['particle_contact_s'] > 0.)
    result['pair_contact_rate'] = float(row['robot_pair_contact_s'] > 0.)
    result['lost_any_rate'] = float(row['lost_clusters'] > 0)
    for k in ('50', '90', '100'):
        reached = row['time_to_removal_s'][k]
        result[f't{k}_reached_rate'] = float(reached is not None)
        result[f't{k}_horizon_penalty_s'] = horizon if reached is None else float(reached)
    for k in ('yield_events', 'yield_agent_steps', 'persistent_yield_events'):
        result[k] = float(row['controller'].get(k, 0))
    return result


def scene_vectors(rows, horizon):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['seed']].append(metrics(row, horizon))
    return {seed: {k: float(np.mean([r[k] for r in values])) for k in values[0]}
            for seed, values in grouped.items()}


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                scenes=len(values))


def compare(name, a, b, horizon):
    av, bv = scene_vectors(a, horizon), scene_vectors(b, horizon)
    if av.keys() != bv.keys():
        raise ValueError(f'{name}: incomplete pairing')
    for scene in av:
        ac = {r['control_seed'] for r in a if r['seed'] == scene}
        bc = {r['control_seed'] for r in b if r['seed'] == scene}
        if ac != bc:
            raise ValueError(f'{name}: unmatched control seeds')
    return dict(name=name, direction='A minus B',
                metrics={k: interval([av[s][k]-bv[s][k] for s in sorted(av)]) for k in av[next(iter(av))]})


def analyze(directory):
    manifest, rows, result = audit(directory)
    if not result['passed']:
        descriptive = {}
        for cell in manifest['cells']:
            name = arm_label(cell)
            if name in descriptive:
                continue
            attempted = [r for r in rows if r['_arm'] == name]
            complete = [r for r in attempted if r['status'] == 'completed']
            total = len(manifest['scenarios'])*sum(arm_label(c) == name for c in manifest['cells'])
            successes = sum(r['cluster_safe_success'] for r in complete)
            measured = [metrics(r, manifest['duration_s']) for r in complete]
            descriptive[name] = dict(requested=total, recorded=len(attempted), completed=len(complete),
                failed=sum(r['status'] != 'completed' for r in attempted),
                missing=total-len(complete), cluster_safe_success_bounds=[successes/total,
                    (successes+total-len(complete))/total],
                conditional_completed_means={k: float(np.mean([r[k] for r in measured]))
                    for k in measured[0]} if measured else {})
        return dict(audit=result, descriptive_only=descriptive, comparisons_blocked=True,
                    failed_attempts=[r for r in rows if r['status'] != 'completed'],
                    analysis_source_sha256=sha(Path(__file__)))
    horizon = manifest['duration_s']
    groups = defaultdict(list)
    for row in rows:
        groups[row['_arm']].append(row)
    arms = {}
    for name, group in groups.items():
        scenes = scene_vectors(group, horizon)
        arms[name] = dict(episodes=len(group), scenes=len(scenes),
            control_seeds=sorted({r['control_seed'] for r in group}),
            metrics={k: interval([v[k] for v in scenes.values()]) for k in next(iter(scenes.values()))},
            minimum_spacing_mm=min((r['minimum_spacing_mm'] for r in group
                                    if r['minimum_spacing_mm'] is not None), default=None),
            t100_completed_only_mean_s=float(np.mean([r['time_to_removal_s']['100'] for r in group
                if r['time_to_removal_s']['100'] is not None])) if any(r['time_to_removal_s']['100'] is not None for r in group) else None)
    comparisons = []
    single = groups['single_sequential_n1_d2_fixed_total']
    for n in (2, 3):
        for d in (1, 2, 4):
            parallel = groups[f'multi_parallel_n{n}_d{d}_fixed_total']
            comparisons.append(compare(f'parallel N={n}, d={d} vs local single', parallel, single, horizon))
        shield = [r for r in groups[f'multi_parallel_n{n}_d2_fixed_total'] if r['control_seed'] == 42]
        comparisons.append(compare(f'shield vs unshielded N={n}, seed42', shield,
                                   groups[f'multi_unshielded_n{n}_d2_fixed_total'], horizon))
        comparisons.append(compare(f'per-cluster vs fixed-total budget N={n}, seed42',
                                   groups[f'multi_parallel_n{n}_d2_per_cluster'], shield, horizon))
    comparisons.append(compare('privileged route vs local single, seed42',
                               groups['single_route_n1_d2_fixed_total'],
                               [r for r in single if r['control_seed'] == 42], horizon))
    return dict(audit=result, arms=arms, comparisons=comparisons,
                bootstrap=dict(unit='scene; controller seeds averaged within scene',
                    seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPEATS,
                    exploratory=True, multiplicity_adjusted=False),
                auc_convention='Hold terminal removal through the common 180 s horizon; raw AUC retained.',
                time_convention='Unreached milestones receive 180 s penalty; not an estimate of their true completion time.',
                analysis_source_sha256=sha(Path(__file__)))


def report(result):
    audit_result = result['audit']
    lines = ['# EXP0046 revision 2 paired diagnostic results — 2026-10-04', '',
        f"Audit: **{'PASS' if audit_result['passed'] else 'BLOCKED'}**; "
        f"{audit_result['completed']}/{audit_result['requested']} episodes completed; "
        f"{audit_result['failed']} failed processes.", '']
    if not audit_result['passed']:
        lines += ['The predeclared numerical gate failed. **No method ranking or paired performance '
            'claim is admissible from this batch.** Original failures remain in the requested '
            'denominator; later debugging reproductions cannot replace them.', '']
        lines += ['- '+s for s in audit_result['issues']]
        lines += ['', '## Descriptive outcomes only', '',
            'Rows below are for debugging and next-study design. Metrics are conditional on '
            'completed processes; bounds count every requested episode and treat missing results '
            'as failure/success at the lower/upper endpoints. No intervals, ranking or superiority '
            'claim is made. There are 10 scenes; repeated controller seeds are not independent scenes.', '',
            '| Arm | Complete / requested | Failed | Cluster-safe bounds % | Removal % among complete | AUC % among complete | Spacing violations % among complete |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name, arm in result['descriptive_only'].items():
            m = arm['conditional_completed_means']
            low, high = arm['cluster_safe_success_bounds']
            values = ' | '.join(f"{100*m[k]:.1f}" if k in m else 'NA' for k in
                ('removal', 'auc_180_fraction', 'spacing_violation_rate'))
            lines.append(f"| {name} | {arm['completed']}/{arm['requested']} | {arm['failed']} | "
                         f"[{100*low:.1f}, {100*high:.1f}] | {values} |")
        lines += ['', '## Definitions and next gate', '',
            '- Primary cluster-safe success: full clearing, <1 total cluster-s wall contact, no particle/pair contact, no cluster loss, no separation violation.',
            '- AUC holds terminal removal to a common 180 s horizon. Unreached clearance times receive a 180 s penalty in the detailed JSON, not a fictitious completion time.',
            '- Fixed total catalytic rate is 0.36 mass/s; per-cluster arms have N times that capacity. Neither total magnetic power nor material volume is matched.',
            '- The policy excludes route/true-flow/edge queries. Local geometry and tracking, including 6 mm peer imaging, remain simulated sensor assumptions.',
            '- Separation control is heuristic; ideal independent actuators are uncalibrated. These results do not establish magnetic independence or hardware transfer.',
            '- Capture and diagnose the failing process first. Keep any subsequent runtime repair and new performance batch separately registered; do not silently rerun the missing row.',
            '', 'All raw checksums, missing-data bounds and conditional metrics are in `analysis.json`.', '']
        return '\n'.join(lines)
    lines += ['This is an exploratory MCA development study: 10 distinct scenes, 180 s horizon. '
        'Seeds 42/43/44 are controller and measurement-noise repeats, not independent training runs. '
        'No sealed test and no RL training. Frozen source, all episode identities, shared scene hashes, '
        'unique accepted scene seeds and resource conventions pass the audit.', '',
        '## Per-arm outcomes', '',
        '| Arm | Episodes | Raw % | Safe % | Cluster-safe % | Removal % | AUC / 180 % | T100 penalty s | Spacing violation % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, arm in result['arms'].items():
        m = arm['metrics']
        values = [100*m[k]['mean'] for k in ('task_success', 'safe_success', 'cluster_safe_success',
                                            'removal', 'auc_180_fraction')]
        values += [m['t100_horizon_penalty_s']['mean'], 100*m['spacing_violation_rate']['mean']]
        lines.append(f"| {name} | {arm['episodes']} | "+' | '.join(f'{v:.1f}' for v in values)+' |')
    lines += ['', '## Paired differences with scene-bootstrap 95% intervals', '',
        'Each row is A minus B. Positive success/removal/AUC favors A; negative time and spacing '
        'violations favor A. Intervals resample 10 scenes (10,000 draws), averaging repeated controller '
        'seeds within each scene. Small diagnostic sample; no multiple-comparison correction.', '',
        '| Comparison | Cluster-safe pp | Removal pp | AUC pp | T100 penalty s | Spacing violation pp |',
        '|---|---:|---:|---:|---:|---:|']
    for comparison in result['comparisons']:
        cells = []
        for k in ('cluster_safe_success', 'removal', 'auc_180_fraction',
                  't100_horizon_penalty_s', 'spacing_violation_rate'):
            v = comparison['metrics'][k]
            scale = 1 if k.endswith('_s') else 100
            cells.append(f"{scale*v['mean']:+.1f} [{scale*v['ci95'][0]:+.1f}, {scale*v['ci95'][1]:+.1f}]")
        lines.append('| '+comparison['name']+' | '+' | '.join(cells)+' |')
    lines += ['', '## Definitions and limits', '',
        '- Safe Success requires full clearing and <1 total cluster-second wall contact. Cluster-safe success also requires no particle contact, no lost cluster, no pair contact and no threshold violation.',
        '- Raw AUC stops when an episode terminates. For fair efficiency comparison, terminal removal is held through 180 s; successful early termination thus receives the remaining full-clearance area. Raw files are unchanged.',
        '- T50/T90/T100 are null in raw files when unreached. Table times assign those episodes a 180 s penalty, including exits. This is a failure-aware score, not an estimate of eventual completion or a survival estimator.',
        '- Primary arms fix total catalytic-rate proxy at 0.36 mass/s. They do not match total material volume or magnetic power. Secondary per-cluster arms deliberately increase total capacity.',
        '- N=1 has no cluster-pair separation constraint. Starts are nested, pre-deployed and rotated across scenes; this is not common-inlet deployment.',
        '- Policy input excludes route maps, true flow and edge IDs. Imaging of local geometry, tracked clot status, and peers within 6 mm is simulated, with 2.5% feature noise and 0.02 mm centroid error; hardware feasibility remains unverified.',
        '- Actuators are ideally independent. Separation does not establish magnetic controllability; no calibrated field coupling or hardware evidence exists here.',
        '- The spacing filter is a heuristic and can violate its threshold. Prolonged yield is only a deadlock proxy. Squared command is not magnetic energy.',
        '- Earlier native failures were not reproduced in this batch; their cause remains unknown. Zero observed failures does not establish a crash repair.',
        '', 'Detailed metric intervals, contacts, path, active cluster time and paired effects: `analysis.json` and `arm_metrics.csv`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.directory/'REPORT.md').write_text(report(result))
    if not result['audit']['passed']:
        print(json.dumps(result['audit'], indent=2)); raise SystemExit(2)
    with (args.directory/'arm_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arm', 'episodes', 'scenes', 'metric', 'mean', 'ci95_low', 'ci95_high'])
        for name, arm in result['arms'].items():
            for metric, values in arm['metrics'].items():
                writer.writerow([name, arm['episodes'], arm['scenes'], metric, values['mean'], *values['ci95']])
    print(json.dumps({k: v for k, v in result['audit'].items() if k != 'raw_files'}, indent=2))


if __name__ == '__main__':
    main()
