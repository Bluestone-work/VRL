"""Audit EXP0047 arms and estimate paired scene-level learning effects."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


METRICS = ('cluster_safe_success', 'task_success', 'removal', 'removal_auc_180',
           'wall_contact_s', 'spacing_violation_pair_s')


def exact_binomial_ci(successes, trials):
    """Per-fixed-policy scene interval, including all-zero boundary uncertainty."""
    if trials == 0: return [0., 1.]
    lower = 0. if successes == 0 else float(beta.ppf(.025, successes, trials-successes+1))
    upper = 1. if successes == trials else float(beta.ppf(.975, successes+1, trials-successes))
    return [lower, upper]


def scene_bootstrap(differences, seed=4707):
    x = np.asarray(differences, float)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(), scenes=len(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, issues, sources, expected, scenes, raw_hashes = [], [], [], {}, {}, {}
    contracts, stages = set(), set()
    actual_configs, sensor_specs, catalytic_budgets = {}, {}, {}
    for folder in args.input:
        manifest = json.loads((folder/'manifest.json').read_text())
        sources.append(manifest['source_hashes'])
        contracts.add(manifest.get('observation_contract', 'ideal'))
        stages.add(manifest['stage'])
        path = folder/'attempts.jsonl'
        raw_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if len(data) != manifest['requested']:
            issues.append(f'{folder}: incomplete requested attempts')
        for arm in manifest['arms']:
            if arm['label'] in expected: issues.append('Duplicate arm '+arm['label'])
            expected[arm['label']] = manifest['scenes']
        rows += data
    if any(s != sources[0] for s in sources): issues.append('Source mismatch between matrices')
    if len(contracts) != 1: issues.append('Mixed observation contracts')
    if len(stages) != 1: issues.append('Mixed development and confirmation stages')
    tracked = bool(contracts) and contracts <= {'tracked', 'temporal', 'memory_prior'}
    metrics = METRICS + (('sensor_confirmed_all_cleared', 'false_visual_completion',
                         'physical_cluster_safe_success', 'T90_with_horizon_penalty',
                         'elapsed_s', 'command_squared_s') if tracked else ())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['arm']].append(row)
        if row['status'] != 'completed':
            issues.append(f"Failed {row['arm']} scene {row['scene_seed']}: {row['status']}")
        else:
            if tracked:
                if not row.get('completion_uses_observed_confirmation', False):
                    issues.append('Tracked rollout used oracle completion')
                row['T90_with_horizon_penalty'] = (
                    row['time_to_removal_s']['90'] if row['time_to_removal_s']['90'] is not None else 180.)
                reset = row.get('reset_info', {})
                config, sensor = reset.get('actual_config'), row.get('sensor_assumptions')
                if config is None or sensor is None:
                    issues.append('Missing actual physics/sensor provenance')
                else:
                    key = (row['scene_seed'], row['clusters'])
                    if key in actual_configs and actual_configs[key] != config:
                        issues.append('Unmatched actual simulator configuration')
                    if row['scene_seed'] in sensor_specs and sensor_specs[row['scene_seed']] != sensor:
                        issues.append('Unmatched actual sensor assumptions')
                    actual_configs[key], sensor_specs[row['scene_seed']] = config, sensor
                    budget = reset.get('aggregate_lysis_capacity_mass_per_s')
                    if budget is None or reset.get('resource_budget') != 'fixed_total':
                        issues.append('Missing/unmatched catalytic budget')
                    elif row['scene_seed'] in catalytic_budgets and abs(catalytic_budgets[row['scene_seed']]-budget)>1e-12:
                        issues.append('Unmatched aggregate catalytic capacity')
                    else:
                        catalytic_budgets[row['scene_seed']] = budget
            if row['source_hashes'] != sources[0]: issues.append('Row source mismatch')
            seed = row['scene_seed']
            if seed in scenes and scenes[seed] != row['scenario_hash']: issues.append('Unpaired scene')
            scenes[seed] = row['scenario_hash']
    for arm, seeds in expected.items():
        if sorted(r['scene_seed'] for r in grouped[arm]) != sorted(seeds):
            issues.append('Missing/duplicate scene in '+arm)
    summary = dict(audit_passed=not issues, issues=issues, raw_hashes=raw_hashes,
                   scenes=len(scenes), arms={}, comparisons=[], cluster_count_comparisons=[], diagnostic_only=True,
                   observation_contracts=sorted(contracts), stages=sorted(stages))
    summary['actual_configuration_pairs_checked'] = len(actual_configs)
    for arm, data in grouped.items():
        complete = [r for r in data if r['status'] == 'completed']
        success = sum(r['cluster_safe_success'] for r in complete)
        count = len(expected[arm])
        summary['arms'][arm] = dict(completed=len(complete), requested=count,
            success_missing_bounds=[success/count, (success+count-len(complete))/count],
            cluster_safe_ci95_missing_envelope=[exact_binomial_ci(success, count)[0],
                exact_binomial_ci(success+count-len(complete), count)[1]],
            conditional_means={k: float(np.mean([r[k] for r in complete])) for k in metrics} if complete else {})
    # Comparisons are admitted only if every requested process succeeded.
    if not issues:
        learned_groups = defaultdict(list)
        for arm, data in grouped.items():
            if arm.startswith('learned_'):
                learned_groups[data[0]['training_steps']] += data
        for steps, data in sorted(learned_groups.items()):
            per_scene = defaultdict(list)
            for row in data: per_scene[row['scene_seed']].append(row)
            for comparator in ('joint_n3', 'memory_n3', 'v2_spacing_n3'):
                if comparator not in grouped: continue
                baseline = {r['scene_seed']: r for r in grouped[comparator]}
                if per_scene.keys() != baseline.keys(): raise ValueError('Incomplete pairing')
                changes = {k: scene_bootstrap([np.mean([r[k] for r in per_scene[s]])-baseline[s][k]
                           for s in sorted(per_scene)]) for k in metrics}
                summary['comparisons'].append(dict(training_steps=steps, comparator=comparator,
                    direction='learning minus heuristic', training_seeds=sorted({r['training_seed'] for r in data}),
                    inference_unit='scene; learning-seed repeats averaged within scene', metrics=changes))
        for controller in ('joint', 'memory'):
            multi, single = controller+'_n3', controller+'_n1'
            if multi not in grouped or single not in grouped: continue
            a = {r['scene_seed']:r for r in grouped[multi]}
            b = {r['scene_seed']:r for r in grouped[single]}
            if a.keys() != b.keys(): raise ValueError('Incomplete N=1/N=3 scene pairing')
            changes = {key:scene_bootstrap([a[s][key]-b[s][key] for s in sorted(a)]) for key in metrics}
            summary['cluster_count_comparisons'].append(dict(controller=controller,
                direction='N=3 minus N=1; same local controller and catalytic-rate proxy budget',
                attribution='parallelism comparison, not a learning contribution', metrics=changes))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    title = 'EXP0049 temporal-candidate' if contracts == {'temporal'} else ('EXP0048 tracked' if tracked else 'EXP0047 ideal-sensor')
    if contracts == {'memory_prior'}: title = 'EXP0050 memory-prior'
    lines = [f"# {title} local-learning analysis", '',
        f"Audit: {'PASS' if not issues else 'BLOCKED'}; {len(rows)} recorded attempts; {len(scenes)} scenes.", '',
        'Rates below are developmental, not sealed results. Three training seeds on one scene '
        'are correlated repeats. Intervals resample scenes, not individual robot samples. '
        'Secondary-endpoint intervals are exploratory and not multiplicity adjusted. '
        'A degenerate all-zero bootstrap difference interval is NOT evidence of equivalence. '
        'Per-arm primary-rate intervals below are exact binomial intervals for fixed policies '
        'across scenes, widened over unknown failures if necessary.', '',
        '| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Cluster-safe 95% CI % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, v in summary['arms'].items():
        m = v['conditional_means']
        cells = [f"{m[k]*(100 if k in METRICS[:4] else 1):.3f}" for k in METRICS] if m else ['NA']*6
        lo,hi = v['cluster_safe_ci95_missing_envelope']
        cells.append(f'[{100*lo:.3f}, {100*hi:.3f}]')
        lines.append(f"| {arm} | {v['completed']}/{v['requested']} | "+' | '.join(cells)+' |')
    if tracked:
        lines += ['', 'Primary success requires physical clearance AND measured completion, '
            'with safety evaluated through observed stopping time. All sensing values are '
            'engineered stress assumptions; this is not a validated imaging or hardware study.', '',
            '| Arm | Visual completion % | False visual completion % | Physically safe % | T90, failure=180 s | Observed stop s | Squared command s |',
            '|---|---:|---:|---:|---:|---:|---:|']
        keys = ('sensor_confirmed_all_cleared', 'false_visual_completion',
                'physical_cluster_safe_success', 'T90_with_horizon_penalty', 'elapsed_s', 'command_squared_s')
        for arm, v in summary['arms'].items():
            m = v['conditional_means']
            cells = [f"{m[k]*(100 if j < 3 else 1):.3f}" for j, k in enumerate(keys)] if m else ['NA']*6
            lines.append(f"| {arm} | "+' | '.join(cells)+' |')
    if issues:
        lines += ['', '**Comparisons blocked:**']+['- '+x for x in issues]
    else:
        lines += ['', '## Paired effects (learning minus heuristic)', '',
            '| Budget / comparator | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] | Spacing pair-s [95% CI] |',
            '|---|---:|---:|---:|---:|---:|']
        for comparison in summary['comparisons']:
            cells = []
            for k in ('cluster_safe_success', 'removal', 'removal_auc_180', 'wall_contact_s', 'spacing_violation_pair_s'):
                value = comparison['metrics'][k]
                scale = 100 if k in METRICS[:4] else 1
                cells.append(f"{value['mean']*scale:+.3f} [{value['ci95'][0]*scale:+.3f}, {value['ci95'][1]*scale:+.3f}]")
            lines.append(f"| {comparison['training_steps']} / {comparison['comparator']} | "+' | '.join(cells)+' |')
        if summary['cluster_count_comparisons']:
            lines += ['', '## Method 2 versus method 1, same conventional control', '',
                'These paired effects isolate the stated cluster-count comparison, not learning. '
                'Aggregate catalytic-rate proxy is matched; magnetic power, material and deployment are not.', '',
                '| Controller | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] |',
                '|---|---:|---:|---:|---:|']
            for comparison in summary['cluster_count_comparisons']:
                cells=[]
                for key in ('cluster_safe_success','removal','removal_auc_180','wall_contact_s'):
                    v=comparison['metrics'][key];scale=100 if key in METRICS[:4] else 1
                    cells.append(f"{v['mean']*scale:+.3f} [{v['ci95'][0]*scale:+.3f}, {v['ci95'][1]*scale:+.3f}]")
                lines.append(f"| {comparison['controller']} | "+' | '.join(cells)+' |')
    lines += ['', 'Joint filtering is shared with the strong conventional comparator. '
        'Its improvement over old v2 spacing is not a learning contribution. No claim of '
        'calibrated magnetic independence, hardware transfer or safety certification is made.', '']
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(audit_passed=not issues, issues=issues,
                         arms=summary['arms'], comparisons=summary['comparisons']), indent=2))


if __name__ == '__main__':
    main()
