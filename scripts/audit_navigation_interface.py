"""Audit saved oracle costs and paired navigation outcomes; never access sealed episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PRIMITIVES = ('advance', 'pass_left', 'pass_right', 'away', 'center', 'slow', 'stop')


def dataset_audit(directory):
    labels, costs, gaps = [], [], []
    files = sorted(Path(directory).glob('*.npz'))
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            labels.append(data['label'])
            costs.append(data['costs'])
            slots = data['seq'][:, -1, 21:57].reshape(-1, 6, 6)
            gaps.append(np.min(np.where(slots[:, :, 5] > .5, slots[:, :, 4], np.inf), axis=1))
    if not labels:
        raise ValueError(f'No oracle episodes in {directory}')
    labels, costs, gaps = np.concatenate(labels), np.concatenate(costs), np.concatenate(gaps)
    if costs.shape != (len(labels), len(PRIMITIVES)) or not np.isfinite(costs).all():
        raise ValueError('Expected finite seven-primitive oracle costs')
    if np.any((labels < 0) | (labels >= len(PRIMITIVES))):
        raise ValueError('Primitive labels out of range')
    ordered = np.sort(costs, axis=1)
    margin = ordered[:, 1]-ordered[:, 0]
    chosen = costs[np.arange(len(labels)), labels]
    result = dict(directory=str(directory), episodes=len(files), samples=len(labels),
                  primitive_names=PRIMITIVES, label_counts=np.bincount(labels, minlength=7).tolist(),
                  label_argmin_consistent=bool(np.allclose(chosen, ordered[:, 0])),
                  median_cost_margin=float(np.median(margin)), margin_below_005=float(np.mean(margin < .05)),
                  always_stop_mean_regret=float(np.mean(costs[:, 6]-ordered[:, 0])),
                  always_advance_mean_regret=float(np.mean(costs[:, 0]-ordered[:, 0])))
    for name, selected in [('detected_gap_below_02', gaps < .2), ('detected_gap_at_least_02', gaps >= .2)]:
        result[name] = dict(samples=int(selected.sum()),
                            label_share=(np.bincount(labels[selected], minlength=7)/max(int(selected.sum()), 1)).tolist())
    return result


def read_rows(path):
    from marl.hierarchical_navigation import relaxed_success
    rows = {}
    errors = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record.get('error') or not record.get('row'):
            errors.append(dict(anatomy=record.get('anatomy'), seed=record.get('seed'), error=record.get('error')))
            continue
        row = record['row'].copy()
        key = (row['anatomy'], row['seed'], row.get('clusters', 1))
        if key in rows:
            raise ValueError(f'Duplicate episode in {path}: {key}')
        row['relaxed_safe_success'] = relaxed_success(row)
        rows[key] = row
    return rows, errors


def paired_audit(reference_path, candidate_path, reference_method=None, candidate_method=None):
    def selected_file(path, method):
        if method is None:
            return read_rows(path)
        from marl.hierarchical_navigation import relaxed_success
        rows, errors = {}, []
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if record.get('method') != method:
                continue
            if record.get('error') or not record.get('row'):
                errors.append(record.get('error'))
                continue
            row = record['row'].copy()
            key = (row['anatomy'], row['seed'], row.get('clusters', 1))
            if key in rows:
                raise ValueError(f'Duplicate {method} episode: {key}')
            row['relaxed_safe_success'] = relaxed_success(row)
            rows[key] = row
        return rows, errors

    reference, reference_errors = selected_file(reference_path, reference_method)
    candidate, candidate_errors = selected_file(candidate_path, candidate_method)
    paired = sorted(reference.keys() & candidate.keys())
    if not paired:
        raise ValueError('No paired episodes')
    for key in paired:
        if reference[key]['scenario_hash'] != candidate[key]['scenario_hash']:
            raise ValueError(f'Scenario mismatch: {key}')
        if 'initial_state_hash' in reference[key] and 'initial_state_hash' in candidate[key]:
            if reference[key]['initial_state_hash'] != candidate[key]['initial_state_hash']:
                raise ValueError(f'Initial state mismatch: {key}')
    result = dict(paired=len(paired), reference_only=len(reference.keys()-candidate.keys()),
                  candidate_only=len(candidate.keys()-reference.keys()),
                  reference_errors=reference_errors, candidate_errors=candidate_errors, metrics={})
    rng = np.random.default_rng(0)
    bootstrap = rng.integers(len(paired), size=(4000, len(paired)))
    for metric in ('cluster_safe_success', 'relaxed_safe_success', 'task_success', 'wall_contact_s',
                   'max_continuous_wall_contact_s', 'obstacle_events_static', 'obstacle_events_dynamic'):
        baseline = np.array([reference[key][metric] for key in paired], float)
        tested = np.array([candidate[key][metric] for key in paired], float)
        delta = tested-baseline
        interval = np.quantile(delta[bootstrap].mean(axis=1), [.025, .975])
        result['metrics'][metric] = dict(reference=float(baseline.mean()), candidate=float(tested.mean()),
                                         delta=float(delta.mean()), paired_bootstrap_95=interval.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='*', type=Path, default=[])
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--reference-method')
    parser.add_argument('--candidate-method')
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    report = dict(datasets=[dataset_audit(directory) for directory in arguments.datasets])
    if bool(arguments.reference) != bool(arguments.candidate):
        parser.error('Provide both --reference and --candidate')
    if arguments.reference:
        report['paired'] = paired_audit(arguments.reference, arguments.candidate,
                                        arguments.reference_method, arguments.candidate_method)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
