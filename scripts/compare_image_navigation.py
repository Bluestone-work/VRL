"""Compare image-sensing navigation results on exactly paired development scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    'task_success', 'cluster_safe_success', 'relaxed_safe_success',
    'wall_contact_s', 'max_continuous_wall_contact_s',
    'obstacle_events_static', 'obstacle_events_dynamic',
)


def read_rows(path, method=None):
    rows = {}
    errors = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if method is not None and record.get('method', record.get('row', {}).get('method')) != method \
                and record.get('row', {}).get('method') != method:
            continue
        if record.get('error') or not record.get('row'):
            errors.append(record.get('error'))
            continue
        row = dict(record['row'])
        if 'relaxed_safe_success' not in row:
            from marl.hierarchical_navigation import relaxed_success
            row['relaxed_safe_success'] = relaxed_success(row)
        key = (row['anatomy'], int(row['seed']))
        if key in rows:
            raise ValueError(f'duplicate episode in {path}: {key}')
        if row.get('sensing_model') != 'image':
            raise ValueError(f'non-image result in {path}: {key}')
        rows[key] = row
    if not rows:
        raise ValueError(f'no rows selected from {path}')
    return rows, errors


def summary(rows):
    return {
        'episodes': len(rows),
        **{metric: float(np.mean([float(row[metric]) for row in rows.values()])) for metric in METRICS},
    }


def paired(reference, candidate):
    keys = sorted(reference.keys() & candidate.keys())
    if len(keys) != len(reference) or len(keys) != len(candidate):
        raise ValueError('episode sets are not identical')
    for key in keys:
        if reference[key].get('scenario_hash') != candidate[key].get('scenario_hash'):
            raise ValueError(f'scenario mismatch: {key}')
    rng = np.random.default_rng(20261007)
    draws = rng.integers(len(keys), size=(10000, len(keys)))
    result = {}
    for metric in METRICS:
        baseline = np.asarray([float(reference[key][metric]) for key in keys])
        tested = np.asarray([float(candidate[key][metric]) for key in keys])
        delta = tested-baseline
        boot = delta[draws].mean(axis=1)
        result[metric] = dict(reference=float(baseline.mean()), candidate=float(tested.mean()),
                              delta=float(delta.mean()), bootstrap_95=np.quantile(boot, [.025, .975]).tolist())
    return dict(episodes=len(keys), metrics=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-method')
    parser.add_argument('--candidate', action='append', type=Path, required=True)
    parser.add_argument('--candidate-method', action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_method and len(args.candidate_method) != len(args.candidate):
        parser.error('provide one --candidate-method per candidate')
    reference, reference_errors = read_rows(args.reference, args.reference_method)
    report = dict(protocol=dict(reference=str(args.reference), reference_method=args.reference_method,
                                candidates=[str(path) for path in args.candidate],
                                candidate_methods=args.candidate_method),
                  reference=summary(reference), reference_errors=reference_errors, candidates={})
    for index, path in enumerate(args.candidate):
        method = args.candidate_method[index] if args.candidate_method else None
        if method == '-':
            method = None
        rows, errors = read_rows(path, method)
        name = method or path.stem
        report['candidates'][name] = dict(summary=summary(rows), errors=errors,
                                          paired_vs_reference=paired(reference, rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
