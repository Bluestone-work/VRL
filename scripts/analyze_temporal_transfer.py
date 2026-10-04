"""Explicit cross-revision comparison on identical fresh tracked scenes.

The temporal model adds source files; it must preserve every shared tracked
sensor, simulator, reward and conventional-controller source hash. Checkpoint
and model differences are retained, not silently treated as identical sources.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.analyze_local_learning import scene_bootstrap, METRICS


def read_matrix(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    rows = [json.loads(line) for line in (folder/'attempts.jsonl').read_text().splitlines()]
    issues = []
    if len(rows) != manifest['requested']: issues.append('Incomplete attempts: '+str(folder))
    for arm in manifest['arms']:
        found = [r for r in rows if r['arm'] == arm['label']]
        if sorted(r['scene_seed'] for r in found) != sorted(manifest['scenes']):
            issues.append('Missing/duplicate scene: '+arm['label'])
    for row in rows:
        if row['status'] != 'completed':
            issues.append('Failed attempt: '+row['arm']+' / '+str(row['scene_seed']))
        elif row['source_hashes'] != manifest['source_hashes']:
            issues.append('Row source mismatch')
        elif not row.get('completion_uses_observed_confirmation', False):
            issues.append('Unmatched completion rule')
    return manifest, rows, issues


def audit_shared_sources(feedforward, temporal):
    issues = []
    a, b = feedforward['source_hashes'], temporal['source_hashes']
    if any(b.get(k) != v for k, v in a.items()):
        issues.append('Shared tracked source changed between model revisions')
    if feedforward['observation_contract'] != 'tracked' or temporal['observation_contract'] != 'temporal':
        issues.append('Unexpected model contracts')
    if feedforward['stage'] != 'validation' or temporal['stage'] != 'validation':
        issues.append('Cross-revision development analysis rejects confirmation data')
    if feedforward['scenes'] != temporal['scenes']:
        issues.append('Unmatched scene pools')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedforward', type=Path, required=True)
    parser.add_argument('--temporal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fm, fr, fi = read_matrix(args.feedforward)
    tm, tr, ti = read_matrix(args.temporal)
    issues = fi+ti+audit_shared_sources(fm, tm)
    scene_hashes = {}
    for row in fr+tr:
        if row['status'] != 'completed': continue
        if row['scene_seed'] in scene_hashes and scene_hashes[row['scene_seed']] != row['scenario_hash']:
            issues.append('Physical scene mismatch')
        scene_hashes[row['scene_seed']] = row['scenario_hash']
    def group(rows):
        grouped = defaultdict(list)
        for row in rows:
            if row['status'] == 'completed': grouped[row['scene_seed']].append(row)
        return grouped
    f, t = group(fr), group(tr)
    budgets = {side: sorted({r['training_steps'] for r in rows if r['status'] == 'completed'})
               for side, rows in [('feedforward', fr), ('temporal', tr)]}
    seeds = {side: sorted({r['training_seed'] for r in rows if r['status'] == 'completed'})
             for side, rows in [('feedforward', fr), ('temporal', tr)]}
    if not issues:
        for scene in fm['scenes']:
            fs, ts = sorted(r['training_seed'] for r in f[scene]), sorted(r['training_seed'] for r in t[scene])
            if fs != ts or len(fs) != len(set(fs)):
                issues.append('Unmatched/duplicate training-seed repeats')
    metrics = {}
    if not issues:
        metrics = {key: scene_bootstrap([np.mean([r[key] for r in t[scene]])-
            np.mean([r[key] for r in f[scene]]) for scene in fm['scenes']]) for key in METRICS}
    summary = dict(audit_passed=not issues, issues=issues, training_steps=budgets, training_seeds=seeds,
        equal_training_budget=budgets['feedforward']==budgets['temporal'],
        direction='temporal minus feedforward; training-seed repeats averaged within scene',
        metrics=metrics, source_hashes={'feedforward':fm['source_hashes'], 'temporal':tm['source_hashes']},
        raw_hashes={str(p/'attempts.jsonl'):hashlib.sha256((p/'attempts.jsonl').read_bytes()).hexdigest()
                    for p in (args.feedforward,args.temporal)}, diagnostic_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# EXP0049 explicit architecture transfer comparison','',
           f"Audit: {'PASS' if not issues else 'BLOCKED'}.",
           f"Training budgets: feedforward {budgets['feedforward']}; temporal {budgets['temporal']}.", '',
           'Different model/checkpoint hashes are retained. Every common tracked sensing, physics '
           'and conventional-control source must match. This is development evidence only.', '']
    if issues: lines += ['- '+i for i in issues]
    else:
        lines += ['| Metric | Temporal minus feedforward | Scene-bootstrap 95% CI |',
                  '|---|---:|---:|']
        for key,m in metrics.items():
            lines.append(f"| {key} | {m['mean']:+.6f} | [{m['ci95'][0]:+.6f}, {m['ci95'][1]:+.6f}] |")
    lines += ['', 'Comparisons are not evidence of hardware transfer or publication-level novelty. '
              'Secondary intervals are exploratory and not multiplicity-adjusted. '
              'An all-zero paired bootstrap interval does not establish equivalence.', '']
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','raw_hashes')},indent=2))


if __name__ == '__main__': main()
