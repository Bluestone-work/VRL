"""Aggregate a scheduled-policy evaluation against the paired SwitchSettle rule."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--eval', type=Path, required=True)
    ap.add_argument('--rule', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    learner = load(a.eval/'episodes.jsonl'); rule = [r for r in load(a.rule/'episodes.jsonl') if r['method'] == 'switch_settle_015']
    key = lambda r: (r['panel'], r['split'], r['suite'], r['anatomy'], r['clusters'], r['seed'])
    L, R = {key(r): r for r in learner}, {key(r): r for r in rule}; common = sorted(set(L)&set(R))
    metrics = ('task_success', 'strict_success', 'removal', 'removal_auc', 'T90_300', 'wall_contact_s')
    by = defaultdict(list)
    for k in common: by[k[0]].append(k)
    tables = []; lines = ['# Scheduled-Settle paired evaluation', '', 'Comparison uses the same scenes and `switch_settle_015` rule.']
    for panel, keys in by.items():
        lines += ['', f'## {panel}', '', '| method | n | task % | strict % | removal % | T90_300 s | wall s | Δ strict |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        rule_strict = float(np.mean([R[k]['strict_success'] for k in keys]))
        for name, data in [('SwitchSettle', R), ('FlowResidual', L)]:
            rr = [data[k] for k in keys]; vals = {m: float(np.mean([r[m] for r in rr])) for m in metrics}
            delta = '—' if name == 'SwitchSettle' else f'{100*(vals["strict_success"]-rule_strict):+.1f}'
            lines.append(f"| {name} | {len(rr)} | {100*vals['task_success']:.1f} | {100*vals['strict_success']:.1f} | {100*vals['removal']:.1f} | {vals['T90_300']:.1f} | {vals['wall_contact_s']:.2f} | {delta} |")
            tables.append(dict(panel=panel, method=name, n=len(rr), metrics=vals))
    overall = dict(n=len(common), learner_strict=float(np.mean([L[k]['strict_success'] for k in common])), rule_strict=float(np.mean([R[k]['strict_success'] for k in common])))
    overall['delta_strict_pp'] = 100*(overall['learner_strict']-overall['rule_strict'])
    lines += ['', '## Overall', '', json.dumps(overall, indent=2)]
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n'); (a.out/'aggregate.json').write_text(json.dumps(dict(overall=overall, tables=tables), indent=2))


if __name__ == '__main__':
    main()
