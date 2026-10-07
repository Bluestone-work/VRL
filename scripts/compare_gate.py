"""Paired comparison of G2 result files against rule_switch(0.3), APF and teacher v2 (bootstrap 95% CI).
usage: compare_gate.py FILE [FILE ...]  (paths relative to the gates directory or absolute)"""
from __future__ import annotations
import json, os, sys
import numpy as np

G = 'research/validation/OBST_BENCH_20261006/gates/'
MET = ('cluster_safe_success', 'task_success', 'removal_auc', 'obstacle_events', 'obstacle_events_static', 'obstacle_events_dynamic', 'wall_contact_s')


def rd(f):
    f = f if os.path.isabs(f) or f.startswith('research/') else G+f
    return {(r['anatomy'], r['seed']): r['row'] for r in map(json.loads, open(f)) if not r.get('error')}


def main(files):
    g1 = {(r['anatomy'], r['seed']): r for r in map(json.loads, open(G+'G1_teacher_vs_apf.jsonl'))}
    refs = {'rule_switch': rd('G2_rule_switch0.3.jsonl'), 'APF': {k: v['apf'] for k, v in g1.items()}}
    rng = np.random.default_rng(20261007)
    for f in files:
        S = rd(f); ks = sorted(S); print(f'### {f}  n={len(ks)}')
        for nm, R in refs.items():
            assert all(S[k]['scenario_hash'] == R[k]['scenario_hash'] for k in ks), 'pairing mismatch'
            parts = []
            for m in MET:
                x = np.array([float(S[k][m] or 0) for k in ks]); y = np.array([float(R[k][m] or 0) for k in ks]); d = x-y
                b = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
                parts.append(f'{m[:14]:14s} {x.mean():7.3f} vs {y.mean():7.3f}  diff {d.mean():+.3f} [{np.quantile(b, .025):+.3f},{np.quantile(b, .975):+.3f}]')
            print(f'  vs {nm}'); print('    '+'\n    '.join(parts))


if __name__ == '__main__':
    main(sys.argv[1:])
