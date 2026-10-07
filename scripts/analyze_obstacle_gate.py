"""Paired comparison of a G2 result file against the G1 APF / teacher rows (bootstrap 95% CI).
usage: analyze_obstacle_gate.py RESULT.jsonl [apf|teacher]"""
from __future__ import annotations
import json, sys
import numpy as np

G1 = 'research/validation/OBST_BENCH_20261006/gates/G1_teacher_vs_apf.jsonl'
KEYS = ('cluster_safe_success', 'task_success', 'removal_auc', 'removal', 'wall_contact_s', 'obstacle_events',
        'obstacle_events_static', 'obstacle_events_dynamic', 'lost', 'path_mm')


def boot(d, rng, B=20000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return np.quantile(m, [.025, .975])


def main(path, ref='apf'):
    g1 = {(r['anatomy'], r['seed']): r for r in map(json.loads, open(G1).read().splitlines())}
    rows = [r for r in map(json.loads, open(path).read().splitlines())]
    err = [r for r in rows if r['error']]; rows = [r for r in rows if not r['error']]
    pairs = []
    for r in rows:
        b = g1[(r['anatomy'], r['seed'])][ref]
        assert b['scenario_hash'] == r['row']['scenario_hash'], 'pairing mismatch'
        pairs.append((r['row'], b))
    rng = np.random.default_rng(20261007)
    out = dict(file=path, ref=ref, n=len(pairs), errors=len(err))
    for k in KEYS:
        x = np.array([[float(s[k] or 0), float(b[k] or 0)] for s, b in pairs]); d = x[:, 0]-x[:, 1]; lo, hi = boot(d, rng)
        out[k] = dict(student=round(x[:, 0].mean(), 4), ref=round(x[:, 1].mean(), 4), diff=round(d.mean(), 4), ci=[round(lo, 4), round(hi, 4)])
    t = [(s['t100_s'], b['t100_s']) for s, b in pairs if s['t100_s'] is not None and b['t100_s'] is not None]
    out['t100_both'] = dict(n=len(t), diff=float(np.mean([u-v for u, v in t])) if t else None)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:3]))
