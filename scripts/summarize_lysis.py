"""Summary tables for benchmark v4 (scripts/benchmark_lysis.py), per N, with paired bootstrap vs a reference.

usage: summarize_lysis.py FILE.jsonl [FILE2.jsonl ...] [--ref METHOD] [--out MD]
Rows are paired by (clusters, anatomy, seed); paired deltas use the scenes both methods completed without error.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [  # key, label, kind
    ('relaxed_success', 'Relaxed %', 'pct'), ('strict_success', 'Strict %', 'pct'),
    ('task_success', 'Cleared %', 'pct'), ('removal', 'Removal %', 'pct'),
    ('t50_s', 'T50 s', 'cens'), ('t90_s', 'T90 s', 'cens'), ('t100_s', 'T100 s', 'cens'),
    ('removal_auc', 'AUC', 'f3'),
    ('wall_contact_s', 'Wall s', 'f2'), ('wall_ge_1s', 'Wall>=1s %', 'pct'),
    ('max_continuous_wall_contact_s', 'MaxWall s', 'f2'),
    ('interference_free', 'NoInterf %', 'pct'), ('spacing_violation_pair_s', 'Viol pair-s', 'f2'),
    ('coupling_exposure', 'Coupling', 'f2'), ('tpg_hold_s', 'Hold s', 'f1'), ('path_mm', 'Path mm', 'f0'),
]


def value(r, k):
    v = r.get(k)
    if k.startswith('t') and k.endswith('_s') and k[1:-2].isdigit():
        return float(r['horizon_s']) if v is None else float(v)        # censored at the horizon
    return float(v) if v is not None else np.nan


def fmt(x, kind):
    if np.isnan(x):
        return '-'
    return {'pct': f'{100*x:.1f}', 'cens': f'{x:.1f}', 'f3': f'{x:.3f}', 'f2': f'{x:.2f}', 'f1': f'{x:.1f}',
            'f0': f'{x:.0f}'}[kind]


def load(files):
    rows = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if 'error' not in r:
                rows.append(r)
    return rows


def boot(d, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(rows, ref=None, cols=COLS):
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r['clusters']][r['method']][(r['anatomy'], r['seed'])] = r
    out = []
    for n in sorted(by):
        out.append(f'\n### N = {n}\n')
        methods = sorted(by[n], key=lambda m: (m != ref, m))
        out.append('| method | eps | ' + ' | '.join(c[1] for c in cols) + ' |')
        out.append('|---|---|' + '---|'*len(cols))
        for m in methods:
            rs = list(by[n][m].values())
            cells = [fmt(np.nanmean([value(r, k) for r in rs]), kind) for k, _, kind in cols]
            out.append(f'| {m} | {len(rs)} | ' + ' | '.join(cells) + ' |')
        if ref and ref in by[n]:
            out.append(f'\nPaired vs {ref} (mean delta [95% bootstrap CI], common scenes):\n')
            keys = [('relaxed_success', 'pct'), ('task_success', 'pct'), ('removal', 'pct'), ('t90_s', 'cens'),
                    ('removal_auc', 'f3'), ('wall_contact_s', 'f2'), ('spacing_violation_pair_s', 'f2')]
            out.append('| method | n | ' + ' | '.join(k for k, _ in keys) + ' |')
            out.append('|---|---|' + '---|'*len(keys))
            for m in methods:
                if m == ref:
                    continue
                common = sorted(set(by[n][m]) & set(by[n][ref]))
                if not common:
                    continue
                cells = []
                for k, kind in keys:
                    d = np.array([value(by[n][m][s], k)-value(by[n][ref][s], k) for s in common])
                    mu, lo, hi = boot(d)
                    sc = 100 if kind == 'pct' else 1
                    cells.append(f'{sc*mu:+.2f} [{sc*lo:+.2f}, {sc*hi:+.2f}]')
                out.append(f'| {m} | {len(common)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+'); p.add_argument('--ref'); p.add_argument('--out', type=Path)
    a = p.parse_args()
    s = table(load(a.files), a.ref)
    print(s)
    if a.out:
        a.out.write_text(s+'\n')


if __name__ == '__main__':
    main()
