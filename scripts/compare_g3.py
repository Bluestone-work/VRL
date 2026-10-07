"""Paired comparison of G3 (image sensing) result files against rule_switch and APF under image sensing."""
from __future__ import annotations
import collections, json, sys
import numpy as np

G = 'research/validation/OBST_BENCH_20261006/gates/'
MET = ('cluster_safe_success', 'task_success', 'removal_auc', 'obstacle_events', 'wall_contact_s')


def rd(f):
    rows = [json.loads(x) for x in open(G+f)]
    return {(r['anatomy'], r['seed']): r['row'] for r in rows if not r.get('error')}, sum(1 for r in rows if r.get('error'))


def main(files):
    refs = {'rule_switch(img)': rd('G3img_rule_switch.jsonl')[0], 'APF(img)': rd('G3img_rule_apf.jsonl')[0]}
    rng = np.random.default_rng(20261007)
    for f in files:
        S, err = rd(f); ks = sorted(S); print(f'### {f}  n={len(ks)} errors={err}  {dict(collections.Counter(S[k]["termination_reason"] for k in ks))}')
        for nm, R in refs.items():
            assert all(S[k]['scenario_hash'] == R[k]['scenario_hash'] for k in ks)
            print(f'  vs {nm}')
            for m in MET:
                x = np.array([float(S[k][m] or 0) for k in ks]); y = np.array([float(R[k][m] or 0) for k in ks]); d = x-y
                b = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
                print(f'    {m:22s} {x.mean():7.3f} vs {y.mean():7.3f}  diff {d.mean():+.3f} [{np.quantile(b, .025):+.3f},{np.quantile(b, .975):+.3f}]')


if __name__ == '__main__':
    main(sys.argv[1:])
