"""Multi-dimensional benchmark table from benchmark_multicluster JSONL files.

Per method x N: cluster-safe success, raw success, removal, T50/T90/T100 (median over episodes,
censored at the horizon -> reported as '>H' when the median is censored), normalised removal AUC,
total path length (mm) and path per removed clot, wall contact (mean / median / ratio / longest run),
particle contacts, spacing violation pair-seconds, prolonged-yield events, timeout rate.
Means are over anatomies (each anatomy weighted equally); anatomy groups from anatomy_holdout_v1.
usage: summarize_benchmark.py DIR [DIR ...] [--group all|train|held_out] [--per-anatomy]
"""
from __future__ import annotations
import argparse
import collections
import glob
import json

import numpy as np

COLS = [('cluster_safe_success', 'Safe %', 100), ('task_success', 'Raw %', 100), ('removal', 'Removal %', 100),
        ('t50_s', 'T50 s', None), ('t90_s', 'T90 s', None), ('t100_s', 'T100 s', None), ('removal_auc', 'AUC %', 100),
        ('path_mm', 'Path mm', 1), ('path_per_clot', 'Path/clot mm', 1), ('wall_contact_s', 'Wall s', 1),
        ('wall_median', 'Wall med s', 1), ('wall_contact_ratio', 'Wall %', 100), ('max_continuous_wall_contact_s', 'Wall run s', 1),
        ('particle_events', 'Particle ev', 1), ('violation_pair_s', 'Spacing pair-s', 1), ('prolonged_yield_events', 'Deadlock ev', 1),
        ('timeout', 'Timeout %', 100)]


def load(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(f'{d}/*.jsonl'):
            for l in open(f):
                r = json.loads(l)
                r['violation_pair_s'] = r.get('spacing', {}).get('spacing_violation_pair_s', 0.)
                # episodes end when all clots clear: the cleared state persists to the horizon
                r['removal_auc'] = r['removal_auc'] + max(r['horizon_s']-r['elapsed_s'], 0.)/r['horizon_s']*r['removal']
                r['path_per_clot'] = r['path_mm']/max(r['removal']*len(r['plan'] and sum(r['plan'], [])), 1e-9) if r['removal'] > 0 else np.nan
                rows.append(r)
    return rows


def stat(rs, key, horizon):
    v = [r.get(key) for r in rs]
    if key in ('t50_s', 't90_s', 't100_s'):
        x = np.array([horizon+1 if t is None else t for t in v], float)
        med = float(np.median(x))
        return med if med <= horizon else np.inf
    if key == 'wall_median':
        return float(np.median([r['wall_contact_s'] for r in rs]))
    return float(np.nanmean(np.array(v, float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dirs', nargs='+'); p.add_argument('--group', default='all'); p.add_argument('--per-anatomy', action='store_true')
    a = p.parse_args()
    reg = json.load(open('configs/evaluation_splits.json'))
    anat = reg['anatomy_order'] if a.group == 'all' else reg['anatomy_holdout_v1'][a.group]
    rows = [r for r in load(a.dirs) if r['anatomy'] in anat]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['method'], r['clusters'], r['anatomy'])].append(r)
    keys = sorted({(m, n) for m, n, _ in g}, key=lambda x: (x[1], x[0]))
    print(f'### group={a.group}  ({len(anat)} anatomies)\n')
    print('| N | method | info | n | ' + ' | '.join(c[1] for c in COLS) + ' |')
    print('|---|---|---|---:|' + '---:|'*len(COLS))
    for m, n in keys:
        per = [g[(m, n, x)] for x in anat if g[(m, n, x)]]
        H = per[0][0]['horizon_s']
        vals = []
        for key, _, scale in COLS:
            xs = [stat(rs, key, H) for rs in per]
            if key in ('t50_s', 't90_s', 't100_s'):
                med = float(np.median(xs)); vals.append(f'>{H:.0f}' if not np.isfinite(med) else f'{med:.0f}')
            else:
                vals.append(f'{np.nanmean(xs)*scale:.1f}')
        info = per[0][0]['information']
        print(f'| {n} | {m} | {info} | {sum(map(len, per))} | ' + ' | '.join(vals) + ' |')
    if a.per_anatomy:
        print('\n### Safe success per anatomy (%)\n')
        print('| anatomy | ' + ' | '.join(f'{m} N{n}' for m, n in keys) + ' |'); print('|---|' + '---:|'*len(keys))
        for x in anat:
            print(f'| {x} | ' + ' | '.join(f"{100*np.mean([r['cluster_safe_success'] for r in g[(m, n, x)]]):.0f}" if g[(m, n, x)] else '—' for m, n in keys) + ' |')


if __name__ == '__main__':
    main()
