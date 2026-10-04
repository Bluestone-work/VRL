"""Paired per-scene differences between two benchmark methods (same anatomy, seed, N), bootstrap 95% CI.
usage: paired_benchmark_stats.py A B DIR [DIR ...] [--metric cluster_safe_success]"""
import argparse, glob, json
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('a'); p.add_argument('b'); p.add_argument('dirs', nargs='+')
p.add_argument('--metric', default='cluster_safe_success'); a = p.parse_args()
reg = json.load(open('configs/evaluation_splits.json')); hold = reg['anatomy_holdout_v1']
rows = {}
for d in a.dirs:
    for f in glob.glob(f'{d}/*.jsonl'):
        for l in open(f):
            r = json.loads(l); rows[(r['method'], r['clusters'], r['anatomy'], r['seed'])] = r
rng = np.random.default_rng(0)
for n in (1, 2, 3):
    for grp, an in (('all', reg['anatomy_order']), ('held_out', hold['held_out'])):
        keys = [(x, s) for (m, nn, x, s) in rows if m == a.a and nn == n and x in an and (a.b, n, x, s) in rows]
        if not keys:
            continue
        val = lambda r: float(r[a.metric] if r[a.metric] is not None else r['horizon_s']+1)
        d = np.array([val(rows[(a.a, n, x, s)])-val(rows[(a.b, n, x, s)]) for x, s in keys])
        # resample anatomies then scenes (two-level) to respect anatomy clustering
        an_list = sorted({x for x, _ in keys}); by = {x: d[[i for i, (y, _) in enumerate(keys) if y == x]] for x in an_list}
        bs = [np.mean([rng.choice(by[x], len(by[x])).mean() for x in rng.choice(an_list, len(an_list))]) for _ in range(2000)]
        print(f'N={n} {grp:8s} {a.a} - {a.b} {a.metric}: {np.mean([by[x].mean() for x in an_list]):+.3f} '
              f'[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  scenes {len(d)} better {int((d > 0).sum())} worse {int((d < 0).sum())}')
