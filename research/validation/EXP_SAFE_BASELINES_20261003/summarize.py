"""Safe-navigation table: privileged and fair controllers reported separately (never ranked together)."""
import collections, glob, json, re, sys
import numpy as np
sys.path.insert(0, '.')
from scripts.safe_metrics import aggregate
E = 'research/validation/EXP_SAFE_BASELINES_20261003'
reg = json.load(open('configs/evaluation_splits.json')); order = reg['anatomy_order']; hold = reg['anatomy_holdout_v1']
rows = collections.defaultdict(list); info = {}
for f in glob.glob(f'{E}/diag_*.jsonl'):
    for l in open(f):
        r = json.loads(l); m = re.sub(r'_s4[234]$', '', r['policy'])
        rows[(m, r['policy'], r['anatomy'])].append(r); info[m] = r['information']
methods = sorted({k[0] for k in rows}, key=lambda m: (info[m] != 'privileged', m))
def method_stats(m, anatomies):
    """Mean over anatomies of per-anatomy rates; seed range of that mean for learned methods."""
    seeds = sorted({k[1] for k in rows if k[0] == m})
    per_seed = []
    for s in seeds:
        per_a = [aggregate(rows[(m, s, a)]) for a in anatomies if rows[(m, s, a)]]
        per_seed.append({k: np.mean([x[k] for x in per_a]) for k in per_a[0]})
    mean = {k: np.mean([p[k] for p in per_seed]) for k in per_seed[0]}
    rng = (min(p['safe_success'] for p in per_seed), max(p['safe_success'] for p in per_seed)) if len(per_seed) > 1 else None
    return mean, rng
cols = [('raw_success', 'Raw', 100), ('safe_success', 'Safe', 100), ('unsafe_success_gap', 'Gap', 100),
        ('mean_wall_contact_s', 'wall mean s', 1), ('median_wall_contact_s', 'wall median s', 1),
        ('wall_contact_ratio', 'wall ratio %', 100), ('max_continuous_wall_contact_s', 'max run s', 1),
        ('timeout_rate', 'timeout %', 100), ('removal', 'removal %', 100)]
for title, anat in (('All 14 anatomies', order), ('Train anatomies (anatomy_holdout_v1)', hold['train']), ('Held-out anatomies', hold['held_out'])):
    print(f'\n### {title}\n'); print('| method | info | ' + ' | '.join(c[1] for c in cols) + ' | route cos | wrong half | Safe seed range |')
    print('|---|---|' + '---:|'*(len(cols)+3))
    for m in methods:
        st, rng = method_stats(m, anat)
        rc = np.nanmean([np.mean([r['route_cos'] for r in rows[k] if r['route_cos'] is not None] or [np.nan]) for k in rows if k[0] == m and k[2] in anat])
        wh = np.nanmean([np.mean([r['wrong_half_space'] for r in rows[k] if r['wrong_half_space'] is not None] or [np.nan]) for k in rows if k[0] == m and k[2] in anat])
        print(f'| {m} | {info[m]} | ' + ' | '.join(f'{st[c]*s:.1f}' for c, _, s in cols) +
              f' | {rc:.2f} | {100*wh:.0f}% | ' + (f'{100*rng[0]:.1f}–{100*rng[1]:.1f}' if rng else '—') + ' |')
print('\n### Safe success per anatomy (%; learned methods: seed mean)\n')
print('| anatomy | ' + ' | '.join(methods) + ' |'); print('|---|' + '---:|'*len(methods))
for a in order:
    print(f'| {a} | ' + ' | '.join(f"{100*np.mean([aggregate(rows[k])['safe_success'] for k in rows if k[0]==m and k[2]==a]):.0f}" for m in methods) + ' |')
