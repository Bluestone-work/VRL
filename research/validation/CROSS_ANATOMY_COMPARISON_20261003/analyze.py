"""Build REPORT.md and summary.json from the CROSS_ANATOMY_20261003 sealed-ledger entries."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NAMES = {'pure_rl': 'Pure RL (EXP40)', 'route_prior': 'Route only', 'route_avoid': 'Route + avoid',
         'route_avoid_wait': 'Route + avoid + wait (strongest traditional)', 'residual': 'Residual RL (EXP43)',
         'residual_shield': 'Residual RL + shield'}
KIND = {'pure_rl': 'pure RL', 'route_prior': 'traditional', 'route_avoid': 'traditional', 'route_avoid_wait': 'traditional',
        'residual': 'residual RL', 'residual_shield': 'residual RL'}
ledger = json.loads((ROOT/'research/sealed_test/LEDGER.json').read_text())
anat = json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_order']
eps = defaultdict(dict)  # (method, seed) -> anatomy -> episodes
for e in ledger['entries']:
    if e['study'] != 'CROSS_ANATOMY_20261003' or not e['complete']:
        continue
    m, rest = e['label'].split('_seed_')
    s, a = rest.split('_', 1)
    eps[(m, int(s))][a] = json.loads(Path(e['result']).read_text())['episodes']
rate = lambda ep, k: float(np.mean([x[k] for x in ep]))
vec = lambda ep, k: np.array([x[k] for x in ep], float)
methods = [m for m in NAMES if any(k[0] == m for k in eps)]
seeds = {m: sorted(s for mm, s in eps if mm == m) for m in methods}
summary = {}
for m in methods:
    per = {}
    for a in anat:
        rows = [eps[(m, s)][a] for s in seeds[m] if a in eps[(m, s)]]
        per[a] = dict(complete=[rate(r, 'success') for r in rows], collision_free=[rate(r, 'collision_free_success') for r in rows],
                      removal=[rate(r, 'removal_fraction') if 'removal_fraction' in r[0] else None for r in rows])
    summary[m] = per
lines = ['# Cross-anatomy sealed comparison: pure RL vs traditional vs residual RL', '',
         'All 14 anatomies, 500 registered sealed-test layouts each, one evaluation per (method, seed, anatomy) in study CROSS_ANATOMY_20261003. '
         'Every method was trained, tuned and selected on MCA only; its checkpoint was fixed before this run, so the other 13 anatomies are zero-shot for all methods. '
         'Learned methods: 3 seeds each (the MCA-validation-selected checkpoint per seed); traditional controllers are deterministic (1 run). Values are complete-clearance / collision-free-clearance percent; learned methods show the seed mean.', '']
def cell(m, a, k):
    v = summary[m][a][k]
    return f'{100*np.mean(v):.1f}' if v else '—'
lines += ['## Overall (mean over 14 anatomies)', '', '| method | type | complete | collision-free | worst anatomy (complete) | seed range of 14-anatomy mean (complete) |', '|---|---|---:|---:|---|---|']
for m in methods:
    comp = [np.mean(summary[m][a]['complete']) for a in anat]
    cf = [np.mean(summary[m][a]['collision_free']) for a in anat]
    w = int(np.argmin(comp))
    per_seed = [np.mean([summary[m][a]['complete'][i] for a in anat]) for i in range(len(seeds[m]))]
    rng = f'{100*min(per_seed):.1f}–{100*max(per_seed):.1f}' if len(per_seed) > 1 else '— (deterministic)'
    lines.append(f'| {NAMES[m]} | {KIND[m]} | {100*np.mean(comp):.1f} | {100*np.mean(cf):.1f} | {anat[w]} {100*comp[w]:.1f} | {rng} |')
lines += ['', '## Per anatomy (complete / collision-free, %)', '', '| anatomy | ' + ' | '.join(NAMES[m] for m in methods) + ' |', '|---|' + '---:|'*len(methods)]
for a in anat:
    lines.append(f'| {a} | ' + ' | '.join(f'{cell(m, a, "complete")} / {cell(m, a, "collision_free")}' for m in methods) + ' |')
# paired comparisons on identical layouts (seed-matched for learned pairs; vs a deterministic controller every seed is paired against it)
def paired(m1, m2, k):
    out = []
    for i, s in enumerate(seeds[m1]):
        s2 = seeds[m2][i] if len(seeds[m2]) > 1 else seeds[m2][0]
        out.append(np.mean([(vec(eps[(m1, s)][a], k)-vec(eps[(m2, s2)][a], k)).mean() for a in anat]))
    return out
def wins(m1, m2, k):
    w = 0
    for a in anat:
        d = np.mean([vec(eps[(m1, s)][a], k).mean() for s in seeds[m1]]) - np.mean([vec(eps[(m2, s)][a], k).mean() for s in seeds[m2]])
        w += d > 0.005
    l = 0
    for a in anat:
        d = np.mean([vec(eps[(m1, s)][a], k).mean() for s in seeds[m1]]) - np.mean([vec(eps[(m2, s)][a], k).mean() for s in seeds[m2]])
        l += d < -0.005
    return w, l
pairs = [('residual_shield', 'route_avoid_wait'), ('residual', 'route_avoid'), ('residual_shield', 'pure_rl'),
         ('route_avoid_wait', 'pure_rl'), ('residual_shield', 'residual')]
lines += ['', '## Paired differences on identical layouts (14-anatomy mean, pp; per learned seed)', '',
          '| comparison | complete | collision-free | anatomies better / worse (complete, >0.5 pp) |', '|---|---|---|---|']
for a, b in pairs:
    if a in methods and b in methods:
        pc, pf = paired(a, b, 'success'), paired(a, b, 'collision_free_success')
        w, l = wins(a, b, 'success')
        lines.append(f'| {NAMES[a]} − {NAMES[b]} | ' + ' / '.join(f'{100*x:+.1f}' for x in pc) + f' | ' + ' / '.join(f'{100*x:+.1f}' for x in pf) + f' | {w} / {l} |')
lines += ['', 'Caveats: synthetic in-vitro simulation, not calibrated physiology or hardware; residual RL is structured (rule prior + learned residual + rule shield), not pure RL; all tuning on MCA only.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
(OUT/'summary.json').write_text(json.dumps(dict(seeds=seeds, per_anatomy=summary), indent=1)+'\n')
print('\n'.join(lines))
