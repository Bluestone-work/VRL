"""Per-layout cross-seed consistency, per-branch residual-target rates, travel direction on target branch (read-only)."""
import json, sys, glob, collections
import numpy as np
R = json.load(open(sys.argv[1]))
by = collections.defaultdict(dict)
for r in R: by[r['seed']][r['train_seed']] = r['success']
cnt = collections.Counter(sum(v.values()) for v in by.values())
print('layouts by #seeds succeeding (0..3):', dict(sorted(cnt.items())))
p = np.mean([r['success'] for r in R]); print('independent-model expectation:', {k: round(100*np.math.comb(3, k)*p**k*(1-p)**(3-k), 1) for k in range(4)})
res = collections.Counter(); tot = collections.Counter(); dirn = collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[2]+'/seed_*/ep_*.npz')):
    z = np.load(f); cb = z['clot_branch']; fm = z['final_mass']
    for j, b in enumerate(cb):
        tot[int(b)] += 1; res[int(b)] += int(fm[j] > 0)
    if not z['success']:
        # direction needed on target branch: is target downstream (distal) or upstream of the robot?
        mass, branch, assign, active, pos, flow = (z[k] for k in ('mass', 'branch', 'assign', 'active', 'pos', 'flow'))
        last = np.flatnonzero((mass > 0).sum(1) == 1)
        for t in last[::10]:
            for i in range(branch.shape[1]):
                j = assign[t, i]
                if j < 0 or not active[t, i] or branch[t, i] != cb[j]: continue
                fl = flow[t, i]; d = z['clot_pos'][j]-pos[t, i]
                if np.linalg.norm(fl) > 1e-6: dirn['downstream' if d@fl > 0 else 'upstream'].append(1)
print('residual-target rate by clot branch:', {b: f'{res[b]}/{tot[b]}={res[b]/tot[b]:.2f}' for b in sorted(tot)})
print('failed last-target, robot on target branch: target lies', {k: len(v) for k, v in dirn.items()})
