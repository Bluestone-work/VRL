"""Route alignment and geodesic progress vs remaining-target count and distance (read-only)."""
import json, sys, glob
import numpy as np
bins_d = [0, .12, .5, 1, 2, 4, 8, 16, 1e9]
acc = {}
start_far = {True: [], False: []}
for f in sorted(glob.glob(sys.argv[1]+'/seed_*/ep_*.npz')):
    z = np.load(f); succ = bool(z['success'])
    mass, geo, act, route, assign, active = (z[k] for k in ('mass', 'geo', 'act', 'route', 'assign', 'active'))
    k = mass.shape[1]; geo = np.where(geo < 0, np.inf, geo)
    alive = (mass > 0).sum(1)
    a = np.clip(assign, 0, k-1)
    ra = np.take_along_axis(route, a[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0]
    ga = np.take_along_axis(geo, a[:, :, None], axis=2)[:, :, 0]
    gn = np.vstack([ga[1:], ga[-1:]])
    cmd = np.linalg.norm(act, axis=2)
    cos = (act*ra).sum(2)/np.maximum(cmd*np.linalg.norm(ra, axis=2), 1e-9)
    prog = (ga-gn)/.1
    same = (np.concatenate([alive[1:], alive[-1:]]) == alive)
    ok = active & (cmd > 1e-9) & (assign >= 0) & np.isfinite(ga) & np.isfinite(gn) & same[:, None]
    for t_, i in zip(*np.nonzero(ok)):
        db = int(np.searchsorted(bins_d, ga[t_, i], side='right')-1)
        key = (int(alive[t_]), db, succ)
        s = acc.setdefault(key, [0, 0., 0., 0.]); s[0] += 1; s[1] += cos[t_, i]; s[2] += prog[t_, i]; s[3] += prog[t_, i] < 0
    # distance of closest robot to last target when it became the last one
    one = np.flatnonzero(alive == 1)
    if len(one):
        j = int(np.flatnonzero(mass[one[0]] > 0)[0])
        start_far[succ].append(float(np.where(active[one[0]], geo[one[0], :, j], np.inf).min()))
out = {}
for (al, db, succ), (c, sc, sp, sr) in sorted(acc.items()):
    out[f'alive{al}|d[{bins_d[db]},{bins_d[db+1]})|{"S" if succ else "F"}'] = dict(n=c, cos=round(sc/c, 3), prog_mm_s=round(sp/c, 3), receding=round(sr/c, 3))
for k_, v in out.items(): print(k_, v)
for s in (True, False):
    v = np.array(start_far[s]); print('success' if s else 'fail', 'closest-robot geo to last target when it became last: n', len(v), 'quantiles', np.percentile(v, [10, 25, 50, 75, 90]).round(2))
json.dump(dict(table=out, last_target_start={str(k): v for k, v in start_far.items()}), open(sys.argv[2], 'w'))
