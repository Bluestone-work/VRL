"""Route cosine / geodesic progress by robot location relative to its assigned target (read-only)."""
import json, sys, glob
import numpy as np
parent = {0: -1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
def anc(b):
    out = [b]
    while parent[out[-1]] >= 0: out.append(parent[out[-1]])
    return out
acc = {}
for f in sorted(glob.glob(sys.argv[1]+'/seed_*/ep_*.npz')):
    z = np.load(f); succ = bool(z['success'])
    mass, geo, act, route, assign, active, branch, cb = (z[k] for k in ('mass', 'geo', 'act', 'route', 'assign', 'active', 'branch', 'clot_branch'))
    k = mass.shape[1]; geo = np.where(geo < 0, np.inf, geo); alive = (mass > 0).sum(1)
    a = np.clip(assign, 0, k-1)
    ra = np.take_along_axis(route, a[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0]
    ga = np.take_along_axis(geo, a[:, :, None], axis=2)[:, :, 0]; gn = np.vstack([ga[1:], ga[-1:]])
    cmd = np.linalg.norm(act, axis=2); cos = (act*ra).sum(2)/np.maximum(cmd*np.linalg.norm(ra, axis=2), 1e-9)
    same = np.concatenate([alive[1:], alive[-1:]]) == alive
    tb = cb[a]
    for t, i in zip(*np.nonzero(active & (assign >= 0) & np.isfinite(ga) & np.isfinite(gn) & same[:, None] & (ga > .5))):
        b = int(branch[t, i]); T = int(tb[t, i])
        loc = 'target_branch' if b == T else 'upstream' if b in anc(T) else 'off_path'
        key = (int(min(alive[t], 2)), loc, succ)
        s = acc.setdefault(key, [0, 0., 0.]); s[0] += 1; s[1] += cos[t, i]; s[2] += (ga[t, i]-gn[t, i])/.1
out = {f'alive{"1" if al == 1 else ">=2"}|{loc}|{"S" if s else "F"}': dict(n=c, cos=round(sc/c, 3), prog_mm_s=round(sp/c, 3)) for (al, loc, s), (c, sc, sp) in sorted(acc.items())}
for k_, v in out.items(): print(k_, v)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
