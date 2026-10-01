"""Cosine of executed action with candidate directions the observation exposes (read-only):
route bearing to own assigned target (slot-indexed in obs 76:108), straight-line bearing to own target,
straight-line bearing to geodesically nearest target (fixed obs 16:19), route bearing to nearest target, local flow."""
import json, sys, glob
import numpy as np
def unit(v): return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
acc = {}
for f in sorted(glob.glob(sys.argv[1]+'/seed_*/ep_*.npz')):
    z = np.load(f); succ = bool(z['success'])
    mass, geo, act, route, assign, active, pos, cp, flow = (z[k] for k in ('mass', 'geo', 'act', 'route', 'assign', 'active', 'pos', 'clot_pos', 'flow'))
    k = mass.shape[1]; geo = np.where(geo < 0, np.inf, geo); alive = (mass > 0).sum(1)
    a = np.clip(assign, 0, k-1); near = np.argmin(geo, axis=2)
    pick = lambda arr, idx: np.take_along_axis(arr, idx[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0]
    ga = np.take_along_axis(geo, a[:, :, None], axis=2)[:, :, 0]
    u = unit(act)
    dirs = dict(route_own=unit(pick(route, a)), straight_own=unit(pick(cp[None, None]-pos[:, :, None], a)),
                straight_nearest=unit(pick(cp[None, None]-pos[:, :, None], near)), route_nearest=unit(pick(route, near)), flow=unit(flow))
    m = active & (assign >= 0) & np.isfinite(ga) & (ga > .7) & (np.linalg.norm(act, axis=2) > 1e-9)
    key = ('1' if 0 else ('last' if True else ''), succ)
    for ph in ('1', '>=2'):
        mm = m & ((alive == 1) if ph == '1' else (alive >= 2))[:, None]
        if not mm.any(): continue
        s = acc.setdefault((ph, succ), {d: [0., 0] for d in dirs})
        for d, v in dirs.items():
            c = (u*v).sum(2)[mm]; s[d][0] += c.sum(); s[d][1] += c.size
out = {f'alive{ph}|{"S" if sc else "F"}': {d: round(v[0]/v[1], 3) for d, v in s.items()} for (ph, sc), s in sorted(acc.items())}
for k_, v in out.items(): print(k_, v)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
