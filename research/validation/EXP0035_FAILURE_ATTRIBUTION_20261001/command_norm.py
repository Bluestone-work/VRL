"""Pre-normalisation policy command magnitude vs route alignment (read-only).
Under command_speed='unit' every nonzero proposal executes at full speed, so a near-zero
proposal (the policy 'wanting' to slow down) becomes a full-speed move in an arbitrary direction."""
import json, sys, glob
import numpy as np
from environments.mca_physical_env import bound_command
bins = [0, .1, .2, .3, .5, .7, 1.01, 9]
acc = {}; summ = {}
for f in sorted(glob.glob(sys.argv[1]+'/seed_*/ep_*.npz')):
    z = np.load(f); succ = bool(z['success'])
    mass, geo, act, route, assign, active, raw = (z[k] for k in ('mass', 'geo', 'act', 'route', 'assign', 'active', 'raw'))
    k = mass.shape[1]; geo = np.where(geo < 0, np.inf, geo); alive = (mass > 0).sum(1)
    a = np.clip(assign, 0, k-1)
    ra = np.take_along_axis(route, a[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0]
    ga = np.take_along_axis(geo, a[:, :, None], axis=2)[:, :, 0]
    # 'act' is the world-frame execution before bound_command; its clipped norm is what 'bounded' mode would have run
    pre = np.linalg.norm(np.clip(act, -1, 1), axis=2)
    cos = (act*ra).sum(2)/np.maximum(np.linalg.norm(act, axis=2)*np.linalg.norm(ra, axis=2), 1e-9)
    m = active & (assign >= 0) & np.isfinite(ga) & (ga > .5)
    for t, i in zip(*np.nonzero(m)):
        b = int(np.searchsorted(bins, pre[t, i], side='right')-1)
        key = ('1' if alive[t] == 1 else '>=2', b, succ)
        s = acc.setdefault(key, [0, 0.]); s[0] += 1; s[1] += cos[t, i]
        q = summ.setdefault(('1' if alive[t] == 1 else '>=2', succ), []); q.append(pre[t, i])
for k_, v in sorted(summ.items()):
    v = np.array(v); print('alive', k_[0], 'S' if k_[1] else 'F', 'cmd norm quantiles', np.percentile(v, [10, 25, 50, 75, 90]).round(3), 'frac<0.3', round(float((v < .3).mean()), 3))
out = {}
for (al, b, s), (c, sc) in sorted(acc.items()):
    out[f'alive{al}|norm[{bins[b]},{bins[b+1]})|{"S" if s else "F"}'] = dict(n=c, cos=round(sc/c, 3))
for k_, v in out.items(): print(k_, v)
json.dump(dict(table=out), open(sys.argv[2], 'w'), indent=1)
