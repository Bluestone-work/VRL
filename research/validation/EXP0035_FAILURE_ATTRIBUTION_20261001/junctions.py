"""Branch decisions at junctions and where time is spent (read-only). Tree: 0 root -> 1,2,3,4; 1 -> 5."""
import json, sys, glob
import numpy as np
parent = {0: -1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
def anc(b):
    out = [b]
    while parent[out[-1]] >= 0: out.append(parent[out[-1]])
    return out
def path(a, b):
    A, B = anc(a), anc(b); lca = next(x for x in A if x in B)
    return set(A[:A.index(lca)+1]) | set(B[:B.index(lca)+1])
res = {}
for f in sorted(glob.glob(sys.argv[1]+'/seed_*/ep_*.npz')):
    z = np.load(f); succ = bool(z['success'])
    mass, branch, assign, active, geo = z['mass'], z['branch'], z['assign'], z['active'], np.where(z['geo'] < 0, np.inf, z['geo'])
    cb = z['clot_branch']; alive = (mass > 0).sum(1)
    for phase in (4, 3, 2, 1):
        steps = np.flatnonzero(alive == phase)
        if not len(steps): continue
        key = (phase, succ)
        r = res.setdefault(key, dict(turns=0, wrong=0, robot_steps=0, on_target_branch=0, on_path=0, off_path=0, upstream_of_target=0))
        for i in range(branch.shape[1]):
            for s in steps:
                if not active[s, i] or assign[s, i] < 0: continue
                tb = int(cb[assign[s, i]]); b = int(branch[s, i])
                r['robot_steps'] += 1
                if b == tb: r['on_target_branch'] += 1
                elif b in anc(tb): r['upstream_of_target'] += 1
                else: r['off_path'] += 1
                if s+1 < len(branch) and active[s+1, i] and branch[s+1, i] != b and alive[s+1] == phase:
                    r['turns'] += 1
                    if int(branch[s+1, i]) not in path(b, tb): r['wrong'] += 1
out = {}
for (ph, s), r in sorted(res.items()):
    rs = max(r['robot_steps'], 1)
    out[f'alive{ph}|{"S" if s else "F"}'] = dict(robot_steps=r['robot_steps'], frac_on_target_branch=round(r['on_target_branch']/rs, 3),
        frac_upstream=round(r['upstream_of_target']/rs, 3), frac_off_path=round(r['off_path']/rs, 3),
        junction_crossings=r['turns'], wrong_turn_frac=round(r['wrong']/max(r['turns'], 1), 3),
        crossings_per_robot_min=round(r['turns']/rs*600, 2))
for k, v in out.items(): print(k, v)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
