"""Committed branch decisions: branch visits lasting >= MIN steps (2 mm at 1 mm/s), judged against
the tree path to the robot's assigned target at entry time. Chatter at a junction is ignored. Read-only."""
import json, sys, glob
import numpy as np
MIN = 20
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
    mass, branch, assign, active, cb = z['mass'], z['branch'], z['assign'], z['active'], z['clot_branch']
    alive = (mass > 0).sum(1)
    for i in range(branch.shape[1]):
        b = branch[:, i]; T = int(active[:, i].sum())
        runs = []; s = 0
        for t in range(1, T+1):
            if t == T or b[t] != b[s]:
                if t-s >= MIN: runs.append((s, int(b[s])))
                s = t
        merged = []
        for r in runs:
            if not merged or merged[-1][1] != r[1]: merged.append(r)
        for (s0, b0), (s1, b1) in zip(merged, merged[1:]):
            j = assign[s1, i]
            if j < 0 or mass[s1, j] <= 0: continue
            if assign[s0, i] != j: continue  # target switched in between: not a decision about this target
            key = (int(alive[s1]), succ)
            r = res.setdefault(key, [0, 0])
            r[0] += 1; r[1] += int(b1 not in path(b0, int(cb[j])))
out = {f'alive{a}|{"S" if s else "F"}': dict(committed_decisions=n, wrong=w, wrong_frac=round(w/max(n, 1), 3)) for (a, s), (n, w) in sorted(res.items())}
for k, v in out.items(): print(k, v)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
