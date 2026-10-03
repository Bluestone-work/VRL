"""Summarize diagnostic rollouts: per anatomy and train / held-out means (anatomy_holdout_v1)."""
import collections, glob, json, sys
import numpy as np
E = 'research/validation/EXP_TEACHER_BC_20261003'
split = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']
order = json.load(open('configs/evaluation_splits.json'))['anatomy_order']
tags = sys.argv[1:]
rows = collections.defaultdict(list)
for t in tags:
    for f in glob.glob(f'{E}/diag_{t}_*.jsonl'):
        for l in open(f):
            r = json.loads(l); rows[(t, r['anatomy'])].append(r)
keys = ('success', 'collision_free', 'removal', 'elapsed_s', 'wall_contact_s', 'particle_events', 'teacher_cos', 'teacher_stop_agreement')
def m(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return float(np.mean(v)) if v else float('nan')
print('| anatomy | set | ' + ' | '.join(tags) + ' |'); print('|---|---|' + '---|'*len(tags))
for a in order:
    s = 'train' if a in split['train'] else 'held-out'
    print(f'| {a} | {s} | ' + ' | '.join(f"{100*m(rows[(t,a)],'success'):.0f} / {100*m(rows[(t,a)],'collision_free'):.0f}" if rows[(t,a)] else '—' for t in tags) + ' |')
for s in ('train', 'held_out'):
    for k in keys:
        vals = []
        for t in tags:
            per = [m(rows[(t, a)], k) for a in split[s] if rows[(t, a)]]
            vals.append(f'{np.nanmean(per):.3f}' if per else '—')
        print(f'| **{s} mean {k}** | | ' + ' | '.join(vals) + ' |')
