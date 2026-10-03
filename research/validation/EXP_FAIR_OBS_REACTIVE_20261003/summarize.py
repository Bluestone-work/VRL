"""Per-anatomy and train / held-out summary of the fair-observation rollouts."""
import collections, glob, json, sys
import numpy as np
E = 'research/validation/EXP_FAIR_OBS_REACTIVE_20261003'
reg = json.load(open('configs/evaluation_splits.json'))
split, order = reg['anatomy_holdout_v1'], reg['anatomy_order']
tags = sys.argv[1:] or ['teacher', 'reactive_bearing', 'reactive_path']
rows = collections.defaultdict(list)
for f in glob.glob(f'{E}/diag_*.jsonl'):
    for l in open(f):
        r = json.loads(l); rows[(r['policy'], r['anatomy'])].append(r)
def m(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return float(np.mean(v)) if v else float('nan')
print('| anatomy | ' + ' | '.join(tags) + ' |'); print('|---|' + '---|'*len(tags))
for a in order:
    print(f'| {a} | ' + ' | '.join(f"{100*m(rows[(t,a)],'success'):.0f} / {100*m(rows[(t,a)],'collision_free'):.0f}" for t in tags) + ' |')
for k in ('success', 'collision_free', 'removal', 'elapsed_s', 'wall_contact_s', 'particle_events', 'route_cos', 'wrong_half_space'):
    vals = [np.nanmean([m(rows[(t, a)], k) for a in order]) for t in tags]
    print(f'| **14-anatomy mean {k}** | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
