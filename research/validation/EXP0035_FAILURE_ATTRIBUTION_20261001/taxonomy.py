"""Rule-based failure taxonomy (first matching rule wins), per failed episode (read-only)."""
import json, sys, collections
import numpy as np
R = json.load(open(sys.argv[1])); F = [r for r in R if not r['success']]
def classify(r):
    rem = r['remaining']
    if any(d['touched'] for d in rem): return 'insufficient_contact (contact began, mass left)'
    if any(d['min_geo_ever'] < .12 for d in rem): return 'reached_contact_zone_no_lysis'
    if any(d['min_geo_ever'] < .7 for d in rem): return 'near_miss (<0.7 mm, never in contact)'
    if r['lost'] >= 2 or any((d['dead_wall_frac'] or 0) > .2 for d in rem): return 'collision/stuck (robots lost or wall-bound)'
    if any(not d['ever_on_branch'] for d in rem): return 'never_entered_target_branch (branch selection)'
    return 'wandering_far (entered branch/upstream, net progress ~0)'
out = collections.defaultdict(lambda: collections.Counter())
for r in F:
    phase = 'last_target' if r['n_cleared'] == 3 else 'early (<=1 cleared)' if r['n_cleared'] <= 1 else 'two_left'
    out[phase][classify(r)] += 1; out['ALL'][classify(r)] += 1
for k, v in out.items(): print(k, sum(v.values()), dict(v.most_common()))
osc = [d['dead_path_mm']/max(d['dead_net_mm'], 1e-3) for r in F for d in r['remaining'] if d['dead_path_mm'] > 0]
print('dead-window path/net displacement ratio quantiles', np.percentile(osc, [10, 25, 50, 75, 90]).round(1))
json.dump({k: dict(v) for k, v in out.items()}, open(sys.argv[2], 'w'), indent=1)
