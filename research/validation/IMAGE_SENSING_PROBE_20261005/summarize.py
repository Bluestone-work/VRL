import json, numpy as np, collections
R=[json.loads(l) for l in open('research/validation/IMAGE_SENSING_PROBE_20261005/rows.jsonl')]
err=[r for r in R if 'error' in r]; R=[r for r in R if 'error' not in r]
by={(r['anatomy'],r['seed'],r['clusters'],r['sensing_model']):r for r in R}
print('rows',len(R),'errors',len(err))
for n in (1,3):
    keys=[(a,s,c) for (a,s,c,m) in by if c==n and m=='noise' and (a,s,c,'image') in by]
    if not keys: continue
    f=lambda m,k: np.array([float(by[x+(m,)][k] or 0) for x in keys])
    for k in ('cluster_safe_success','task_success','removal_auc','particle_events','wall_contact_s','lost'):
        a,b=f('noise',k),f('image',k); d=b-a
        bs=[np.random.default_rng(i).choice(d,len(d)).mean() for i in range(2000)]
        print(f'N={n} pairs={len(keys)} {k:22s} noise={a.mean():.3f} image={b.mean():.3f} diff={d.mean():+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]')
    pe=[by[x+('image',)]['perception'] for x in keys]
    print(f'  perception err mean {np.mean([p["err_mean_mm"] for p in pe]):.3f} p95 {np.mean([p["err_p95_mm"] for p in pe]):.3f} max {max(p["err_max_mm"] for p in pe):.3f} mm; wall image {np.mean([by[x+("image",)]["walltime_s"] for x in keys]):.0f}s')
