"""Paired quick check of every pilot checkpoint vs rule_apf on 84 dev episodes (14 anatomies x seeds 0-2 x N=1,3)."""
import glob, json, os, sys; sys.path.insert(0, '.')
import numpy as np
from multiprocessing import Pool
S = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']; ANAT = S['train']+S['held_out']
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_obstacles as bo
    ck, an, k, n = a
    r = bo.run('drl', n, an, 2600000000+k*100000, ckpt=ck)
    return ck, (an, r['seed'], n), r
if __name__ == '__main__':
    cks = sorted(glob.glob(os.path.dirname(__file__)+'/*_s[0-9]/policy.pt'))
    J = [(c, an, k, n) for c in cks for k in range(3) for an in ANAT for n in (1, 3)]
    with Pool(22) as p: out = list(p.imap_unordered(job, J, chunksize=2))
    B = {(r['anatomy'], r['seed'], r['clusters']): r for r in map(json.loads, open('research/validation/OBST_BENCH_20261006/rule_apf.jsonl'))}
    for c in cks:
        Q = {k: r for cc, k, r in out if cc == c}; k = sorted(set(Q) & set(B))
        print(c, 'pairs', len(k))
        for name, f in [('safe', lambda r: r['cluster_safe_success']), ('obstacle-hit ep', lambda r: r['obstacle_events'] > 0),
                        ('wall>=1s', lambda r: r['wall_contact_s'] >= 1), ('task', lambda r: r['task_success'])]:
            a = np.array([float(f(Q[x])) for x in k]); b = np.array([float(f(B[x])) for x in k]); d = a-b
            bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(2000)]
            print(f'  {name:16s} DRL {a.mean():.3f} APF {b.mean():.3f} diff {d.mean():+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]')
