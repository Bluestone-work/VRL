"""Evaluate the 3 PPO seeds (union physics, deployable obs) on the v2 dev scenes: 14 anatomies x 30 seeds x N=1/2/3."""
import json, sys, os
sys.path.insert(0, '.')
from multiprocessing import Pool
S = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']; ANAT = S['train']+S['held_out']
OUT = 'research/validation/BENCHMARK_V2_UNION_20261005/rl_dep.jsonl'
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_deployable as bd
    from marl.deployable_sensing import DeployableConfig
    sd, an, k, n = a
    bd.RLController.CKPT['path'] = f'research/runs/BENCH_PPO_UNION_20261005/seed{sd}/policy.pt'
    try:
        r = bd.run('rl_dep', n, an, 2600000000+k*100000, 300., 2., DeployableConfig(), 'union', 'noise')
        r['method'] = f'rl_dep_s{sd}'
    except Exception as e:
        r = dict(method=f'rl_dep_s{sd}', anatomy=an, seed_k=k, clusters=n, error=repr(e))
    return json.dumps(r, default=str)
if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r['method'], r['anatomy'], r.get('seed'), r['clusters']))
    jobs = [(sd, an, k, n) for k in range(30) for sd in range(3) for an in ANAT for n in (1, 2, 3)
            if (f'rl_dep_s{sd}', an, 2600000000+k*100000, n) not in done]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 22) as p, open(OUT, 'a') as f:
        for row in p.imap_unordered(job, jobs, chunksize=2):
            f.write(row+'\n'); f.flush()
