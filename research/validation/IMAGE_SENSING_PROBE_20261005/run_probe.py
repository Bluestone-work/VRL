"""Paired probe: image-based perception chain vs. Gaussian-noise sensing (dev seeds only)."""
import json, sys, os
sys.path.insert(0, '.')
from multiprocessing import Pool
ANAT = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']
ANAT = ANAT['train']+ANAT['held_out']
OUT = 'research/validation/IMAGE_SENSING_PROBE_20261005/rows.jsonl'
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_deployable as bd
    from marl.deployable_sensing import DeployableConfig
    an, k, n, sens = a
    m = 'pursuit_dep' if n == 1 else 'pursuit_tpg_dep'
    try:
        r = bd.run(m, n, an, 2600000000+k*100000, 300., 2., DeployableConfig(), 'union', sens)
    except Exception as e:
        return json.dumps(dict(anatomy=an, seed_k=k, clusters=n, sensing_model=sens, error=repr(e)))
    return json.dumps(r, default=str)
if __name__ == '__main__':
    jobs = [(an, k, n, s) for k in range(5) for an in ANAT for n in (1, 3) for s in ('noise', 'image')]
    with Pool(6) as p, open(OUT, 'a') as f:
        for row in p.imap_unordered(job, jobs):
            f.write(row+'\n'); f.flush()
