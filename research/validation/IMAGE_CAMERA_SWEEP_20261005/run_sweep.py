"""Imaging-quality sensitivity: pixel size 20/40/60 um and per-episode randomised camera (dev seeds 0-4, 14 anatomies, N=1/3)."""
import json, sys, os
sys.path.insert(0, '.')
from multiprocessing import Pool
S = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']; ANAT = S['train']+S['held_out']
OUT = 'research/validation/IMAGE_CAMERA_SWEEP_20261005/rows.jsonl'
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_deployable as bd
    from marl.deployable_sensing import DeployableConfig
    an, k, n, cam = a
    try:
        r = bd.run('pursuit_dep' if n == 1 else 'pursuit_tpg_dep', n, an, 2600000000+k*100000, 300., 2., DeployableConfig(), 'union', 'image', cam)
        r['camera_setting'] = cam
    except Exception as e:
        r = dict(anatomy=an, seed_k=k, clusters=n, camera_setting=cam, error=repr(e))
    return json.dumps(r, default=str)
if __name__ == '__main__':
    jobs = [(an, k, n, c) for k in range(5) for an in ANAT for n in (1, 3) for c in ('0.04', '0.06', 'random')]
    with Pool(22) as p, open(OUT, 'a') as f:
        for row in p.imap_unordered(job, jobs):
            f.write(row+'\n'); f.flush()
