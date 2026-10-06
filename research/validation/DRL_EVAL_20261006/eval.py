"""Benchmark the rule pipeline and the IR-PPO controller with IMAGE perception on the v2 dev scenes
(14 anatomies x 30 seeds x N=1/2/3, union physics, default camera). Method tags:
  rule_img        pursuit (N=1) / pursuit+TPG (N>1), image perception           (the rule baseline)
  irppo_s{k}      same pipeline + learned residual, seed k
usage: eval.py METHOD WORKERS [CKPT]"""
import json, os, sys
sys.path.insert(0, '.')
from multiprocessing import Pool
S = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']; ANAT = S['train']+S['held_out']
TAG, W = sys.argv[1], int(sys.argv[2]); CK = sys.argv[3] if len(sys.argv) > 3 else None
OUT = f'research/validation/DRL_EVAL_20261006/{TAG}.jsonl'
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_deployable as bd
    from marl.deployable_sensing import DeployableConfig
    an, k, n = a
    try:
        r = bd.run('pursuit_dep' if n == 1 else 'pursuit_tpg_dep', n, an, 2600000000+k*100000, 300., 2.,
                   DeployableConfig(), 'union', 'image', 'default', drl=CK)
        r['method'] = TAG
    except Exception as e:
        r = dict(method=TAG, anatomy=an, seed_k=k, clusters=n, error=repr(e))
    return json.dumps(r, default=str)
if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r['anatomy'], r.get('seed'), r['clusters']))
    jobs = [(an, k, n) for k in range(30) for an in ANAT for n in (1, 2, 3) if (an, 2600000000+k*100000, n) not in done]
    with Pool(W) as p, open(OUT, 'a') as f:
        for row in p.imap_unordered(job, jobs, chunksize=1):
            f.write(row+'\n'); f.flush()
