"""v3 benchmark on the dev scenes: 14 anatomies x 30 seeds x N=1/2/3 (paired across methods).
usage: eval.py TAG METHOD WORKERS [CKPT] [SENSING]"""
import json, os, sys
sys.path.insert(0, '.')
from multiprocessing import Pool
S = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']; ANAT = S['train']+S['held_out']
TAG, M, W = sys.argv[1], sys.argv[2], int(sys.argv[3]); CK = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
SENS = sys.argv[5] if len(sys.argv) > 5 else 'noise'
OUT = f'research/validation/OBST_BENCH_20261006/{TAG}.jsonl'
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_obstacles as bo
    an, k, n = a
    try:
        r = bo.run(M, n, an, 2600000000+k*100000, ckpt=CK, sensing=SENS); r['method'] = TAG
    except Exception as e:
        r = dict(method=TAG, anatomy=an, seed=2600000000+k*100000, clusters=n, error=repr(e))
    return json.dumps(r, default=str)
if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r['anatomy'], r.get('seed'), r['clusters']))
    jobs = [(an, k, n) for k in range(30) for an in ANAT for n in (1, 2, 3) if (an, 2600000000+k*100000, n) not in done]
    with Pool(W) as p, open(OUT, 'a') as f:
        for row in p.imap_unordered(job, jobs, chunksize=2):
            f.write(row+'\n'); f.flush()
