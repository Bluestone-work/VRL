"""Evaluate every trained v3 run (research/runs/OBST_DRL_20261006/<variant>_s<k>/policy.pt) on the dev scenes
(14 anatomies x 30 seeds x N=1/2/3), plus T-IRPPO with image-based cluster tracking (robustness), in one pool.
Resumable: rows already in <tag>.jsonl are skipped."""
import glob, json, os, sys
sys.path.insert(0, '.')
from multiprocessing import Pool
S = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']; ANAT = S['train']+S['held_out']
D = 'research/validation/OBST_BENCH_20261006'
def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.benchmark_obstacles as bo
    tag, ck, sens, an, k, n = a
    try:
        r = bo.run('drl', n, an, 2600000000+k*100000, ckpt=ck, sensing=sens); r['method'] = tag
    except Exception as e:
        r = dict(method=tag, anatomy=an, seed=2600000000+k*100000, clusters=n, error=repr(e))
    return tag, json.dumps(r, default=str)
if __name__ == '__main__':
    runs = sorted(glob.glob('research/runs/OBST_DRL_20261006/*_s[0-9]/policy.pt'))
    specs = [(os.path.basename(os.path.dirname(c)), c, 'noise') for c in runs]
    specs += [(os.path.basename(os.path.dirname(c))+'_img', c, 'image') for c in runs if '/tres_s' in c]
    jobs = []
    for tag, ck, sens in specs:
        done = set()
        if os.path.exists(f'{D}/{tag}.jsonl'):
            for l in open(f'{D}/{tag}.jsonl'):
                r = json.loads(l); done.add((r['anatomy'], r.get('seed'), r['clusters']))
        jobs += [(tag, ck, sens, an, k, n) for k in range(30) for an in ANAT for n in (1, 2, 3)
                 if (an, 2600000000+k*100000, n) not in done]
    print(len(specs), 'runs', len(jobs), 'episodes', flush=True)
    files = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 22) as p:
        for tag, row in p.imap_unordered(job, jobs, chunksize=2):
            f = files.setdefault(tag, open(f'{D}/{tag}.jsonl', 'a')); f.write(row+'\n'); f.flush()
