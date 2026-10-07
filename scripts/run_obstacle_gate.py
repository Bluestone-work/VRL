"""Paired development gate for the privileged obstacle lookahead teacher.
Only anatomy_holdout_v1 train+held_out scenes and non-sealed seeds are used.
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp
from pathlib import Path
from scripts.benchmark_obstacles import run

SPLIT = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']
ANATOMIES = SPLIT['train'] + SPLIT['held_out']

def job(x):
    anatomy, seed, n, horizon = x
    out = {'anatomy': anatomy, 'seed': seed, 'clusters': n}
    try:
        base = run('rule_apf', n, anatomy, seed, horizon=horizon)
        teach = run('teacher', n, anatomy, seed, horizon=horizon)
        out.update(apf=base, teacher=teach, error=None)
    except Exception as e:
        out['error'] = repr(e)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12); p.add_argument('--first-seed',type=int,default=2600000000)
    p.add_argument('--seeds',type=int,default=6); p.add_argument('--clusters',type=int,default=1)
    p.add_argument('--horizon',type=float,default=300.); a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[(an,a.first_seed+k,a.clusters,a.horizon) for an in ANATOMIES for k in range(a.seeds)]
    done=set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            try:
                r=json.loads(line); done.add((r['anatomy'],r['seed'],r['clusters']))
            except Exception: pass
    jobs=[j for j in jobs if j[:3] not in done]
    with mp.get_context('fork').Pool(a.workers) as pool, a.out.open('a') as f:
        for r in pool.imap_unordered(job,jobs,chunksize=1):
            f.write(json.dumps(r,default=str)+'\n'); f.flush(); print(json.dumps({'key':(r['anatomy'],r['seed']), 'error':r.get('error')}),flush=True)

if __name__=='__main__': main()
