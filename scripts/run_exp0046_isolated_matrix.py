"""Predeclared paired diagnostic matrix; live progress, source freeze, no sealed access."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import shutil

from scripts.evaluate_multicluster import source_hashes
from scripts.evaluate_multicluster_isolated import run_isolated
from scripts.multicluster_protocol import REVISION

ROOT=Path(__file__).resolve().parents[1]


def cells():
    result=[]
    for control_seed in (42,43,44):
        result.append(dict(method='single_sequential',clusters=1,d_min_mm=2.,control_seed=control_seed,budget='fixed_total'))
        for n in (2,3):
            for d in (1.,2.,4.):
                result.append(dict(method='multi_parallel',clusters=n,d_min_mm=d,control_seed=control_seed,budget='fixed_total'))
    result.append(dict(method='single_route',clusters=1,d_min_mm=2.,control_seed=42,budget='fixed_total'))
    for n in (2,3):
        result.append(dict(method='multi_unshielded',clusters=n,d_min_mm=2.,control_seed=42,budget='fixed_total'))
        result.append(dict(method='multi_parallel',clusters=n,d_min_mm=2.,control_seed=42,budget='per_cluster'))
    return result


def label(c):
    return f"{c['method']}_n{c['clusters']}_d{c['d_min_mm']:g}_s{c['control_seed']}_{c['budget']}"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed-base',type=int,default=1302010000)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.episodes<1 or args.workers<1: p.error('Positive episode/worker counts required')
    args.out.mkdir(parents=True,exist_ok=False)
    jobs=cells()
    seeds=list(range(args.seed_base,args.seed_base+args.episodes))
    # A diagnostic pool must never overlap any registered evaluation split.
    splits=json.loads((ROOT/'configs/evaluation_splits.json').read_text())
    for anatomy in splits['anatomies'].values():
        for split in anatomy.values():
            if isinstance(split,dict) and 'seed_base' in split:
                assert not any(split['seed_base']<=s<split['seed_base']+split['count'] for s in seeds)
    hashes=source_hashes()
    snapshot=args.out/'source_snapshot'
    for f in hashes:
        dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
    manifest=dict(revision=REVISION,registered_at=datetime.now().astimezone().isoformat(),
                  scenarios=seeds,cells=jobs,requested_episodes=len(jobs)*len(seeds),duration_s=180.,
                  anatomy='mca_m1_lvo',source_hashes=hashes,workers=args.workers,
                  control_seeds_are_not_training_seeds=True,
                  peer_sensing_radius_mm=6.,position_noise_mm=.02,observation_noise=.025,
                  diagnostic_only=True,sealed_test_used=False)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=[]
    def run(c):
        assert source_hashes()==hashes,'Source changed during experiment'
        result=run_isolated(**c,seeds=seeds,out=args.out/(label(c)+'.jsonl'))
        assert source_hashes()==hashes,'Source changed during experiment'
        return dict(label=label(c),completed=result['completed_episodes'],failed=result['failed_episodes'])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks={executor.submit(run,c):c for c in jobs}
        for future in as_completed(tasks):
            c=tasks[future]
            try: status.append(future.result())
            except Exception as e: status.append(dict(label=label(c),error=repr(e)))
            (args.out/'progress.json').write_text(json.dumps(dict(finished_cells=len(status),total_cells=len(jobs),results=status),indent=2)+'\n')
            print(json.dumps(dict(matrix_progress=f'{len(status)}/{len(jobs)}',last=status[-1])),flush=True)
    (args.out/'matrix_status.json').write_text(json.dumps(dict(complete=len(status)==len(jobs),results=status),indent=2)+'\n')
    if any(r.get('error') or r.get('failed',0) for r in status): raise SystemExit(2)


if __name__=='__main__': main()
