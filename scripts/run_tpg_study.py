"""Process-isolated EXP0053 preflight, fixed-budget training and paired pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.tpg_episode import ROOT,PROTOCOL,tpg_hashes
from scripts.run_option_learning import atomic_json


def run_job(job,out,source):
    folder=out/job['name'];folder.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m']+job['args']
    environment=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    record=dict(name=job['name'],command=command,started_at=datetime.now().astimezone().isoformat())
    atomic_json(folder/'requested.json',record);start=time.monotonic()
    with (folder/'stdout.log').open('x') as stream:
        process=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    record.update(returncode=process.returncode,wall_s=time.monotonic()-start,source_unchanged=source==tpg_hashes())
    atomic_json(folder/'completion.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','pilot'),required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();p=json.loads(PROTOCOL.read_text());source=tpg_hashes()
    args.out.mkdir(parents=True,exist_ok=False)
    snapshots=args.out/'source_snapshot'
    for filename in source:
        target=snapshots/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/filename).read_bytes())
    atomic_json(args.out/'manifest.json',dict(protocol=p,source_hashes=source,phase=args.phase,confirmation_accessed=False))
    attempts=[]
    def batch(jobs,phase):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(run_job,j,args.out,source) for j in jobs]
            for future in as_completed(futures):
                record=future.result();attempts.append(record)
                with (args.out/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                atomic_json(args.out/'status.json',dict(phase=phase,completed=len(attempts),failures=sum(r['returncode']!=0 or not r['source_unchanged'] for r in attempts)))
                print(json.dumps(record),flush=True)
        if any(r['returncode']!=0 or not r['source_unchanged'] for r in attempts):
            atomic_json(args.out/'status.json',dict(phase='failed_gate',attempts=len(attempts),ranking_admitted=False));return False
        return True
    def evaluation(name,seed,policy,variant='both',checkpoint=None,clusters=3):
        command=['scripts.run_tpg_learning','evaluate','--scene',str(seed),'--policy',policy,
            '--variant',variant,'--clusters',str(clusters),'--duration',str(p['duration_s']),
            '--out',str(args.out/'results'/f'{name}.jsonl')]
        if checkpoint:command+=['--checkpoint',str(checkpoint)]
        return dict(name=name,args=command)
    if args.phase=='preflight':
        jobs=[evaluation(f'{policy}_{seed}',seed,policy) for seed in range(p['preflight_scene_base'],p['preflight_scene_base']+p['preflight_scenes']) for policy in ('tpg','untrained')]
        passed=batch(jobs,'preflight')
        if passed:
            pairs=[]
            for seed in range(p['preflight_scene_base'],p['preflight_scene_base']+p['preflight_scenes']):
                a,b=[json.loads((args.out/'results'/f'{policy}_{seed}.jsonl').read_text()) for policy in ('tpg','untrained')]
                pairs.append(dict(scene=seed,initial_match=a['actual_initial_snapshot_hash']==b['actual_initial_snapshot_hash'],
                    final_match=a['final_state_hash']==b['final_state_hash']))
            passed=all(x['initial_match'] and x['final_match'] for x in pairs)
            atomic_json(args.out/'untrained_identity.json',dict(pairs=pairs,passed=passed))
        atomic_json(args.out/'status.json',dict(phase='completed' if passed else 'failed_gate',attempts=len(attempts),ranking_admitted=False))
        return
    train_jobs=[];checkpoints={}
    for variant in p['variants']:
        for index,seed in enumerate(p['training_seeds']):
            name=f'train_{variant}_{seed}';folder=args.out/'training'/name
            checkpoints[variant,seed]=folder/f"policy_{p['pilot_steps_per_seed']}.pt"
            train_jobs.append(dict(name=name,args=['scripts.run_tpg_learning','train','--variant',variant,'--seed',str(seed),
                '--steps',str(p['pilot_steps_per_seed']),'--scene-base',str(p['training_scene_base']+index*p['training_scene_stride']),
                '--out',str(folder)]))
    if not batch(train_jobs,'training'):return
    evaluations=[]
    for scene in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']):
        for variant in p['variants']:
            for seed in p['training_seeds']:
                evaluations.append(evaluation(f'{variant}_{seed}_{scene}',scene,'learned',variant,checkpoints[variant,seed]))
        for policy in ('tpg','untrained'):evaluations.append(evaluation(f'{policy}_{scene}',scene,policy))
        # Original strong comparator retains its own frozen control mechanics.
        for label,policy,n in (('balanced_1s','balanced',3),('memory_n1','memory',1)):
            name=f'{label}_{scene}'
            evaluations.append(dict(name=name,args=['scripts.run_aligned_options','evaluate','--policy',policy,
                '--variant','hierarchical','--decision-steps','10','--clusters',str(n),'--scene-base',str(scene),
                '--episodes','1','--duration',str(p['duration_s']),'--out',str(args.out/'results'/f'{name}.jsonl')]))
    if not batch(evaluations,'validation'):return
    atomic_json(args.out/'status.json',dict(phase='completed',attempts=len(attempts),training_runs=len(train_jobs),
        evaluations=len(evaluations),source_unchanged=source==tpg_hashes(),confirmation_accessed=False))


if __name__=='__main__':main()
