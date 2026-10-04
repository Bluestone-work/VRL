"""Run one complete, prospective frozen-policy matrix under the clean runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.run_clean_residual_eval import ROOT, protocol, source_hashes, enforce_runtime, checked_checkpoint
from scripts.run_option_learning import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();args.out=args.out.resolve()
    runtime=enforce_runtime();p=protocol();source=source_hashes()
    args.out.mkdir(parents=True,exist_ok=False)
    for name in source:
        destination=args.out/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/name).read_bytes())
    parents=[]
    for variant in p['variants']:
        for seed in p['training_seeds']:
            payload,path,digest=checked_checkpoint(variant,seed)
            parents.append(dict(variant=variant,seed=seed,path=str(path),sha256=digest,runtime=payload['runtime']))
    atomic_json(args.out/'manifest.json',dict(protocol=p,source_hashes=source,runtime=runtime,
        parents=parents,new_training_steps=0,confirmation_accessed=False,
        created_at=datetime.now().astimezone().isoformat()))
    environment=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    testcmd=[sys.executable,'-m','pytest','-q','tests/test_conservative_residual.py','tests/test_measured_marl.py',
             'tests/test_safety_marl.py','tests/test_measured_tpg.py','tests/test_command_aligned_tracking.py']
    with (args.out/'test_admission.log').open('x') as stream:
        result=subprocess.run(testcmd,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    atomic_json(args.out/'test_admission.json',dict(command=testcmd,returncode=result.returncode))
    if result.returncode:
        atomic_json(args.out/'status.json',dict(phase='failed_test_admission',ranking_admitted=False));raise SystemExit(2)
    attempts=[]

    def job(name,command):return dict(name=name,command=[sys.executable,'-m']+command)

    def run(j):
        folder=args.out/'jobs'/j['name'];folder.mkdir(parents=True,exist_ok=False)
        rec=dict(**j,requested_at=datetime.now().astimezone().isoformat())
        atomic_json(folder/'requested.json',rec);start=time.monotonic()
        with (folder/'stdout.log').open('x') as stream:
            process=subprocess.run(j['command'],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        rec.update(returncode=process.returncode,wall_s=time.monotonic()-start,source_unchanged=source==source_hashes())
        atomic_json(folder/'completion.json',rec);return rec

    def batch(jobs,phase):
        atomic_json(args.out/f'{phase}_requested.json',jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run,j) for j in jobs]):
                record=future.result();attempts.append(record)
                with (args.out/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                atomic_json(args.out/'status.json',dict(phase=phase,completed_attempts=len(attempts),
                    failures=sum(r['returncode']!=0 or not r['source_unchanged'] for r in attempts)))
                print(json.dumps(dict(name=record['name'],returncode=record['returncode'])),flush=True)
        if any(r['returncode']!=0 or not r['source_unchanged'] for r in attempts):
            atomic_json(args.out/'status.json',dict(phase='failed_gate',attempts=len(attempts),ranking_admitted=False))
            raise SystemExit(2)

    def evaljob(label,scene,variant='r_mappo',seed=42,trained=False,rule=False,phase='results'):
        command=['scripts.run_clean_residual_eval','--variant',variant,'--seed',str(seed),'--scene',str(scene),
                 '--out',str(args.out/phase/f'{label}.jsonl')]
        if trained:command.append('--trained')
        if rule:command.append('--rule')
        return job(label,command)

    preflight=[]
    for scene in range(p['preflight_scene_base'],p['preflight_scene_base']+p['preflight_scenes']):
        preflight.append(evaljob(f'pre_tpg_{scene}',scene,rule=True,phase='preflight'))
        for variant in p['variants']:
            preflight.append(evaljob(f'pre_{variant}_{scene}',scene,variant,phase='preflight'))
    batch(preflight,'preflight')
    checks=[]
    for scene in range(p['preflight_scene_base'],p['preflight_scene_base']+p['preflight_scenes']):
        ref=json.loads((args.out/'preflight'/f'pre_tpg_{scene}.jsonl').read_text())
        for variant in p['variants']:
            other=json.loads((args.out/'preflight'/f'pre_{variant}_{scene}.jsonl').read_text())
            checks.append(dict(scene=scene,variant=variant,
                initial=ref['actual_initial_snapshot_hash']==other['actual_initial_snapshot_hash'],
                scenario=ref['scenario_hash']==other['scenario_hash'],final=ref['final_state_hash']==other['final_state_hash']))
    passed=all(c['initial'] and c['scenario'] and c['final'] for c in checks)
    atomic_json(args.out/'preflight_audit.json',dict(passed=passed,checks=checks))
    if not passed:
        atomic_json(args.out/'status.json',dict(phase='failed_identity',ranking_admitted=False));raise SystemExit(2)
    evaluations=[]
    for scene in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']):
        for variant in p['variants']:
            for seed in p['training_seeds']:
                evaluations.append(evaljob(f'{variant}_{seed}_{scene}',scene,variant,seed,trained=True))
        evaluations.append(evaljob(f'tpg_{scene}',scene,rule=True))
        for label,policy,n in (('balanced_1s','balanced',3),('memory_n1','memory',1)):
            name=f'{label}_{scene}'
            evaluations.append(job(name,['scripts.run_aligned_options','evaluate','--policy',policy,
                '--variant','hierarchical','--decision-steps','10','--clusters',str(n),'--scene-base',str(scene),
                '--episodes','1','--duration',str(p['duration_s']),'--out',str(args.out/'results'/f'{name}.jsonl')]))
    batch(evaluations,'evaluation')
    atomic_json(args.out/'status.json',dict(phase='completed',evaluations=len(evaluations),attempts=len(attempts),
        source_unchanged=source==source_hashes(),new_training_steps=0,confirmation_accessed=False))
    result=subprocess.run([sys.executable,'-m','scripts.analyze_clean_residual',str(args.out)],cwd=ROOT,env=environment)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
