"""Run EXP0059 high-level isolation once, retaining all failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from scripts.run_high_isolation_eval import ROOT,protocol,source_hashes,enforce_runtime
from scripts.run_option_learning import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();args.out=args.out.resolve()
    runtime=enforce_runtime();p=protocol();source=source_hashes();args.out.mkdir(parents=True,exist_ok=False)
    for name in source:
        d=args.out/'source_snapshot'/name;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/name).read_bytes())
    atomic_json(args.out/'manifest.json',dict(protocol=p,source_hashes=source,runtime=runtime,confirmation_accessed=False,created_at=datetime.now().astimezone().isoformat()))
    env=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1');attempts=[]
    def job(name,command):return dict(name=name,command=[sys.executable,'-m']+command)
    def run(j):
        folder=args.out/'jobs'/j['name'];folder.mkdir(parents=True,exist_ok=False);rec=dict(**j,requested_at=datetime.now().astimezone().isoformat());atomic_json(folder/'requested.json',rec);start=time.monotonic()
        with (folder/'stdout.log').open('x') as stream:proc=subprocess.run(j['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        rec.update(returncode=proc.returncode,wall_s=time.monotonic()-start,source_unchanged=source==source_hashes());atomic_json(folder/'completion.json',rec);return rec
    def batch(jobs,phase):
        atomic_json(args.out/f'{phase}_requested.json',jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(run,j) for j in jobs]):
                rec=fut.result();attempts.append(rec)
                with (args.out/'attempts.jsonl').open('a') as s:s.write(json.dumps(rec)+'\n')
                atomic_json(args.out/'status.json',dict(phase=phase,completed_attempts=len(attempts),failures=sum(r['returncode']!=0 or not r['source_unchanged'] for r in attempts)))
                print(json.dumps({'name':rec['name'],'returncode':rec['returncode']}),flush=True)
        if any(r['returncode'] or not r['source_unchanged'] for r in attempts):atomic_json(args.out/'status.json',dict(phase='failed_gate',ranking_admitted=False));raise SystemExit(2)
    jobs=[]
    parent=ROOT/p['parent_study']/'training'
    for scene in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']):
        for seed in p['training_seeds']:
            low_paths={v:parent/f'train_{v}_{seed}'/f"policy_{p['training_steps']}.pt" for v in p['low_variants']}
            for low,low_path in low_paths.items():
                for high in p['high_variants']:
                    high_path=parent/f'train_{high}_{seed}'/f"policy_{p['training_steps']}.pt" if high!='rule' else None
                    label=f'{high}_high__{low}_low_{seed}_{scene}'
                    cmd=['scripts.run_high_isolation_eval','--high-variant',high,'--low-variant',low,'--seed',str(seed),'--scene',str(scene),'--low-checkpoint',str(low_path),'--out',str(args.out/'results'/f'{label}.jsonl')]
                    if high_path:cmd += ['--high-checkpoint',str(high_path)]
                    jobs.append(job(label,cmd))
    batch(jobs,'evaluation')
    atomic_json(args.out/'status.json',dict(phase='completed',evaluations=len(jobs),attempts=len(attempts),source_unchanged=source==source_hashes(),confirmation_accessed=False))
    print(json.dumps({'evaluations':len(jobs),'attempts':len(attempts)}))


if __name__=='__main__':main()
