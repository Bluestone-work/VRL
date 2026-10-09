"""Finite EXP0060 queue: matched budgets, all seeds, explicit failures, final report.

Stops launching work at the registered deadline, never silently retries a failed
scientific run. Each process has its own stdout/stderr and immutable source copy.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/validation/EXP0060_VASCULAR_OPTION_RL_20261006'
RUNS=ROOT/'research/runs/EXP0060_VASCULAR_OPTION_RL_20261006'
PY='/home/wj/.cache/vascular-research/cpu312-clean-20261004/bin/python'
RENDER_PY='/home/wj/miniconda3/envs/v/bin/python'
DEADLINE=datetime(2026,10,6,8,35,tzinfo=timezone(timedelta(hours=8))).timestamp()
ARMS=['memory_mappo','ff_mappo','memory_ippo']
CPUS=['0-5','8-15','16-23']


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'eval').mkdir(exist_ok=True); (OUT/'jobs').mkdir(exist_ok=True)
    snap=OUT/'source_snapshot'
    if snap.exists():raise RuntimeError('Queue already registered: do not overwrite source or results')
    snap.mkdir()
    for folder in ['scripts','marl','environments','configs']:
        shutil.copytree(ROOT/folder,snap/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()}
    (OUT/'SOURCE_SHA256.json').write_text(json.dumps(hashes,indent=2))
    env=dict(os.environ,PYTHONPATH=str(snap),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    anatomies=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']['train']
    jobs=[]; failures=[]
    def status(stage):
        tmp=OUT/'STATUS.tmp'
        tmp.write_text(json.dumps(dict(pid=os.getpid(),stage=stage,time=datetime.now().isoformat(),
                         deadline='2026-10-06 08:35 Asia/Shanghai',jobs=jobs,failures=failures),indent=2))
        tmp.replace(OUT/'STATUS.json')
    def run(name,args,cpu='0-5,8-23',python=PY,cwd=snap):
        reserve=2400 if len(args)>1 and args[1]=='train' else 0
        left=DEADLINE-time.time()-reserve
        if left<120:return dict(name=name,status='not_started_deadline',returncode=None)
        cmd=['taskset','-c',cpu,python,'-u','-m',*map(str,args)]
        start=time.time()
        with (OUT/'jobs'/f'{name}.log').open('w') as log:
            log.write(json.dumps(dict(command=cmd,cwd=str(cwd),started=datetime.now().isoformat()))+'\n'); log.flush()
            pr=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=pr.wait(timeout=max(1,left-60))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(pr.pid,signal.SIGTERM)
                try:pr.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(pr.pid,signal.SIGKILL); pr.wait()
                return dict(name=name,status='deadline_terminated',returncode=pr.returncode,seconds=time.time()-start)
        return dict(name=name,status='complete' if rc==0 else 'failed',returncode=rc,seconds=time.time()-start)
    def save_result(r):
        jobs.append(r)
        if r['status']!='complete':failures.append(r)
    def report():
        reportenv=dict(env,PYTHONPATH=str(ROOT))
        with (OUT/'jobs'/'report.log').open('a') as log:
            subprocess.run(['taskset','-c','0-5,8-23',PY,'-m','scripts.report_vascular_option_study'],
                           cwd=ROOT,env=reportenv,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    (OUT/'QUEUE_PROTOCOL.json').write_text(json.dumps(dict(arms=ARMS,seeds=[0,1,2],workers=6,updates=480,
                rollout=128,epochs=4,joint_control_steps_budget=480*6*128*5,max_minutes_per_run=140,
                evaluation_anatomies=anatomies,scenes_per_anatomy=6,deadline='2026-10-06 08:35 +08:00'),indent=2))
    status('training')
    for seed in range(3):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[]
            for arm,cpu in zip(ARMS,CPUS):
                name=f'{arm}_s{seed}'
                args=['scripts.vascular_option_study','train','--arm',arm,'--seed',seed,'--out',RUNS/name,
                      '--workers',6,'--updates',480,'--rollout',128,'--minutes',140]
                futures.append(pool.submit(run,name,args,cpu))
            for f in as_completed(futures):save_result(f.result()); status(f'training_seed_{seed}')
        report()
        if DEADLINE-time.time()<1800:break
    status('paired_evaluation')
    # Same action-space fixed controls prevent attributing manual parameter choices to RL.
    evaluations=[]
    for option in range(9):
        evaluations.append((f'fixed_{option}',None,3,'noise',option,False))
    for seed in range(3):
        for arm in ARMS:
            name=f'{arm}_s{seed}'; ck=RUNS/name/'final.pt'
            if ck.exists():evaluations.append((name,ck,3,'noise',0,False))
    # N=1 sequential baseline, low-level memory policy, no-TPG and image-transfer probes.
    evaluations.append(('fixed_0',None,1,'noise',0,False))
    evaluations.append(('fixed_0_no_tpg',None,3,'noise',0,True))
    evaluations.append(('fixed_0',None,3,'image',0,False))
    for seed in range(3):
        ck=RUNS/f'memory_mappo_s{seed}'/'final.pt'
        if ck.exists():
            evaluations.extend([(f'memory_mappo_s{seed}',ck,1,'noise',0,False),
                                (f'memory_mappo_no_tpg_s{seed}',ck,3,'noise',0,True),
                                (f'memory_mappo_s{seed}',ck,3,'image',0,False)])
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures=[]
        for method,ck,n,sensing,option,no_tpg in evaluations:
            # Separate files per anatomy; a failed subprocess remains visible, no silent scene skipping.
            for anatomy in anatomies:
                name=f'{method}_N{n}_{sensing}_{anatomy}'
                args=['scripts.vascular_option_study','eval','--arm',method,'--out',OUT/'eval'/f'{name}.jsonl',
                      '--anatomies',anatomy,'--scenes',6 if sensing=='noise' else 2,'--clusters',n,
                      '--sensing',sensing,'--fixed-option',option]
                if ck:args+=['--checkpoint',ck]
                if no_tpg:args+=['--no-tpg']
                futures.append(pool.submit(run,name,args))
        for index,f in enumerate(as_completed(futures)):
            save_result(f.result())
            if index%12==0:status('paired_evaluation'); report()
    report(); status('figures')
    # Fixed predetermined scene, irrespective of whether the rollout succeeds.
    ck=RUNS/'memory_mappo_s0'/'final.pt'
    if ck.exists() and DEADLINE-time.time()>180:
        name='final_replay'
        save_result(run(name,['scripts.vascular_option_study','eval','--arm','memory_mappo_s0',
            '--checkpoint',ck,'--out',OUT/'final_replay.jsonl','--anatomies','mca_m1_lvo','--scenes',1,'--capture']))
        trace=OUT/'final_replay_mca_m1_lvo_2810000000.trace.json'
        if trace.exists():
            save_result(run('final_gui',['scripts.view_vascular_option_rl','--trace',trace,'--headless',
                            '--out',ROOT/'research/figures/EXP0060_20261006/gui'],python=RENDER_PY))
    report(); status('finished_with_failures' if failures else 'complete')


if __name__=='__main__':main()
