"""Compare causal estimators on identical images/commands; do not change policy."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from marl.command_aligned_tracking import CommandAlignedProcessor
from scripts.run_option_learning import make_episode
from scripts.option_learning_episode import ROOT, option_hashes


def single(policy, seed, reference, out):
    ep=make_episode(seed)
    processor=ep.sensor.processor
    shadow=CommandAlignedProcessor(ep.cfg.clusters,ep.sensor.imager.preoperative_targets,
        ep.sensor.spec,speed=ep.library.speed,body_radius=ep.env.config.robot_radius_mm,
        duration=ep.env.config.episode_duration_s)
    ingest=processor.ingest;execute=ep.sensor.execute;prepare=ep.prepare
    def paired_ingest(frame):
        ingest(frame);shadow.ingest(frame)
    def paired_execute(command):
        requested=execute(command)
        shadow.record_command(ep.env.elapsed_s,requested)
        return requested
    processor.ingest=paired_ingest;ep.sensor.execute=paired_execute
    errors={'old_position':[],'aligned_position':[],'old_velocity':[],'aligned_velocity':[],
            'old_pair_distance':[],'aligned_pair_distance':[]}
    last=-1.
    def record():
        nonlocal last
        if ep.env.elapsed_s==last:return
        last=ep.env.elapsed_s
        now=ep.env.elapsed_s
        alternative=shadow.observe(now)
        for i in np.flatnonzero(ep.packet.active & alternative.active):
            stamp,p,v=processor.tracks[('robot',int(i))]
            old_position=p+(now-stamp)*v
            stamp,p,_=shadow.tracks[('robot',int(i))]
            new_position=p+(now-stamp)*shadow.drift.get(int(i),np.zeros(3))+ep.library.speed*shadow.commands.integral(stamp,now)[i]
            true_position=ep.env.positions_mm[i]
            errors['old_position'].append(float(np.linalg.norm(old_position-true_position)))
            errors['aligned_position'].append(float(np.linalg.norm(new_position-true_position)))
            physical_velocity=ep.env.velocity_mm_s[i]
            errors['old_velocity'].append(float(np.linalg.norm(ep.packet.navigation[i,3:6]*ep.library.speed-physical_velocity)))
            errors['aligned_velocity'].append(float(np.linalg.norm(alternative.navigation[i,3:6]*ep.library.speed-physical_velocity)))
            for j in range(i+1,ep.cfg.clusters):
                if ep.packet.peer_visible[i,j] and alternative.peer_visible[i,j]:
                    real=float(np.linalg.norm(ep.env.positions_mm[j]-ep.env.positions_mm[i]))
                    errors['old_pair_distance'].append(abs(float(np.linalg.norm(ep.packet.peer_relative_mm[i,j]))-real))
                    errors['aligned_pair_distance'].append(abs(float(np.linalg.norm(alternative.peer_relative_mm[i,j]))-real))
    def paired_prepare():
        prepare();record()
    ep.prepare=paired_prepare;record()
    try:
        while not ep.done:ep.step_option(ep.conventional_options(policy))
        result=ep.result(policy)
        expected=json.loads(Path(reference).read_text().splitlines()[0])
        same=result['final_state_hash']==expected['final_state_hash']
        summary=dict(policy=policy,scene_seed=seed,status='completed',shadow_only=True,
            truth_to_controller=False,shadow_to_controller=False,
            same_final_state_as_reference=same,final_state_hash=result['final_state_hash'],
            reference=str(reference),reference_source_hashes=expected['source_hashes'],
            current_source_hashes=option_hashes(),
            shadow_sha256=hashlib.sha256((ROOT/'marl/command_aligned_tracking.py').read_bytes()).hexdigest(),
            errors={k:dict(samples=len(v),mean=float(np.mean(v)),p95=float(np.quantile(v,.95)))
                    for k,v in errors.items()},
            note='Same images and commands; outcome truth is scoring only. Improved estimation alone is not learned-policy superiority or a deployment claim.')
        Path(out).write_text(json.dumps(summary,indent=2)+'\n')
        if not same:raise RuntimeError('Shadow instrumentation changed the reference trajectory')
    finally:ep.close()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='single':
        single(sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5]);return
    out=ROOT/'research/validation/EXP0051_SHADOW_TRACKING_20261004'
    out.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for policy in ('memory','priority'):
        for seed in (1740000000,1740000001):
            reference=ROOT/f'research/validation/EXP0051_PREFLIGHT_20261004/parts/{policy}_n3_scene{seed}.jsonl'
            tasks.append(dict(policy=policy,scene_seed=seed,reference=str(reference)))
    (out/'manifest.json').write_text(json.dumps(dict(tasks=tasks,
        source_hashes=option_hashes(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shadow_sha256=hashlib.sha256((ROOT/'marl/command_aligned_tracking.py').read_bytes()).hexdigest(),
        purpose='paired estimator residuals only; no new information or actions reach the policy',
        confirmation_accessed=False),indent=2)+'\n')
    def run(task):
        label=f"{task['policy']}_{task['scene_seed']}"
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/(label+'.console.txt')).open('x') as f:
            p=subprocess.run([sys.executable,'-X','faulthandler',str(Path(__file__).resolve()),'single',
                task['policy'],str(task['scene_seed']),task['reference'],str(out/(label+'.json'))],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        row=dict(task,returncode=p.returncode,status='completed' if p.returncode==0 else 'process_exit')
        if p.returncode==0:row.update(json.loads((out/(label+'.json')).read_text()))
        return row
    with ThreadPoolExecutor(2) as pool,(out/'attempts.jsonl').open('x',buffering=1) as f:
        for row in pool.map(run,tasks):
            f.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k in ('policy','scene_seed','status','same_final_state_as_reference','errors')}),flush=True)


if __name__=='__main__':main()
