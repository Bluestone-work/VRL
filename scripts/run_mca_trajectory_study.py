"""EXP32: explicit trajectory observations versus identical masked inputs."""
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from environments.mca_physical_env import DynamicsConfig
from scripts.train_mca_parallel import hashes
from scripts.train_mca_physical import atomic_json
from scripts.mca_training_gate import require_feasibility_certificate

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/wj/miniconda3/envs/v/bin/python'
RESOURCES = [(42,'cuda:0','0-5'),(43,'cuda:1','8-13'),(44,'cuda:0','14-19')]


def prepare(study, name):
    path = ROOT/f'configs/experiments/EXP_0032_{name}_PURE_RL.json'
    protocol=json.loads(path.read_text())
    assert protocol['pure_rl'] and not protocol['world_model'] and not protocol['guidance_controller']
    assert protocol['timesteps_per_seed']==500000 and protocol['eval_episodes']==100
    assert protocol['exploration_schedule'] is None
    require_feasibility_certificate(protocol,ROOT)
    source=hashes(path,ROOT/protocol['physics_config'])
    preflight=ROOT/f'research/runs/EXP0032_{name}_PREFLIGHT_20260930'
    status=json.loads((preflight/'status.json').read_text());manifest=json.loads((preflight/'manifest.json').read_text())
    if status['phase']!='completed' or status['transitions']!=16384 or manifest['source_sha256']!=source:
        raise ValueError(f'{name}: missing preflight or changed sources')
    out=study/name.lower();out.mkdir(exist_ok=False)
    atomic_json(out/'protocol.json',protocol);atomic_json(out/'source_sha256.json',source)
    files=set(source)
    for record in protocol['initialization_checkpoints'].values():
        if hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Changed parent checkpoint')
        files.add(record['path'])
    for field in ('feasibility_certificate','reference_certificate'):
        files.add(protocol[field]);files.update(json.loads((ROOT/protocol[field]).read_text())['source_sha256'])
    files.add('scripts/evaluate_mca_frozen_checkpoint.py')
    for name_to_copy in sorted(files):
        dest=out/'source_snapshot'/name_to_copy;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name_to_copy,dest)
    atomic_json(out/'correction_status.json',dict(phase='trajectory_observation_queued',timestamp=time.time(),
                message=protocol['dashboard_message'],success_target_reached=False))
    return out,path,protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='research/runs/EXP_0032_TRAJECTORY_20260930a')
    args=p.parse_args();study=(ROOT/args.out).resolve();study.mkdir(exist_ok=False)
    prepared=[prepare(study,name) for name in ('TRAJECTORY','MASKED')]
    a,b=prepared[0][2],prepared[1][2]
    for key in ('initialization_checkpoints','ppo','training_seed_base','validation_seed_base','weight_initialization'):
        if a[key]!=b[key]:raise ValueError(f'Unpaired {key}')
    physical=[]
    for protocol in (a,b):
        cfg=asdict(DynamicsConfig.from_json(ROOT/protocol['physics_config']));cfg.pop('obstacle_observation');physical.append(cfg)
    if physical[0]!=physical[1]:raise ValueError('Physical conditions differ across observation arms')
    prior=ROOT/'research/runs/EXP_0031_EXPLORATION_20260930a/study_status.json'
    while True:
        state=json.loads(prior.read_text())
        if state['phase']=='completed':break
        if state['phase']!='running':raise RuntimeError('EXP31 did not finish normally; do not reuse resources')
        atomic_json(study/'study_status.json',dict(phase='waiting_for_exp31',pid=os.getpid(),timestamp=time.time(),
                    predecessor=str(prior),world_model=False))
        time.sleep(5)
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    report={};reference_layouts=None
    for out,path,protocol in prepared:
        workers=[];launches=[]
        for seed,device,cpus in RESOURCES:
            cmd=['taskset','-c',cpus,PYTHON,'-u','-m','scripts.train_mca_parallel','--protocol',str(out/'source_snapshot'/path.relative_to(ROOT)),
                 '--seed',str(seed),'--device',device,'--n-envs','8','--workers','6','--out',str(out/f'seed_{seed}'),
                 '--initialize-from',str(out/'source_snapshot'/protocol['initialization_checkpoints'][str(seed)]['path'])]
            with (out/f'seed_{seed}.stdout.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=out/'source_snapshot',env=environment,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append(child);launches.append(dict(seed=seed,pid=child.pid,command=cmd))
        atomic_json(out/'launch.json',dict(timestamp=time.time(),processes=launches))
        pointer_set=False
        while any(child.poll() is None for child in workers):
            if not pointer_set and all((out/f'seed_{seed}/latest.pt').exists() and (out/f'seed_{seed}/status.json').exists() for seed,_,_ in RESOURCES):
                atomic_json(ROOT/'research/runs/EXP_0023_MCA_PURE_RL_20260929a/active_run.json',
                            dict(path=str(out),experiment=protocol['experiment'],updated_at=time.time()))
                pointer_set=True
            atomic_json(study/'study_status.json',dict(phase='running',arm=out.name,pid=os.getpid(),timestamp=time.time(),
                        processes=launches,completed_arms=list(report),world_model=False))
            time.sleep(5)
        if any(child.returncode!=0 for child in workers):
            atomic_json(study/'study_status.json',dict(phase='failed',arm=out.name,timestamp=time.time(),
                        returncodes=[child.returncode for child in workers]))
            raise RuntimeError('Training failed; preserve artifacts and do not advance')
        results={}
        for seed,_,_ in RESOURCES:
            data=json.loads((out/f'seed_{seed}/evaluation_500000.json').read_text())
            if len(data['episodes'])!=100:raise ValueError('Incomplete validation')
            layouts=[(e['seed'],e['reset_info']) for e in data['episodes']]
            if reference_layouts is None:reference_layouts=layouts
            if layouts!=reference_layouts:raise ValueError('Unpaired validation layouts')
            results[str(seed)]={k:v for k,v in data.items() if k!='episodes'}
        report[out.name]=results
        atomic_json(study/'results.json',dict(results=report,steps_per_arm_seed=500000,evaluation_layouts=100,
                    world_model=False,scope='Paired development validation of trajectory observations, not sealed test'))
    atomic_json(study/'study_status.json',dict(phase='completed',timestamp=time.time(),completed_arms=list(report),world_model=False))


if __name__=='__main__':main()
