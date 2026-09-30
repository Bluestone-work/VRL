"""Launch fresh pure RL only after exact-source constructive feasibility gates."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from scripts.train_mca_parallel import hashes
from scripts.train_mca_physical import atomic_json,reset_with_valid_particles
from scripts.mca_training_gate import require_all_clear_capacity,require_feasibility_certificate

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/experiments/EXP_0026_MCA_IN_VITRO_PURE_RL.json'
OUT=ROOT/'research/runs/EXP_0026_MCA_IN_VITRO_20260930a'
PYTHON='/home/wj/miniconda3/envs/v/bin/python'
PREFLIGHT=ROOT/'research/validation/EXP0026_GPU_PREFLIGHT_20260930'


def main():
    protocol=json.loads(PROTOCOL.read_text())
    env=CompiledMCAPhysicalEnv(DynamicsConfig.from_json(ROOT/protocol['physics_config']))
    reset_with_valid_particles(env,42)
    require_all_clear_capacity(env);gate=require_feasibility_certificate(protocol,ROOT)
    preflight=json.loads((PREFLIGHT/'status.json').read_text())
    if preflight['phase']!='completed':raise ValueError('GPU preflight has not completed')
    source=hashes(PROTOCOL,ROOT/protocol['physics_config'])
    pilot=json.loads((PREFLIGHT/'manifest.json').read_text())
    if pilot['source_sha256']!=source:raise ValueError('GPU preflight is stale')
    parents=protocol.get('initialization_checkpoints',{})
    if parents:
        if set(parents)!={'42','43','44'} or protocol.get('fresh_weights',True):
            raise ValueError('Incomplete or inconsistent warm-start protocol')
        import hashlib
        for record in parents.values():
            if hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Warm-start checkpoint changed')
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/'protocol.json',protocol);atomic_json(OUT/'source_sha256.json',source)
    copied=set(source)
    copied.update(record['path'] for record in parents.values())
    for field in ('feasibility_certificate','reference_certificate'):
        certificate=json.loads((ROOT/protocol[field]).read_text())
        copied.add(protocol[field]);copied.update(certificate['source_sha256'])
    for name in copied:
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    process_env=os.environ.copy()
    process_env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    processes=[]
    for seed,device,cpus in [(42,'cuda:0','0-5'),(43,'cuda:1','8-13'),(44,'cuda:0','14-19')]:
        cmd=['taskset','-c',cpus,PYTHON,'-u','-m','scripts.train_mca_parallel','--protocol',
             str(OUT/'source_snapshot'/PROTOCOL.relative_to(ROOT)),'--seed',str(seed),'--device',device,
             '--n-envs','8','--workers','6','--out',str(OUT/f'seed_{seed}')]
        if parents:
            cmd+=['--initialize-from',str(OUT/'source_snapshot'/parents[str(seed)]['path'])]
        with (OUT/f'seed_{seed}.stdout.log').open('xb') as log:
            p=subprocess.Popen(cmd,cwd=OUT/'source_snapshot',env=process_env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(dict(seed=seed,pid=p.pid,device=device,cpus=cpus,command=cmd))
    atomic_json(OUT/'launch.json',dict(started_at=time.time(),processes=processes,fresh_weights=not bool(parents),
        feasibility=gate,oracle_used_for_training=False,source_snapshot=True))
    atomic_json(OUT/'correction_status.json',dict(phase='validated_in_vitro_training',timestamp=time.time(),
        message=protocol.get('dashboard_message','体外仿真基线已通过可完成性验证。诊断轨迹不计入 RL 成功率；新策略从零训练，独立评估另列。'),
        success_target_reached=False,clinical_or_hardware_validated=False))
    print(json.dumps(dict(run=str(OUT),processes=processes),indent=2))


if __name__=='__main__':main()
