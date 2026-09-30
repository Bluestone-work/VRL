"""Migrate each live EXP23 seed at its next durable checkpoint boundary."""
import json,os,signal,subprocess,time,shutil,hashlib
from pathlib import Path
from scripts.train_mca_parallel import hashes
from scripts.train_mca_physical import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research/runs/EXP_0023_MCA_PURE_RL_20260929a'
OUT=ROOT/'research/runs/EXP_0023_MCA_FAST_20260929b'
PYTHON='/home/wj/miniconda3/envs/v/bin/python'


def main():
    import torch
    OUT.mkdir(parents=True,exist_ok=False)
    source=hashes();atomic_json(OUT/'source_sha256.json',source)
    for name in source:
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    old_procs=json.loads((OLD/'launch.json').read_text())['processes']
    states={s:dict(seed=s,phase='waiting_for_checkpoint',minimum_checkpoint=8192) for s in (42,43,44)}
    atomic_json(OUT/'migration_status.json',states)
    atomic_json(OUT/'protocol.json',dict(experiment='EXP_0023_MCA_PURE_RL',phase='compiled_parallel_continuation',
          parent=str(OLD),budget_per_seed=3000000,seeds=[42,43,44],n_envs=8,workers=6,rollout_transitions=128,
          rollout_time_steps=16,physical_inputs_changed=False,old_n_envs=1,old_rollout_time_steps=128,
          training_trajectory_identical_to_serial=False,numerical_tolerance_mm=.001,
          original_convergence_tolerance_mm=.05,benchmark_steps_per_s=98.03,
          validation='342 passed, 1 skipped; 9 paired trajectories passed; exact parallel resume verified'))
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    launched=[];started=time.time()
    while any(v['phase']=='waiting_for_checkpoint' for v in states.values()):
        if time.time()-started>1200: raise RuntimeError('Checkpoint migration wait exceeded 20 minutes')
        for proc in old_procs:
            seed=proc['seed'];record=states[seed]
            if record['phase']!='waiting_for_checkpoint': continue
            checkpoint=OLD/f'seed_{seed}/latest.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            ts=payload['training_state']['transitions']
            if ts<8192: continue
            pid=proc['pid'];cmdline=Path(f'/proc/{pid}/cmdline').read_text()
            if 'scripts.train_mca_physical' not in cmdline or str(OLD/f'seed_{seed}') not in cmdline: raise RuntimeError('Old PID mismatch')
            os.kill(pid,signal.SIGSTOP)
            parent=OUT/'parent_checkpoints'/f'seed_{seed}.pt';parent.parent.mkdir(exist_ok=True)
            shutil.copy2(checkpoint,parent)
            old_status=json.loads((OLD/f'seed_{seed}/status.json').read_text())
            atomic_json(OUT/f'parent_status_seed_{seed}.json',old_status)
            device='cuda:1' if seed==43 else 'cuda:0';cpus={42:'0-5',43:'8-13',44:'14-19'}[seed]
            cmd=['taskset','-c',cpus,PYTHON,'-u','-m','scripts.train_mca_parallel','--seed',str(seed),
                 '--device',device,'--n-envs','8','--workers','6','--out',str(OUT/f'seed_{seed}'),'--migrate-from',str(parent)]
            with (OUT/f'seed_{seed}.stdout.log').open('xb') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(phase='starting',old_pid=pid,new_pid=child.pid,restored_steps=ts,
                          last_reported_old_steps=old_status['transitions'],parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),command=cmd)
            launched.append(dict(seed=seed,pid=child.pid,device=device,cpus=cpus,command=cmd))
            atomic_json(OUT/'launch.json',dict(started_at=started,processes=launched))
            deadline=time.time()+120
            while time.time()<deadline:
                path=OUT/f'seed_{seed}/status.json'
                if path.exists():
                    new=json.loads(path.read_text())
                    if new['phase']=='failed': break
                    if new['transitions']>=ts+128 and new['updates']>old_status['updates']-1:
                        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
                        record['phase']='running'
                        old_status.update(phase='superseded_by_compiled_parallel',replacement=str(OUT/f'seed_{seed}'),
                                          restored_checkpoint_steps=ts,migration_timestamp=time.time())
                        atomic_json(OLD/f'seed_{seed}/status.json',old_status)
                        break
                if child.poll() is not None: break
                time.sleep(1)
            if record['phase']!='running':
                child.terminate();os.kill(pid,signal.SIGCONT);record['phase']='migration_failed_original_resumed'
                atomic_json(OUT/'migration_status.json',states)
                raise RuntimeError(f'Migration failed seed {seed}; original resumed')
            print(json.dumps(record),flush=True)
            atomic_json(OUT/'migration_status.json',states)
        time.sleep(2)
    if not all(v['phase']=='running' for v in states.values()): raise RuntimeError('Incomplete migration')
    atomic_json(OLD/'active_run.json',dict(path=str(OUT),phase='compiled_parallel',updated_at=time.time()))
    atomic_json(OUT/'STARTUP_VERIFIED.json',dict(all_three_running=True,migrations=states,verified_at=time.time()))
    print('All seeds migrated and verified.',flush=True)


if __name__=='__main__':main()
