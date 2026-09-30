"""Run the preregistered paired EXP30 pure-RL screening from frozen sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.train_mca_parallel import hashes
from scripts.train_mca_physical import atomic_json
from scripts.mca_training_gate import require_feasibility_certificate

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/wj/miniconda3/envs/v/bin/python'
OLD = ROOT / 'research/runs/EXP_0029_MCA_ALL_RANDOM_20260930a'
ARMS = [('A_FIXED', 'A'), ('B_LOW_ACTOR_LR', 'B')]


def prepare(study, name, short):
    protocol_path = ROOT / f'configs/experiments/EXP_0030_{name}_PURE_RL.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['pure_rl'] and not protocol['world_model'] and not protocol['guidance_controller']
    assert protocol['physics_config'] == 'configs/experiments/EXP_0029_MCA_ALL_RANDOM_DYNAMICS.json'
    require_feasibility_certificate(protocol, ROOT)
    source = hashes(protocol_path, ROOT / protocol['physics_config'])
    preflight = ROOT / f'research/validation/EXP0030_{short}_GPU_PREFLIGHT_20260930'
    status = json.loads((preflight / 'status.json').read_text())
    manifest = json.loads((preflight / 'manifest.json').read_text())
    if status['phase'] != 'completed' or status['transitions'] < 16384:
        raise ValueError(f'{name} GPU preflight incomplete')
    if manifest['source_sha256'] != source:
        raise ValueError(f'{name} GPU preflight source mismatch')
    out = study / name.lower()
    out.mkdir(exist_ok=False)
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'source_sha256.json', source)
    files = set(source)
    for record in protocol['initialization_checkpoints'].values():
        if hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Changed parent checkpoint')
        files.add(record['path'])
    for field in ('feasibility_certificate', 'reference_certificate'):
        files.add(protocol[field])
        files.update(json.loads((ROOT / protocol[field]).read_text())['source_sha256'])
    for name_to_copy in sorted(files):
        destination = out / 'source_snapshot' / name_to_copy
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name_to_copy, destination)
    atomic_json(out / 'correction_status.json', dict(
        phase='ppo_screening', timestamp=time.time(), message=protocol['dashboard_message'],
        success_target_reached=False, clinical_or_hardware_validated=False))
    return out, protocol_path, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='research/runs/EXP_0030_PPO_REPAIR_20260930a')
    args = parser.parse_args()
    study = (ROOT / args.out).resolve()
    for seed in (42, 43, 44):
        status = json.loads((OLD / f'seed_{seed}/status.json').read_text())
        if status['phase'] not in ('paused', 'completed'):
            raise RuntimeError('EXP29 must be saved and paused before reusing its resources')
    study.mkdir(exist_ok=False)
    prepared = [prepare(study, name, short) for name, short in ARMS]
    if prepared[0][2]['initialization_checkpoints'] != prepared[1][2]['initialization_checkpoints']:
        raise ValueError('Unpaired initialization')
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1')
    all_results = {}
    for out, protocol_path, protocol in prepared:
        workers, records = [], []
        for seed, device, cpus in [(42, 'cuda:0', '0-5'), (43, 'cuda:1', '8-13'), (44, 'cuda:0', '14-19')]:
            cmd = ['taskset', '-c', cpus, PYTHON, '-u', '-m', 'scripts.train_mca_parallel',
                   '--protocol', str(out / 'source_snapshot' / protocol_path.relative_to(ROOT)),
                   '--seed', str(seed), '--device', device, '--n-envs', '8', '--workers', '6',
                   '--out', str(out / f'seed_{seed}'), '--initialize-from',
                   str(out / 'source_snapshot' / protocol['initialization_checkpoints'][str(seed)]['path'])]
            with (out / f'seed_{seed}.stdout.log').open('xb') as log:
                p = subprocess.Popen(cmd, cwd=out / 'source_snapshot', env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            workers.append(p)
            records.append(dict(seed=seed, pid=p.pid, device=device, cpus=cpus, command=cmd))
        atomic_json(out / 'launch.json', dict(started_at=time.time(), processes=records,
                    world_model=False, oracle_used_for_training=False, source_snapshot=True))
        pointer_updated = False
        while any(p.poll() is None for p in workers):
            if not pointer_updated and all((out / f'seed_{s}/latest.pt').exists() and
                                          (out / f'seed_{s}/status.json').exists() for s in (42, 43, 44)):
                atomic_json(ROOT / 'research/runs/EXP_0023_MCA_PURE_RL_20260929a/active_run.json',
                            dict(path=str(out), experiment=protocol['experiment'], updated_at=time.time()))
                pointer_updated = True
            atomic_json(study / 'study_status.json', dict(phase='running', arm=out.name,
                        timestamp=time.time(), pid=os.getpid(), workers=records,
                        completed_arms=list(all_results), world_model=False))
            time.sleep(5)
        if any(p.returncode != 0 for p in workers):
            atomic_json(study / 'study_status.json', dict(phase='failed', arm=out.name,
                        timestamp=time.time(), returncodes=[p.returncode for p in workers]))
            raise RuntimeError(f'{out.name} has failed workers; no promotion')
        results = {}
        for seed in (42, 43, 44):
            evaluation = out / f"seed_{seed}/evaluation_{protocol['timesteps_per_seed']}.json"
            data = json.loads(evaluation.read_text())
            results[str(seed)] = {k: v for k, v in data.items() if k != 'episodes'}
        all_results[out.name] = results
        atomic_json(study / 'screening_results.json', dict(results=all_results,
                    scope='20 matched development layouts per policy; not final or sealed-test evidence',
                    automatic_promotion=False, world_model=False))
        print(json.dumps(dict(arm=out.name, results=results)), flush=True)
    atomic_json(study / 'study_status.json', dict(phase='completed', timestamp=time.time(),
                completed_arms=list(all_results), world_model=False, automatic_promotion=False))


if __name__ == '__main__':
    main()
