"""Continue both EXP30 arms from saved optimizers/RNG and evaluate 100 layouts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.train_mca_physical import atomic_json

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/wj/miniconda3/envs/v/bin/python'
RESOURCES = [(42, 'cuda:0', '0-5'), (43, 'cuda:1', '8-13'), (44, 'cuda:0', '14-19')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='research/runs/EXP_0030_PPO_REPAIR_20260930a')
    args = parser.parse_args()
    study = ROOT / args.study
    plan = json.loads((ROOT / 'configs/experiments/EXP_0030_EXTENSION.json').read_text())
    if json.loads((study / 'study_status.json').read_text())['phase'] != 'completed':
        raise ValueError('Both paired screening arms must complete first')
    if (study / 'extension_protocol.json').exists():
        raise ValueError('Extension already registered; inspect its state before restarting')
    atomic_json(study / 'extension_protocol.json', plan)
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1')
    report = {}
    for arm in plan['arm_order']:
        out = study / arm
        protocol = json.loads((out / 'protocol.json').read_text())
        protocol_rel = f"configs/experiments/EXP_0030_{'B_LOW_ACTOR_LR' if arm.startswith('b_') else 'A_FIXED'}_PURE_RL.json"
        frozen_protocol = out / 'source_snapshot' / protocol_rel
        helper = ROOT / 'scripts/evaluate_mca_frozen_checkpoint.py'
        frozen_helper = out / 'source_snapshot/scripts/evaluate_mca_frozen_checkpoint.py'
        if frozen_helper.exists() and frozen_helper.read_bytes() != helper.read_bytes():
            raise ValueError('Refusing to replace a previously staged evaluator')
        if not frozen_helper.exists():
            shutil.copy2(helper, frozen_helper)
        workers, launches = [], []
        for seed, device, cpus in RESOURCES:
            d = out / f'seed_{seed}'
            state = json.loads((d / 'status.json').read_text())
            if state['phase'] != 'completed' or state['transitions'] != 65536:
                raise ValueError('Expected a completed 65,536-step screening checkpoint')
            if (d / 'STOP').exists():
                raise ValueError('STOP marker present; refusing to override it')
            cmd = ['taskset', '-c', cpus, PYTHON, '-u', '-m', 'scripts.train_mca_parallel',
                   '--protocol', str(frozen_protocol), '--seed', str(seed), '--device', device,
                   '--n-envs', '8', '--workers', '6', '--out', str(d), '--resume',
                   '--timesteps', str(plan['target_steps_per_arm_seed'])]
            with (out / f'seed_{seed}.extension.stdout.log').open('xb') as log:
                p = subprocess.Popen(cmd, cwd=out / 'source_snapshot', env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            workers.append(p)
            launches.append(dict(seed=seed, pid=p.pid, command=cmd))
        atomic_json(out / 'extension_launch.json', dict(timestamp=time.time(), processes=launches,
                    resumed_from=65536, target=plan['target_steps_per_arm_seed'],
                    protocol_override='--timesteps only; frozen code/protocol, optimizer and RNG preserved'))
        atomic_json(out / 'correction_status.json', dict(phase='ppo_extended_comparison',
                    timestamp=time.time(), success_target_reached=False,
                    message='EXP30纯RL配对延长：两组每seed累计50万新环境步；先B后A，结束后各100布局复核。当前仍未达到80%，世界模型未接入。'))
        active_set = False
        while any(p.poll() is None for p in workers):
            ready = all(json.loads((out / f'seed_{x["seed"]}/status.json').read_text())['pid'] == x['pid'] for x in launches)
            if ready and not active_set:
                atomic_json(ROOT / 'research/runs/EXP_0023_MCA_PURE_RL_20260929a/active_run.json',
                            dict(path=str(out), experiment=protocol['experiment'], updated_at=time.time()))
                active_set = True
            atomic_json(study / 'extension_status.json', dict(phase='training', arm=arm,
                        pid=os.getpid(), timestamp=time.time(), processes=launches, completed_arms=list(report)))
            time.sleep(5)
        if any(p.returncode != 0 for p in workers):
            atomic_json(study / 'extension_status.json', dict(phase='failed', arm=arm,
                        timestamp=time.time(), returncodes=[p.returncode for p in workers]))
            raise RuntimeError('Continuation failed; preserve artifacts and do not advance')
        # Keep the trainer's original 20-layout files. Full validation is stored
        # separately and its first 20 episodes must reproduce those results.
        workers = []
        for seed, device, cpus in RESOURCES:
            d = out / f'seed_{seed}'
            result = d / 'full_validation' / f"evaluation_{plan['target_steps_per_arm_seed']}.json"
            cmd = ['taskset', '-c', cpus, PYTHON, '-u', '-m', 'scripts.evaluate_mca_frozen_checkpoint',
                   '--checkpoint', str(d / f"policy_{plan['target_steps_per_arm_seed']}.pt"),
                   '--protocol', str(frozen_protocol), '--device', device,
                   '--count', str(plan['evaluation_layouts']), '--out', str(result)]
            with (out / f'seed_{seed}.full_validation.stdout.log').open('xb') as log:
                workers.append(subprocess.Popen(cmd, cwd=ROOT, env=environment,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True))
        while any(p.poll() is None for p in workers):
            atomic_json(study / 'extension_status.json', dict(phase='full_validation', arm=arm,
                        pid=os.getpid(), timestamp=time.time(), evaluation_layouts=plan['evaluation_layouts']))
            time.sleep(5)
        if any(p.returncode != 0 for p in workers):
            atomic_json(study / 'extension_status.json', dict(phase='evaluation_failed', arm=arm,
                        timestamp=time.time(), returncodes=[p.returncode for p in workers]))
            raise RuntimeError('Full validation failed')
        report[arm] = {}
        for seed, _, _ in RESOURCES:
            d = out / f'seed_{seed}'
            filename = f"evaluation_{plan['target_steps_per_arm_seed']}.json"
            short_path, full_path = d / filename, d / 'full_validation' / filename
            short, full = json.loads(short_path.read_text()), json.loads(full_path.read_text())
            for a, b in zip(short['episodes'], full['episodes']):
                if any(a[k] != b[k] for k in ('seed', 'success', 'collision_free_success', 'reset_info')):
                    raise ValueError('Full validation does not reproduce the screening prefix')
            archive = d / 'screening_evaluations'; archive.mkdir(exist_ok=True)
            shutil.copy2(short_path, archive / filename)
            atomic_json(short_path, full)
            report[arm][str(seed)] = {k: v for k, v in full.items() if k != 'episodes'}
        atomic_json(study / 'extended_results.json', dict(results=report,
                    steps_per_arm_seed=plan['target_steps_per_arm_seed'], evaluation_layouts=plan['evaluation_layouts'],
                    world_model=False, scope='Paired development validation, not sealed final test'))
    atomic_json(study / 'extension_status.json', dict(phase='completed', timestamp=time.time(),
                completed_arms=list(report), world_model=False))


if __name__ == '__main__':
    main()
