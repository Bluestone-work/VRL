"""Resume the user-authorized EXP24 baseline from its verified source snapshot.

Physics and MAPPO code must match the checkpoint exactly. The current task
feasibility gate is evaluated explicitly; this is an authorized engineering
baseline with an unreachable all-clear target, not an 80% success claim.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from scripts.mca_training_gate import require_all_clear_capacity
from scripts.train_mca_physical import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'research/runs/EXP_0024_MCA_DISTRIBUTED_20260930a'
PYTHON = '/home/wj/miniconda3/envs/v/bin/python'


def main():
    torch.set_num_threads(1)
    snapshot = RUN/'source_snapshot'
    protocol_path = snapshot/'configs/experiments/EXP_0024_MCA_DISTRIBUTED_STARTS.json'
    protocol = json.loads(protocol_path.read_text())
    checks = []
    for seed in (42, 43, 44):
        directory = RUN/f'seed_{seed}'
        status = json.loads((directory/'status.json').read_text())
        cmdline = Path(f'/proc/{status["pid"]}/cmdline')
        if cmdline.exists() and 'scripts.train_mca_parallel' in cmdline.read_text():
            raise RuntimeError(f'Seed {seed} already has a live training process')
        if status['phase'] != 'paused':
            raise ValueError(f'Seed {seed} must have a saved paused checkpoint')
        path = directory/'latest.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        meta, state = payload['meta'], payload['training_state']
        if meta['training_seed'] != seed or meta['n_envs'] != 8:
            raise ValueError('Checkpoint seed or environment count mismatch')
        if state['transitions'] != status['transitions']:
            raise ValueError('Pause state is not the durable checkpoint')
        for name, expected in meta['source_sha256'].items():
            actual = hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Frozen source mismatch: {name}')
            # Later edits to entrypoint diagnostics are isolated by the snapshot;
            # do not silently omit any model, optimizer, or environment repair.
            if name.startswith(('environments/', 'marl/', 'configs/')):
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Training semantics changed since pause: {name}')
        for key in ('actor', 'critic'):
            if not all(torch.isfinite(value).all() for value in payload[key].values()):
                raise FloatingPointError(f'Nonfinite checkpoint {key}')
        for key in ('actor_optimizer', 'critic_optimizer'):
            for optimizer_state in payload[key]['state'].values():
                if not all(torch.isfinite(v).all() for v in optimizer_state.values() if torch.is_tensor(v)):
                    raise FloatingPointError(f'Nonfinite checkpoint {key}')
        env = state['envs'][0]
        if env.config.robot_initialization != 'distributed_branches':
            raise ValueError('Robot initialization is not distributed')
        feasibility = require_all_clear_capacity(env, allow_unreachable_baseline=True)
        checks.append(dict(seed=seed, checkpoint_steps=state['transitions'],
                           updates=state['updates'], checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           feasibility=feasibility, source_snapshot_verified=True))

    archive = RUN/'resume_history'/time.strftime('%Y%m%d_%H%M%S')
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('launch.json', 'correction_status.json', 'services.json'):
        if (RUN/name).exists():
            shutil.copy2(RUN/name, archive/name)
    for check in checks:
        directory = RUN/f'seed_{check["seed"]}'
        target = archive/directory.name
        target.mkdir()
        for name in ('latest.pt', 'status.json', 'STOP', 'manifest.json'):
            if (directory/name).exists():
                shutil.copy2(directory/name, target/name)
    atomic_json(archive/'authorization.json', dict(
        user_request='开始启动长跑把，记得开启可视化。', timestamp=time.time(),
        scope='Resume unchanged EXP24 distributed-start removal-optimization baseline',
        allow_unreachable_baseline=True, success_target_reached=False,
        source_snapshot=str(snapshot), checks=checks,
        policy_update_source_changed=False, physics_inputs_changed=False,
        target_environment_steps_per_seed=protocol['timesteps_per_seed']))

    process_env = os.environ.copy()
    process_env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1', PYTHONPATH=str(snapshot))
    # Verify actual module resolution before launching detached workers.
    probe = subprocess.run([PYTHON, '-c',
        'from scripts.train_mca_parallel import ROOT; print(ROOT)'],
        cwd=snapshot, env=process_env, capture_output=True, text=True, check=True)
    if Path(probe.stdout.strip()) != snapshot:
        raise RuntimeError('Snapshot import resolution mismatch')
    processes = []
    for seed, device, cpus in [(42, 'cuda:0', '0-5'), (43, 'cuda:1', '8-13'), (44, 'cuda:0', '14-19')]:
        directory = RUN/f'seed_{seed}'
        (directory/'STOP').unlink(missing_ok=True)
        command = ['taskset', '-c', cpus, PYTHON, '-u', '-m', 'scripts.train_mca_parallel',
                   '--protocol', str(protocol_path), '--seed', str(seed), '--device', device,
                   '--n-envs', '8', '--workers', '6', '--out', str(directory), '--resume']
        with (RUN/f'seed_{seed}.resume.stdout.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=snapshot, env=process_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(dict(seed=seed, pid=child.pid, device=device, cpus=cpus,
                              command=command, cwd=str(snapshot)))
        atomic_json(RUN/'launch.json', dict(started_at=time.time(), processes=processes,
                    resumed=True, resume_archive=str(archive), checks=checks,
                    source_snapshot=str(snapshot), allow_unreachable_baseline=True))
    atomic_json(RUN/'correction_status.json', dict(phase='authorized_baseline_resuming',
        message='已按用户要求恢复长跑：保持当前物理参数和分散起点，三个种子各累计300万环境步。当前1秒任务全清除不可达，本轮记录去除量优化结果；未达到80%。',
        timestamp=time.time(), report='research/experiments/EXP_0024_DISTRIBUTED_STARTS.md',
        allow_unreachable_baseline=True, success_target_reached=False))
    print(json.dumps(dict(run=str(RUN), archive=str(archive), processes=processes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
