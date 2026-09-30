"""Start the separately registered distributed-start task with fresh policies."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.train_mca_parallel import hashes
from scripts.train_mca_physical import atomic_json
from scripts.train_mca_physical import reset_with_valid_particles
from scripts.mca_training_gate import require_all_clear_capacity
from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/experiments/EXP_0024_MCA_DISTRIBUTED_STARTS.json'
OUT = ROOT/'research/runs/EXP_0024_MCA_DISTRIBUTED_20260930a'
PYTHON = '/home/wj/miniconda3/envs/v/bin/python'


def main():
    protocol = json.loads(PROTOCOL.read_text())
    gate_env = CompiledMCAPhysicalEnv(DynamicsConfig.from_json(ROOT/protocol['physics_config']))
    reset_with_valid_particles(gate_env, 42)
    require_all_clear_capacity(gate_env)
    source = hashes(PROTOCOL, ROOT/protocol['physics_config'])
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/'protocol.json', protocol)
    atomic_json(OUT/'source_sha256.json', source)
    for name in source:
        dest = OUT/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    processes = []
    for seed, device, cpus in [(42, 'cuda:0', '0-5'), (43, 'cuda:1', '8-13'), (44, 'cuda:0', '14-19')]:
        cmd = ['taskset', '-c', cpus, PYTHON, '-u', '-m', 'scripts.train_mca_parallel',
               '--protocol', str(PROTOCOL), '--seed', str(seed), '--device', device,
               '--n-envs', '8', '--workers', '6', '--out', str(OUT/f'seed_{seed}')]
        with (OUT/f'seed_{seed}.stdout.log').open('xb') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(dict(seed=seed, pid=child.pid, device=device, cpus=cpus, command=cmd))
    atomic_json(OUT/'launch.json', dict(started_at=time.time(), processes=processes,
                fresh_weights=True, parent_run_failed='EXP_0023_MCA_FAST_20260929b',
                initialization='distributed_branches', prior_results_comparable=False))
    print(json.dumps(dict(run=str(OUT), processes=processes), indent=2))


if __name__ == '__main__':
    main()
