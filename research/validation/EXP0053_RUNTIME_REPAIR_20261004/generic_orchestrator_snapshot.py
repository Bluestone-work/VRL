"""Run the preregistered same-version generic NumPy diagnostic and gated pilot.

This is a process-local runtime experiment, not a root-cause or safety claim.
The frozen EXP0053 algorithm and the prior failed attempts remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repair, study = args.repair.resolve(), args.study.resolve()
    protocol = json.loads((repair / 'generic_runtime_protocol.json').read_text())
    runtime = json.loads((repair / 'generic_runtime_context.json').read_text())
    if study.exists():
        raise FileExistsError(study)
    if (repair / 'generic_coordinator_status.json').exists():
        raise FileExistsError('This registered diagnostic has already been requested')
    import numpy as np
    import torch
    from scripts.tpg_episode import tpg_hashes
    source = tpg_hashes()
    reference = json.loads((root / 'research/validation/EXP0053_FINAL_AUDIT_20261004/FINAL_AUDIT.json').read_text())
    if source != reference['current_source_hashes']:
        raise RuntimeError('Frozen algorithm differs from audited corrected source')
    binary = Path(np.core._multiarray_umath.__file__)
    def check_runtime():
        if hashlib.sha256(binary.read_bytes()).hexdigest() != runtime['numpy_binary_sha256']:
            raise RuntimeError('NumPy binary provenance changed')
        if np.core._multiarray_umath.__cpu_dispatch__ or np.__version__ != '1.26.4':
            raise RuntimeError('Not the registered generic NumPy runtime')
        if sys.executable != runtime['executable'] or torch.__version__ != runtime['torch']:
            raise RuntimeError('Interpreter or torch does not match runtime provenance')
        if source != tpg_hashes():
            raise RuntimeError('Frozen algorithm changed during execution')
    check_runtime()
    test_record = json.loads((repair / 'generic_test_gate.json').read_text())
    tests = (repair / test_record['log']).read_text()
    if test_record['returncode'] != 0 or test_record['passed'] < 46:
        raise RuntimeError('The targeted test superset must pass before admission diagnostics')
    if not json.loads((repair / 'numerical_smoke.json').read_text())['passed']:
        raise RuntimeError('Numerical/torch checks did not pass')
    launcher = Path(__file__).read_bytes()
    runtime.update(launcher_sha256=hashlib.sha256(launcher).hexdigest(),
                   admission_protocol=str(repair / 'generic_runtime_protocol.json'),
                   study_source_hashes=source, native_fault_fixed=False)
    (repair / 'generic_orchestrator_snapshot.py').write_bytes(launcher)
    environment = dict(os.environ, PYTHONPATH=str(root), OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    environment.pop('NPY_DISABLE_CPU_FEATURES', None)
    attempts_path = repair / 'generic_attempts.jsonl'

    def record_attempt(row):
        with attempts_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)

    def job(name, arguments, result_folder=None):
        command = [sys.executable, '-m', *arguments]
        requested = dict(name=name, command=command,
                         requested_at=datetime.now().astimezone().isoformat())
        write_json(repair / (name + '_requested.json'), requested)
        started = time.monotonic()
        with (repair / (name + '.log')).open('x') as stream:
            process = subprocess.Popen(command, cwd=root, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if result_folder is not None and result_folder.exists():
                    context_path = result_folder / 'runtime_context.json'
                    if not context_path.exists():
                        write_json(context_path, runtime)
                time.sleep(0.1)
            code = process.wait()
        check_runtime()
        row = dict(**requested, returncode=code, wall_s=time.monotonic()-started)
        if result_folder is not None:
            if result_folder.exists():
                context_path = result_folder / 'runtime_context.json'
                if not context_path.exists():
                    write_json(context_path, runtime)
            state = result_folder / 'status.json'
            row['phase'] = json.loads(state.read_text())['phase'] if state.exists() else 'missing_status'
            row['complete'] = code == 0 and row['phase'] == 'completed'
        else:
            row['complete'] = code == 0
        write_json(repair / (name + '_completion.json'), row)
        return row

    def status(phase, **extra):
        write_json(repair / 'generic_coordinator_status.json',
                   dict(phase=phase, updated_at=datetime.now().astimezone().isoformat(),
                        confirmation_accessed=False, native_fault_fixed=False, **extra))

    status('joint_training_regressions')
    regressions = []
    r = protocol['regression']
    with ThreadPoolExecutor(max_workers=r['workers']) as executor:
        pending = []
        for index in range(r['joint_training_repetitions']):
            name = f'generic_joint_{index}'
            folder = repair / name
            arguments = ['scripts.run_tpg_learning', 'train', '--variant', 'both',
                         '--seed', str(r['joint_training_seed']), '--steps', str(r['joint_steps']),
                         '--scene-base', str(r['joint_scene_base']), '--out', str(folder)]
            pending.append(executor.submit(job, name, arguments, folder))
        for future in as_completed(pending):
            row = future.result()
            regressions.append(row)
            record_attempt(row)
    if not all(row['complete'] for row in regressions):
        status('failed_joint_regression_gate', performance_ranking_admitted=False)
        return 2

    preflights = []
    for index in range(r['full_preflight_repetitions']):
        status('preflight', repetition=index)
        folder = repair / f'generic_preflight_{index}'
        row = job(f'generic_preflight_{index}', ['scripts.run_tpg_study', '--phase', 'preflight',
                  '--out', str(folder), '--workers', str(r['workers'])], folder)
        record_attempt(row)
        identity = folder / 'untrained_identity.json'
        if not row['complete'] or not identity.exists() or not json.loads(identity.read_text())['passed']:
            status('failed_preflight_gate', failed_repetition=index, performance_ranking_admitted=False)
            return 2
        preflights.append(folder)
    repeat_checks = []
    for scene in r['preflight_scenes']:
        for policy in r['preflight_policies']:
            rows = [json.loads((folder / 'results' / f'{policy}_{scene}.jsonl').read_text()) for folder in preflights]
            repeat_checks.append(dict(scene=scene, policy=policy,
                same_initial=len({row['actual_initial_snapshot_hash'] for row in rows}) == 1,
                same_final=len({row['final_state_hash'] for row in rows}) == 1))
    passed = all(row['same_initial'] and row['same_final'] for row in repeat_checks)
    write_json(repair / 'generic_admission.json', dict(passed=passed, tests=tests.strip(),
        runtime=runtime, joint_training_runs=regressions,
        preflight_folders=[str(f) for f in preflights], repeated_determinism=repeat_checks,
        prior_failures_replaced=False, root_cause_proven=False, confirmation_accessed=False))
    if not passed:
        status('failed_repeated_determinism_gate', performance_ranking_admitted=False)
        return 2
    status('pilot_training_and_validation', runtime_admitted=True, study=str(study))
    row = job('generic_pilot', ['scripts.run_tpg_study', '--phase', 'pilot',
              '--out', str(study), '--workers', str(r['workers'])], study)
    record_attempt(row)
    if not row['complete']:
        status('failed_pilot_gate', runtime_admitted=True, performance_ranking_admitted=False)
        return 2
    status('analyzing', runtime_admitted=True)
    row = job('generic_analysis', ['scripts.analyze_tpg_study', str(study),
                                  '--preflight', str(preflights[0])])
    record_attempt(row)
    status('completed' if row['complete'] else 'failed_analysis_gate',
           runtime_admitted=True, performance_ranking_admitted=row['complete'], study=str(study))
    return 0 if row['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
