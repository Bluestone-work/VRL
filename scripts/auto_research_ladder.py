#!/usr/bin/env python3
"""Validation-only, explicitly gated MAPPO ladder. Defaults to a read-only plan.

Stages 1-3 are implemented; stages 4-5 require separate implementation.
No test-set evaluation or automatic significance/equivalence claims are made.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RETRY_SIGNALS = {4, 6, 7, 8, 11}  # SIGILL/ABRT/BUS/FPE/SEGV, never TERM/INT
ARMS = {
    1: {f'{contact}_{critic}': {'contact-mode': contact, 'critic-value-mode': critic}
        for contact in ('euclidean', 'geodesic') for critic in ('q', 'v')},
    2: {'geometric': {'obs-mode': 'geometric'},
        'geometric_v2': {'obs-mode': 'geometric_v2'},
        'no_margin': {'obs-mode': 'geometric_v2', 'no-control-margin': True}},
    3: {'baseline': {}, 'no_coverage': {'coverage-bonus': 0.0},
        'step_cost': {'step-cost': 0.01}, 'single_reward': {'reward-double-count': 'off'},
        'approach': {'approach-scale': 1.0}},
}


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


@contextmanager
def exclusive_lock(path):
    """OS advisory locks release on native crashes; leave inode in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'active research lock: {path}') from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def retryable(returncode):
    return returncode in {-s for s in RETRY_SIGNALS} | {128+s for s in RETRY_SIGNALS}


def resolve_cpu_affinity(cpu_list=None, exclude_cpus='6,7', disabled=False):
    """Respect the process cpuset; exclude this host's suspect CPUs by default."""
    if disabled:
        return None
    available = set(os.sched_getaffinity(0))
    def parse(spec):
        result = set()
        for part in spec.split(','):
            bounds = part.strip().split('-')
            if not part.strip() or len(bounds) > 2:
                raise ValueError('invalid CPU list')
            low, high = int(bounds[0]), int(bounds[-1])
            if low < 0 or high < low:
                raise ValueError('invalid CPU range')
            result.update(range(low, high + 1))
        return result
    selected = parse(cpu_list) if cpu_list is not None else available - (parse(exclude_cpus) if exclude_cpus else set())
    if not selected or not selected <= available:
        raise ValueError('CPU affinity must be nonempty and within the current allowed cpuset')
    return sorted(selected)


def affinity_command(command, cpus):
    return ['taskset', '-c', ','.join(map(str, cpus)), *command] if cpus is not None else command


def command_for(config, run_dir, device, resume=None):
    command = [sys.executable, str(ROOT/'scripts/train_vector_mappo.py'),
               '--run-dir', str(run_dir), '--device', device]
    for key, value in config.items():
        if key == "cpu-affinity":
            continue
        if value is True:
            command.append('--' + key)
        elif value is not False and value is not None:
            command.extend(['--' + key, str(value)])
    if resume:
        command.extend(['--resume', str(resume)])
    return affinity_command(command, config["cpu-affinity"] if "cpu-affinity" in config else resolve_cpu_affinity())


def latest_checkpoint(run_dir):
    checkpoints = list(run_dir.glob('checkpoint_*.pt'))
    candidates = [(int(p.stem.split('_')[-1]), p) for p in checkpoints
                  if p.stem.split('_')[-1].isdigit()]
    # final_policy carries training state too; inspected by recovery subprocess.
    final = run_dir/'final_policy.pt'
    if final.exists():
        candidates.append((float('inf'), final))
    return max(candidates, default=(0, None))[1]


def recover(run_dir, checkpoint, timeout, cpus=None):
    """Read trusted local checkpoint in a bounded child, never import torch here."""
    probe = [sys.executable, '-c',
             'import json,sys,torch; c=torch.load(sys.argv[1],map_location="cpu"); '
             's=c.get("training_state",{}); '
             'print(json.dumps({"transitions":s.get("transitions"),'
             '"dataset_transitions":s.get("dataset_writer",{}).get("total_written")}))',
             str(checkpoint)]
    result = subprocess.run(affinity_command(probe, cpus), capture_output=True, text=True, timeout=timeout)
    with (run_dir/'recovery.log').open('a') as log:
        log.write(json.dumps({'checkpoint': str(checkpoint), 'returncode': result.returncode,
                              'stdout': result.stdout, 'stderr': result.stderr})+'\n')
    if result.returncode:
        raise RuntimeError('checkpoint inspection failed; see recovery.log')
    state = json.loads(result.stdout.strip().splitlines()[-1])
    if state['transitions'] is None:
        raise RuntimeError('checkpoint lacks resumable training state')
    dataset = run_dir/'dataset'
    if (dataset/'manifest.jsonl').exists():
        if state['dataset_transitions'] is None:
            raise RuntimeError('checkpoint lacks dataset position; refusing unaligned resume')
        command = [sys.executable, str(ROOT/'scripts/recover_transition_dataset.py'),
                   '--dataset-dir', str(dataset), '--keep-transitions', str(state['dataset_transitions'])]
        with (run_dir/'recovery.log').open('a') as log:
            result = subprocess.run(affinity_command(command, cpus), stdout=log, stderr=log, timeout=timeout)
        if result.returncode:
            raise RuntimeError('dataset recovery failed')
    return int(state['transitions'])


def run_job(job, device, output, attempts, timeout, cancelled):
    config = job['config']
    identity = fingerprint(config)
    run = output/f"{job['arm']}_seed{config['seed']}_{identity[:12]}"
    with exclusive_lock(run/'.lock'):
        provenance = run/'ladder_config.json'
        if provenance.exists() and json.loads(provenance.read_text())['fingerprint'] != identity:
            raise RuntimeError('configuration fingerprint mismatch')
        atomic_json(provenance, {'fingerprint': identity, 'config': config})
        summary = run/'summary.json'
        if summary.exists() and int(json.loads(summary.read_text()).get('real_transitions', 0)) >= config['timesteps']:
            return {'arm': job['arm'], 'seed': config['seed'], 'status': 'complete', 'skipped': True, 'run': str(run)}
        existing = [int(p.stem.split('_')[-1]) for p in run.glob('attempt_*.json')]
        next_attempt = max(existing, default=0)+1
        if existing:
            previous = json.loads((run/f'attempt_{max(existing)}.json').read_text())
            if previous.get('status') in ('cancelled', 'timeout') or (
                previous.get('returncode') is not None
                and previous['returncode'] != 0
                and not retryable(previous['returncode'])
            ):
                raise RuntimeError(f'previous non-retryable attempt; inspect logs: {run}')
        # attempts is a total budget, persisted across invocations.
        for attempt in range(next_attempt, attempts+1):
            if cancelled.is_set():
                return {'status': 'cancelled', 'run': str(run)}
            resume = latest_checkpoint(run)
            if resume:
                recover(run, resume, min(timeout, 60), config["cpu-affinity"] if "cpu-affinity" in config else resolve_cpu_affinity())
            command = command_for(config, run, device, resume)
            record = {'command': command, 'attempt': attempt, 'started': time.time(), 'status': 'running'}
            record_path = run/f'attempt_{attempt}.json'
            atomic_json(record_path, record)
            started = time.monotonic()
            with (run/f'attempt_{attempt}.stdout.log').open('x') as stdout, (run/f'attempt_{attempt}.stderr.log').open('x') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
                while process.poll() is None:
                    if cancelled.wait(.2) or time.monotonic()-started > timeout:
                        record['status'] = 'cancelled' if cancelled.is_set() else 'timeout'
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                code = process.wait()
            record.update(returncode=code, finished=time.time())
            if record['status'] == 'running':
                record['status'] = 'complete' if code == 0 else 'failed'
            atomic_json(record_path, record)
            if code == 0:
                if not summary.exists() or json.loads(summary.read_text()).get('real_transitions', 0) < config['timesteps']:
                    raise RuntimeError(f'exit 0 without completed budget: {run}')
                return {'arm': job['arm'], 'seed': config['seed'], 'status': 'complete', 'run': str(run)}
            if record['status'] in ('cancelled', 'timeout') or not retryable(code):
                raise RuntimeError(f'job stopped ({record["status"]}, rc={code}); logs: {run}')
        raise RuntimeError(f'job exhausted its {attempts}-attempt budget: {run}')


def paired_interval(left, right, metric="success", draws=5000, seed=0):
    """Descriptive hierarchical paired bootstrap; deliberately returns no p-value.

    Keys are (training seed, scenario, evaluation episode seed). Scenario strata
    receive equal weight. With only three seeds this is not an equivalence test.
    """
    def indexed(records):
        result = {}
        for row in records:
            key = (row["training_seed"], row["scenario"], row["episode_seed"])
            if key in result:
                raise ValueError("duplicate paired episode key")
            result[key] = float(row[metric])
        return result
    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError("paired episode keys must match exactly")
    seeds = sorted({k[0] for k in a})
    scenarios = sorted({k[1] for k in a})
    groups = {(s, t): [b[k]-a[k] for k in a if k[:2] == (s, t)]
              for s in seeds for t in scenarios}
    if any(not values for values in groups.values()):
        raise ValueError("each seed must cover every scenario")
    mean = lambda values: sum(values)/len(values)
    estimate = mean([mean(v) for v in groups.values()])
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choices(seeds, k=len(seeds))
        samples.append(mean([mean(rng.choices(groups[s, t], k=len(groups[s, t])))
                             for s in selected for t in scenarios]))
    samples.sort()
    return {"difference": estimate, "ci95": [samples[int(.025*(draws-1))],
                                               samples[int(.975*(draws-1))]],
            "training_seeds": len(seeds), "interpretation": "descriptive only; manual gate"}


def validation_report(results):
    arms = {}
    for result in results:
        summary = json.loads((Path(result["run"])/"summary.json").read_text())
        evaluation = summary.get("final_evaluation", {})
        if evaluation.get("split") != "validation":
            return {"status": "unavailable: missing validation episode records", "gate": "manual"}
        arms.setdefault(result["arm"], []).extend(
            dict(row, training_seed=result["seed"]) for row in evaluation.get("episodes", []))
    names = sorted(arms)
    if len(names) < 2:
        return {"status": "need at least two arms", "gate": "manual"}
    comparisons = {}
    for name in names[1:]:
        comparisons[name] = {metric: paired_interval(arms[names[0]], arms[name], metric)
                             for metric in ("success", "removal_rate")}
    return {"reference": names[0], "comparisons": comparisons,
            "gate": "manual; non-significance is not equivalence; no p-values or Holm applied"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    affinity = parser.add_mutually_exclusive_group()
    affinity.add_argument('--cpu-list', help='override CPU list, e.g. 0-5,8-23; within current cpuset')
    affinity.add_argument('--no-cpu-affinity', action='store_true', help='disable taskset prefix; inherit affinity')
    parser.add_argument('--exclude-cpus', default='6,7', help='CPU IDs excluded by default (ignored with --cpu-list)')
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--output', type=Path, default=Path('experiments/ladder'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--n-envs', type=int, default=64)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--final-eval-episodes', type=int, default=20)
    parser.add_argument('--scenario-pool', default='anatomical')
    parser.add_argument('--gate-file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= len(set(args.seeds)) <= 3 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('use 1-3 distinct seeds; the study budget is three')
    if len(set(args.devices)) != len(args.devices) or not args.devices:
        raise ValueError('one worker per distinct device')
    if min(args.timesteps, args.n_envs, args.n_steps, args.batch_size, args.n_epochs,
           args.eval_episodes, args.final_eval_episodes, args.max_attempts, args.timeout) <= 0:
        raise ValueError('budgets and timeout must be positive')
    cpus = resolve_cpu_affinity(args.cpu_list, args.exclude_cpus, args.no_cpu_affinity)
    base = {'cpu-affinity': cpus, 'architecture': 'gat', 'control-mode': 'flow_guided', 'contact-mode': 'geodesic',
            'critic-value-mode': 'v', 'dropout': 0.0, 'obs-mode': 'geometric',
            'timesteps': args.timesteps, 'n-envs': args.n_envs, 'n-steps': args.n_steps,
            'batch-size': args.batch_size, 'n-epochs': args.n_epochs, 'scenario-pool': args.scenario_pool,
            'eval-episodes': args.eval_episodes, 'final-eval-episodes': args.final_eval_episodes, 'no-dataset': True,
            'save-interval': max(args.n_envs*args.n_steps, min(args.timesteps, 100000))}
    if args.stage > 1:
        if not args.gate_file:
            raise ValueError('later stages require a human-approved validation gate file')
        gate = json.loads(args.gate_file.read_text())
        if gate.get('approved') is not True or gate.get('split') != 'validation' or gate.get('completed_stage') != args.stage-1:
            raise ValueError('invalid gate approval')
        allowed = {'contact-mode', 'critic-value-mode', 'dropout', 'obs-mode', 'no-control-margin'}
        selected = gate.get('selected_config', {})
        if set(selected)-allowed:
            raise ValueError('gate selected_config contains unsupported overrides')
        base.update(selected)
    arms = args.arms or list(ARMS[args.stage])
    if set(arms)-ARMS[args.stage].keys():
        raise ValueError('unknown stage arm')
    jobs = [{'arm': arm, 'config': {**base, **ARMS[args.stage][arm], 'seed': seed}}
            for arm in arms for seed in args.seeds]
    plan = {'stage': args.stage, 'jobs': jobs, 'devices': args.devices, 'jobs_per_device': 1, 'cpu_affinity': cpus,
            'gate': 'manual, validation only; confidence intervals are descriptive',
            'controller_baseline': 'not implemented', 'stages_4_5': 'not implemented'}
    print(json.dumps(plan, indent=2))
    if not args.execute or args.dry_run:
        return 0
    output = args.output.resolve()
    cancelled = threading.Event()
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: cancelled.set())
    try:
        with exclusive_lock(output/'.ladder.lock'):
            atomic_json(output/f'stage_{args.stage}_plan.json', plan)
            def worker(device, assigned):
                results = []
                for job in assigned:
                    if cancelled.is_set():
                        break
                    results.append(run_job(job, device, output, args.max_attempts, args.timeout, cancelled))
                return results
            with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
                futures = [pool.submit(worker, device, jobs[i::len(args.devices)])
                           for i, device in enumerate(args.devices)]
                results = []
                try:
                    for future in futures:
                        results.extend(future.result())
                except BaseException as exc:
                    cancelled.set()
                    atomic_json(output/f'stage_{args.stage}_failure.json', {'error': str(exc), 'time': time.time()})
                    raise
            atomic_json(output/f'stage_{args.stage}_result.json', {'jobs': results, 'gate': 'awaiting human validation review'})
            if not cancelled.is_set():
                atomic_json(output/f'stage_{args.stage}_validation.json', validation_report(results))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 130 if cancelled.is_set() else 0


if __name__ == '__main__':
    raise SystemExit(main())
