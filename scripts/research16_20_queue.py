"""Frozen-source, resource-bounded EXP16--20 DAG. Run this supervisor in tmux."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def freeze(destination):
    destination.mkdir()
    manifest = {}
    for directory in ('environments', 'marl', 'scripts', 'configs', 'tests'):
        for src in (ROOT/directory).rglob('*'):
            if not src.is_file() or '__pycache__' in src.parts or src.suffix not in ('.py', '.json', '.sh', '.yaml', '.yml'):
                continue
            rel = src.relative_to(ROOT)
            dst = destination/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest[str(rel)] = hashlib.sha256(dst.read_bytes()).hexdigest()
    for name in ('requirements.txt', 'research/EXP16_20_V2_PROTOCOL.md',
                 'research/EXP16_20_V2_VALIDATION.md',
                 'research/validation/EXP16_20_20260927_pinned.xml'):
        src, dst = ROOT/name, destination/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    save(destination/'manifest.json', manifest)
    return manifest


class Queue:
    def __init__(self, args):
        self.args, self.root = args, Path(args.out).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.snapshot = self.root/'snapshot'
        manifest = freeze(self.snapshot)
        (self.root/'jobs').mkdir()
        (self.root/'logs').mkdir()
        save(self.root/'registration.json', dict(config=vars(args), source_files=len(manifest),
            snapshot_manifest_sha256=hashlib.sha256((self.snapshot/'manifest.json').read_bytes()).hexdigest(),
            python=sys.executable, created=dt.datetime.now().isoformat(),
            selection='all fixed budget checkpoints; no best-checkpoint selection',
            evaluation_device='cuda:0', split='development_validation_v2',
            inherited_environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PATH')},
            world_gate='1/3-step obs below persistence AND displacement below kinematics'))
        (self.root/'software.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        self.jobs, self.running, self.seen = {}, {}, set()
        self.events = (self.root/'events.jsonl').open('x', buffering=1)
        self.gpu_cursor = 0
        for seed in args.seeds:
            self.add(f'16_s{seed}', 'train', seed, arm=16)
            self.add(f'motion_data_s{seed}', 'collect', seed, kind='motion')

    def event(self, event, **kw):
        self.events.write(json.dumps(dict(time=dt.datetime.now().isoformat(), event=event, **kw))+'\n')

    def add(self, name, job, seed, **kwargs):
        if name not in self.jobs:
            self.jobs[name] = dict(name=name, job=job, seed=seed, options=kwargs, status='pending', attempt=0)
            self.event('queued', name=name, job=job, seed=seed, options=kwargs)

    def output(self, name):
        return Path(self.jobs[name]['out'])

    def complete(self, name):
        return name in self.jobs and self.jobs[name]['status'] == 'complete'

    def discover(self):
        for seed in self.args.seeds:
            data, fit = f'motion_data_s{seed}', f'motion_fit_s{seed}'
            if self.complete(data):
                self.add(fit, 'fit', seed, kind='motion', data=str(self.output(data)/'data.npz'))
            if self.complete(fit):
                predictor = str(self.output(fit)/'model.pt')
                for arm in (17, 18):
                    self.add(f'{arm}_s{seed}', 'train', seed, arm=arm, predictor=predictor)
            train17 = f'17_s{seed}'
            if train17 in self.jobs and 'out' in self.jobs[train17]:
                checkpoints = sorted(self.output(train17).glob('checkpoint_*.ready.json'),
                                     key=lambda p: int(p.name.split('_')[1].split('.')[0]))
                if checkpoints:
                    ckpt = read(checkpoints[0])
                    previous_failed = True
                    for stage, budget in enumerate(self.args.world_budgets):
                        wd, wf = f'world_data{stage}_s{seed}', f'world_fit{stage}_s{seed}'
                        if previous_failed:
                            self.add(wd, 'collect', seed, arm=19, kind='world',
                                predictor=ckpt['predictor'], checkpoint=ckpt['checkpoint'], model_data_steps=budget)
                        if self.complete(wd):
                            self.add(wf, 'fit', seed, arm=19, kind='world', data=str(self.output(wd)/'data.npz'))
                        if not self.complete(wf):
                            break
                        passed = read(self.output(wf)/'metrics.json')['gate']['passed']
                        if passed:
                            self.add(f'20_s{seed}', 'train', seed, arm=20, predictor=ckpt['predictor'],
                                world=str(self.output(wf)/'model.pt'), model_data_steps=budget)
                            break
                        previous_failed = True
                        marker = f'world_gate_{stage}_{seed}'
                        if marker not in self.seen:
                            self.seen.add(marker)
                            self.event('validation_gate_failed', seed=seed, stage=stage,
                                       metrics=read(self.output(wf)/'metrics.json'))
                        if stage == len(self.args.world_budgets)-1:
                            blocked = f'20_s{seed}'
                            if blocked not in self.jobs:
                                self.jobs[blocked] = dict(name=blocked, job='train', seed=seed,
                                    status='blocked', reason='world validation gate failed at all registered data budgets')
                                self.event('blocked', name=blocked, reason=self.jobs[blocked]['reason'])
        for name, job in list(self.jobs.items()):
            if job['job'] != 'train' or 'out' not in job:
                continue
            for path in self.output(name).glob('checkpoint_*.ready.json'):
                if str(path) in self.seen:
                    continue
                self.seen.add(str(path))
                ckpt = read(path)
                if hashlib.sha256(Path(ckpt['checkpoint']).read_bytes()).hexdigest() != ckpt['sha256']:
                    raise RuntimeError(f'Checkpoint checksum mismatch: {path}')
                self.add(f'eval_{name}_{ckpt["transitions"]}_a{job["attempt"]}', 'evaluate', job['seed'],
                    arm=ckpt['arm'], checkpoint=ckpt['checkpoint'], predictor=ckpt['predictor'], world=ckpt['world'])

    def resources(self):
        try:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            gpu_free = [int(v) for v in memory.split()]
        except (subprocess.SubprocessError, ValueError):
            return []
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))/1024
        if available < 8192 or shutil.disk_usage(self.root).free < 20*1024**3:
            return []
        return [g for g in (0, 1) if len(gpu_free)>g and gpu_free[g] > self.args.gpu_reserve_mb]

    def launch(self):
        candidates = sorted((j for j in self.jobs.values() if j['status']=='pending'),
                            key=lambda j: {'fit': 0, 'collect': 1, 'evaluate': 2, 'train': 3}[j['job']])
        for job in candidates:
            if len(self.running) >= self.args.max_jobs:
                break
            free = self.resources()
            if not free:
                return
            if job['job'] == 'evaluate':
                if 0 not in free or sum(self.jobs[n]['job']=='evaluate' for n in self.running) >= 3:
                    continue
                gpu = 0
            else:
                gpu = free[self.gpu_cursor % len(free)]
                self.gpu_cursor += 1
            job['attempt'] += 1
            out = self.root/'jobs'/f'{job["name"]}_attempt{job["attempt"]}'
            command = [sys.executable, str(self.snapshot/'scripts/research16_20_worker.py'), job['job'],
                '--seed', str(job['seed']), '--device', f'cuda:{gpu}', '--out', str(out),
                '--n-envs', str(self.args.n_envs), '--timesteps', str(self.args.timesteps),
                '--checkpoint-interval', str(self.args.checkpoint_interval),
                '--fit-epochs', str(self.args.fit_epochs), '--eval-episodes', str(self.args.eval_episodes),
                '--epochs', str(self.args.epochs), '--model-data-steps', str(self.args.model_data_steps)]
            for key, value in job['options'].items():
                command += ['--'+key.replace('_', '-'), str(value)]
            affinity = sorted(set(os.sched_getaffinity(0))-{6, 7})
            command = ['taskset', '-c', ','.join(map(str, affinity)), *command]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONPATH=str(self.snapshot),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
            logfile = self.root/'logs'/f'{job["name"]}_attempt{job["attempt"]}.log'
            stream = logfile.open('x')
            process = subprocess.Popen(command, cwd=self.snapshot, env=env, stdout=stream, stderr=subprocess.STDOUT)
            self.running[job['name']] = (process, stream)
            job.update(status='running', pid=process.pid, out=str(out), log=str(logfile), gpu=gpu,
                       started=time.time(), command=command)
            self.event('started', **job)

    def poll(self):
        for name, (process, stream) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del self.running[name]
            job = self.jobs[name]
            marker = {'train': 'complete.json', 'collect': 'data_manifest.json',
                      'fit': 'metrics.json', 'evaluate': 'summary.json'}[job['job']]
            successful = code == 0 and (self.output(name)/marker).is_file()
            job.update(exit_code=code, finished=time.time(), status='complete' if successful else 'failed')
            self.event('finished', **job)
            # Retry in a new directory; failed evidence is never overwritten.
            if not successful and job['attempt'] <= self.args.retries:
                job['status'] = 'pending'
                self.event('retry_pending', name=name, previous_out=job['out'])

    def report(self):
        summaries = []
        for job in self.jobs.values():
            if job['job']=='evaluate' and job['status']=='complete':
                result = read(Path(job['out'])/'summary.json')
                result['out'] = job['out']
                result['checkpoint'] = job['options']['checkpoint']
                result['transitions'] = int(Path(result['checkpoint']).stem.split('_')[1])
                summaries.append(result)
        paired, identity_errors = [], []
        keyed = {(s['arm'], s['transitions'], s['seed']): s for s in summaries}
        for s in summaries:
            baseline = keyed.get((16, s['transitions'], s['seed']))
            if baseline is None or s['arm'] == 16:
                continue
            def episodes(directory):
                return {(r['scenario'], r['episode_seed']): r for r in
                        [json.loads(line) for line in (Path(directory)/'episodes.jsonl').read_text().splitlines()]}
            a, b = episodes(baseline['out']), episodes(s['out'])
            aligned = a.keys() == b.keys() and all(a[k]['initial_hash']==b[k]['initial_hash'] for k in a)
            if not aligned:
                identity_errors.append(dict(arm=s['arm'], seed=s['seed'], transitions=s['transitions']))
            paired.append(dict(arm=s['arm'], seed=s['seed'], transitions=s['transitions'],
                identical_initial_conditions=aligned,
                differences_vs_16={key: float(np.mean([b[k][key]-a[k][key] for k in a]))
                    for key in ('success','removal','path_length','wall_rate','pair_rate')}
                    if aligned else None))
        grouped = {}
        for s in summaries:
            grouped.setdefault((s['arm'], s['transitions']), []).append(s)
        aggregate = []
        for (arm, steps), rows in sorted(grouped.items()):
            # Do not silently choose a favourable retry or average duplicate seeds.
            unique = len({r['seed'] for r in rows}) == len(rows)
            full = unique and set(r['seed'] for r in rows) == set(self.args.seeds)
            metrics = {k: dict(mean=float(np.mean([r['macro'][k] for r in rows])),
                       sample_sd=float(np.std([r['macro'][k] for r in rows], ddof=1)) if len(rows)>1 else None)
                       for k in rows[0]['macro']}
            aggregate.append(dict(arm=arm, transitions=steps, seeds=[r['seed'] for r in rows],
                complete_three_seed_comparison=full and len(rows)>=3, metrics=metrics,
                validation_85_reached=bool(full and len(rows)>=3 and metrics['success']['mean']>=.85
                    and not any(e['arm']==arm and e['transitions']==steps for e in identity_errors))))
        save(self.root/'results.json', dict(evaluations=summaries, aggregate=aggregate,
            paired_differences=paired, initial_identity_errors=identity_errors,
            sealed_test_accessed=False, claim='development validation only'))
        save(self.root/'status.json', dict(updated=dt.datetime.now().isoformat(),
            running=len(self.running), max_jobs=self.args.max_jobs, jobs=self.jobs))
        lines = ['# EXP16–20 v2 实时状态', '', f'更新时间：{dt.datetime.now().isoformat(timespec="seconds")}',
                 f'并行任务：{len(self.running)} / {self.args.max_jobs}；评估仅使用开发验证集。', '',
                 '|任务|状态|尝试|', '|---|---|---|']
        lines += [f'|{n}|{j["status"]}|{j.get("attempt", 0)}|' for n, j in self.jobs.items()]
        lines += ['', '|版本|真实训练 transitions|已评估 seeds|成功率均值 ± SD|平均路径|质量清除|', '|---|---|---|---|---|---|']
        for row in aggregate:
            metric = row['metrics']['success']
            sd = 'N/A' if metric['sample_sd'] is None else f'{100*metric["sample_sd"]:.2f}%'
            lines.append(f'|{row["arm"]}|{row["transitions"]}|{row["seeds"]}|{100*metric["mean"]:.2f}% ± {sd}|'
                         f'{row["metrics"]["path_length"]["mean"]:.4f}|{100*row["metrics"]["removal"]["mean"]:.2f}%|')
        lines += ['', '不足三个训练 seed 的结果为中间结果；85% 仅是开发验证阈值，不能宣称已通过封存测试。',
                  '所有失败、模型验证失败、轨迹、checkpoint 和原始日志均保留。资源不足时等待，不终止其他实验。']
        if identity_errors:
            lines.append(f'警告：{len(identity_errors)} 个配对条件身份不一致，不得声称公平提升。')
        tmp = self.root/'REPORT.md.tmp'
        tmp.write_text('\n'.join(lines)+'\n')
        tmp.replace(self.root/'REPORT.md')

    def run(self):
        self.event('supervisor_started', pid=os.getpid())
        last_print = 0
        while True:
            self.poll()
            self.discover()
            self.launch()
            self.report()
            if time.monotonic()-last_print > 30:
                states = {state: sum(j['status']==state for j in self.jobs.values())
                          for state in ('pending','running','complete','failed','blocked')}
                print(dt.datetime.now().isoformat(timespec='seconds'), states, flush=True)
                try:
                    gpu_status = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'], text=True, timeout=5)
                    self.event('resource_sample', gpu_csv=gpu_status, load_average=os.getloadavg(),
                               disk_free_bytes=shutil.disk_usage(self.root).free, running=len(self.running))
                except subprocess.SubprocessError as error:
                    self.event('resource_monitor_error', error=repr(error))
                last_print = time.monotonic()
            if not self.running and not any(j['status']=='pending' for j in self.jobs.values()):
                missing = [f'{arm}_s{s}' for s in self.args.seeds for arm in (16, 17, 18, 20)
                           if f'{arm}_s{s}' not in self.jobs]
                save(self.root/'complete.json', dict(finished=dt.datetime.now().isoformat(),
                    missing_due_to_failed_dependencies=missing,
                    failed_or_blocked=[n for n,j in self.jobs.items() if j['status'] in ('failed','blocked')],
                    all_policy_arms_complete=not missing and all(self.complete(f'{arm}_s{s}')
                        for s in self.args.seeds for arm in (16,17,18,20))))
                self.event('supervisor_finished', missing=missing)
                break
            time.sleep(self.args.poll_seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--max-jobs', type=int, default=18)
    p.add_argument('--n-envs', type=int, default=56)
    p.add_argument('--timesteps', type=int, default=3000000)
    p.add_argument('--checkpoint-interval', type=int, default=1000000)
    p.add_argument('--model-data-steps', type=int, default=50000)
    p.add_argument('--world-budgets', nargs='+', type=int, default=[50000,150000])
    p.add_argument('--fit-epochs', type=int, default=30)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--retries', type=int, default=1)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--gpu-reserve-mb', type=int, default=4096)
    args = p.parse_args()
    args.max_jobs = max(1, min(args.max_jobs, len(set(os.sched_getaffinity(0))-{6,7})-4))
    try:
        Queue(args).run()
    except Exception as exc:
        # Preserve traceback in the live tmux pane; machine-readable fatal marker.
        root = Path(args.out)
        if root.is_dir():
            import traceback
            failure = root/f'supervisor_failure_{time.time_ns()}.json'
            save(failure, dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
