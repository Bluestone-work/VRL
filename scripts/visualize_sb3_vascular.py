"""Read-only live SB3 dashboard and recorded, frozen-model development replays."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RUNS = ROOT / 'research/runs/EXP0113_SB3_PILOT_20261015'
DEFAULT_OUT = Path('/home/wj/桌面/VRL_SB3_实时训练与3D回放_20261015')
ALGORITHMS = ('ppo', 'sac', 'td3')


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def read_jsonl(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            continue  # A writer may not yet have flushed its last record.
    return rows


def process_alive(pid, run):
    try:
        proc = Path('/proc') / str(pid)
        return (str(run).encode() in (proc / 'cmdline').read_bytes()
                and (proc / 'stat').read_text().split()[2] != 'Z')
    except OSError:
        return False


def collect_status(runs, out):
    jobs = {j['algorithm']: j for j in read_json(runs / 'jobs.json').get('jobs', [])}
    observed = read_jsonl(out / 'observed_progress.jsonl')
    histories = {algo: [r for r in observed if r['algorithm'] == algo] for algo in ALGORITHMS}
    results = []
    for algo in ALGORITHMS:
        run = runs / algo
        config = read_json(run / 'config.json')
        status = read_json(run / 'status.json')
        complete = read_json(run / 'TRAINING_COMPLETE.json')
        evaluated = read_json(run / 'RUN_COMPLETE.json')
        steps = max(status.get('environment_steps', 0), complete.get('environment_steps', 0))
        logs = read_jsonl(run / 'progress.json')
        episodes = read_jsonl(run / 'episodes.jsonl')
        completed_steps = 0
        for row in episodes:
            completed_steps += row.get('environment_steps', 0)
            row['training_environment_step'] = completed_steps
        point = dict(algorithm=algo, steps=steps, minutes=status.get('minutes', 0), captured=timestamp())
        if not histories[algo] or histories[algo][-1]['steps'] != steps:
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'observed_progress.jsonl').open('a') as f:
                f.write(json.dumps(point) + '\n')
            histories[algo].append(point)
        curve = [[r['time/time_elapsed'] / 60, r['time/total_timesteps']]
                 for r in logs if 'time/time_elapsed' in r and 'time/total_timesteps' in r]
        curve.extend([[r['minutes'], r['steps']] for r in histories[algo]])
        curve = sorted({(float(x), int(y)) for x, y in curve if x >= 0})
        job = jobs.get(algo, {})
        alive = process_alive(job.get('pid', -1), run)
        stage = '已完成训练和评估' if evaluated else '训练完成，正在评估' if complete and alive else '训练中' if alive else '进程未运行'
        last_log = logs[-1] if logs else {}
        losses = []
        for record in logs:
            x = record.get('time/total_timesteps')
            if x is None:
                continue
            for key in ('train/value_loss', 'train/critic_loss', 'train/actor_loss'):
                value = record.get(key)
                if value is not None:
                    losses.append(dict(step=x, metric=key.partition('/')[2], value=value))
        results.append(dict(algorithm=algo, steps=steps,
                            target=config.get('environment_steps_requested', 102400),
                            stage=stage, alive=alive, pid=job.get('pid'),
                            age_seconds=round(time.time() - (run / 'status.json').stat().st_mtime, 1) if (run / 'status.json').exists() else None,
                            agent_steps=status.get('agent_steps'), episodes=episodes, episode_count=len(episodes),
                            strict_success=sum(bool(r.get('strict_success')) for r in episodes),
                            mean_removal=sum(r['removal'] for r in episodes) / len(episodes) if episodes else None,
                            mean_return=sum(r['return_sum'] for r in episodes) / len(episodes) if episodes else None,
                            action_saturation=status.get('action_saturation'), progress_curve=curve,
                            losses=losses, updates=complete.get('updates', last_log.get('train/n_updates')),
                            evaluation=read_json(run / 'evaluation_summary.json'),
                            completed=bool(evaluated), device=config.get('device', 'cpu')))
    return dict(updated=timestamp(), runs=str(runs), algorithms=results,
                scope='3 机器人 · 16 动态障碍 · 一个训练种子 · 开发预实验',
                note='训练回合指标用于观察学习过程；冻结模型开发回放与最终九场景评估分别标注。')


def checkpoint_info(path):
    """Read SB3 metadata only after the zip has been completely written."""
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                return None
            data = json.loads(z.read('data'))
        return dict(path=str(path), steps=int(data['num_timesteps']), updates=int(data.get('_n_updates', 0)),
                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
        return None


def pick_checkpoint(runs, algo):
    candidates = list((runs / algo).glob('policy_steps*.zip'))
    final = runs / algo / 'final_model.zip'
    if final.exists():
        candidates.append(final)
    valid = [r for p in candidates if (r := checkpoint_info(p)) is not None]
    if valid:
        result = max(valid, key=lambda r: r['steps'])
        result['origin'] = '正式预实验冻结模型'
    else:
        smoke = ROOT / 'research/runs/EXP0113_SB3_SMOKE_20261015' / (algo + '_v4') / 'final_model.zip'
        result = checkpoint_info(smoke)
        if result is None:
            return None
        result['origin'] = '512 步接口验证模型，独立于当前正式预实验'
    result['algorithm'] = algo
    return result


def record_replay(checkpoint, out):
    import numpy as np
    import torch
    from scripts.train_sb3_vascular import ALGORITHMS as MODELS, VascularSB3Env
    from scripts.export_vrl_v3_all_gifs import geometry

    torch.set_num_threads(1)
    algo = checkpoint['algorithm']
    stem = algo + '_' + str(checkpoint['steps']) + '_' + checkpoint['sha256'][:10]
    destination = out / 'replays'
    destination.mkdir(parents=True, exist_ok=True)
    cached = destination / (stem + '.npz')
    if cached.exists():
        with np.load(cached) as archive:
            metadata = json.loads(str(archive['metadata']))
            geo = json.loads(str(archive['geometry']))
            arrays = {k: archive[k] for k in archive.files if k not in ('metadata', 'geometry') and not k.startswith('full_')}
        if metadata['sha256'] == checkpoint['sha256']:
            atomic_json(destination / (algo + '.json'), replay_payload(arrays, metadata, geo))
            metadata.update(json='replays/' + algo + '.json', npz='replays/' + stem + '.npz')
            return metadata
    model = MODELS[algo].load(checkpoint['path'], device='cpu')
    config = read_json(Path(checkpoint['path']).parent / 'config.json')
    if 'scenario_families' in config:
        from scripts.train_sb3_curriculum import CurriculumVascularEnv
        env = CurriculumVascularEnv(clusters=3, curriculum=False,
                                    prediction=config['prediction'], shaping=config['shaping'],
                                    reward_mode=config.get('reward_mode', 'potential'),
                                    uniform_only=config.get('uniform_only', False))
        options = dict(anatomy='mca_m1_lvo', scene_seed=2600000020, flow=.05, latency=2,
                       difficulty=2,
                       scenario_family='uniform' if config.get('uniform_only', False) else 'route_encounter')
    else:
        env = VascularSB3Env(clusters=3, dynamic_obstacle_count=16)
        options = dict(anatomy='mca_m1_lvo', scene_seed=2600000000, flow=.025, latency=1)
    try:
        obs, _ = env.reset(options=options)
        geo = geometry(env.ep)
        record = dict(t=[], pos=[], particles=[], mass=[], removal=[], wall=[], active=[], targets=[], hold=[], command=[], reward=[], images=[])
        if 'obstacles' in obs:
            record.update(forecasts=[], forecast_valid=[])
        total_wall = 0.
        def capture():
            ep = env.ep
            est, targets, hold, _ = env.current
            record['t'].append(float(ep.env.elapsed_s))
            record['pos'].append(ep.env.positions_mm[:3].copy())
            record['particles'].append(ep.env.positions_mm[3:].copy())
            record['mass'].append(ep.env.masses.copy())
            record['removal'].append(1. - float(ep.env.masses.sum()) / ep.initial)
            record['wall'].append(total_wall)
            record['active'].append(ep.env.active[:3].copy())
            record['targets'].append(np.asarray(targets).copy())
            record['hold'].append(np.asarray(hold).copy())
            record['command'].append(ep.sent_world.copy())
            record['reward'].append(env.return_sum)
            record['images'].append(np.clip((obs['image'][:, :2].astype(float) + .3) / 2.3 * 255, 0, 255).astype('uint8'))
            if 'obstacles' in obs:
                times = np.array([0., .2, .5, 1.])
                frames = env.current[3]
                local = np.concatenate([obs['obstacles'][..., None, :3],
                                        obs['obstacles'][..., 6:15].reshape(3, 8, 3, 3)], axis=2) * 2.
                age = max(float(ep.env.elapsed_s) - float(getattr(est, 'frame_time_s', ep.env.elapsed_s)), 0.)
                world = np.einsum('nij,nkhi->nkhj', frames, local)
                world += est.pos[:, None, None, :] + est.vel[:, None, None, :] * (times[None, None, :, None] + age)
                record['forecasts'].append(world)
                record['forecast_valid'].append(obs['obstacle_valid'].copy())
        capture()
        full = dict(t=[0.], pos=[env.ep.env.positions_mm[:3].copy()], mass=[env.ep.env.masses.copy()],
                    removal=[0.], wall=[0.])
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_wall += float(np.sum(env.ep.info['wall_contact_s']))
            full['t'].append(float(env.ep.env.elapsed_s))
            full['pos'].append(env.ep.env.positions_mm[:3].copy())
            full['mass'].append(env.ep.env.masses.copy())
            full['removal'].append(1. - float(env.ep.env.masses.sum()) / env.ep.initial)
            full['wall'].append(total_wall)
            if env.ep.steps % 10 == 0 or done or truncated:
                capture()
            if done or truncated:
                metrics = info['episode_metrics']
                break
        assert abs(metrics['removal'] - record['removal'][-1]) < 1e-7
        assert abs(metrics['wall_contact_s'] - total_wall) < 1e-5
        assert metrics['scenario_hash'] == env.ep.manifest['scenario_hash']
        arrays = {k: np.asarray(v) for k, v in record.items()}
        metadata = dict(checkpoint, created=timestamp(), deterministic=True,
                        kind='冻结策略的独立开发回放，非训练中的实时轨迹', scene=options,
                        scenario_hash=metrics['scenario_hash'], metrics=metrics,
                        terminal=True, sample_interval_steps=10, physical_steps=env.ep.steps)
        np.savez_compressed(destination / (stem + '.npz'), **arrays,
                            full_t=np.asarray(full['t']), full_pos=np.asarray(full['pos']),
                            full_mass=np.asarray(full['mass']), full_wall=np.asarray(full['wall']),
                            full_removal=np.asarray(full['removal']),
                            geometry=json.dumps(geo), metadata=json.dumps(metadata, ensure_ascii=False))
        atomic_json(destination / (algo + '.json'), replay_payload(arrays, metadata, geo))
        metadata.update(json='replays/' + algo + '.json', npz='replays/' + stem + '.npz')
        print(json.dumps(dict(replay_ready=algo, checkpoint_steps=checkpoint['steps'], strict_success=metrics['strict_success'], removal=metrics['removal'])), flush=True)
        return metadata
    finally:
        env.close()
        del model


def replay_payload(arrays, metadata, geo):
    payload = {k: (v.round(5) if v.dtype.kind == 'f' else v).tolist()
               for k, v in arrays.items() if k != 'images'}
    payload['images'] = [base64.b64encode(im.tobytes()).decode('ascii') for im in arrays['images']]
    payload.update(metadata=metadata, geometry=geo)
    return payload


def plot_training(snapshot, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'ppo': '#007d8a', 'sac': '#de9b26', 'td3': '#7757c7'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    for row in snapshot['algorithms']:
        algo = row['algorithm']
        if row['progress_curve']:
            x, y = zip(*row['progress_curve'])
            axes[0, 0].plot(x, y, color=colors[algo], label=algo.upper())
        episodes = row['episodes']
        x = [r['training_environment_step'] for r in episodes]
        for ax, metric, factor in [(axes[0, 1], 'return_sum', 1), (axes[1, 0], 'removal', 100), (axes[1, 1], 'wall_contact_s', 1)]:
            ax.plot(x, [r[metric] * factor for r in episodes], '.-', color=colors[algo], label=algo.upper())
    titles = ['Training progress', 'Completed training episodes: return', 'Completed training episodes: removal (%)', 'Completed training episodes: wall contact (robot s)']
    for index, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Wall-clock minutes' if index == 0 else 'Cumulative completed environment steps')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('SB3 vascular pilot | 3 robots, 16 obstacles | ' + snapshot['updated'], fontsize=11)
    fig.tight_layout()
    temporary = out / 'training_curves.tmp.png'
    fig.savefig(temporary)
    plt.close(fig)
    temporary.replace(out / 'training_curves.png')


class Dashboard:
    def __init__(self, runs, out):
        self.runs, self.out = runs, out
        self.lock = threading.Lock()
        self.snapshot = {}
        self.manifest = read_json(out / 'manifest.json', {'replays': {}, 'errors': {}})
        self.last_plot = 0
        self.last_sample = 0

    def refresh(self, plot=False):
        with self.lock:
            self.snapshot = collect_status(self.runs, self.out)
            atomic_json(self.out / 'live_status.json', self.snapshot)
        if plot or time.monotonic() - self.last_plot > 60:
            plot_training(self.snapshot, self.out)
            self.last_plot = time.monotonic()
        self.write_page()

    def write_page(self):
        template = (ROOT / 'scripts/sb3_dashboard.html').read_text()
        embedded = {algo: read_json(self.out / r['json']) for algo, r in self.manifest['replays'].items()}
        data = dict(status=self.snapshot, manifest=self.manifest, replays=embedded)
        page = template.replace('__EMBEDDED_DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/'))
        temporary = self.out / 'index.html.tmp'
        temporary.write_text(page, encoding='utf-8')
        temporary.replace(self.out / 'index.html')

    def update_replays(self):
        for algo in ALGORITHMS:
            checkpoint = pick_checkpoint(self.runs, algo)
            if not checkpoint:
                continue
            current = self.manifest['replays'].get(algo, {})
            if current.get('sha256') == checkpoint['sha256']:
                continue
            try:
                metadata = record_replay(checkpoint, self.out)
                with self.lock:
                    self.manifest['replays'][algo] = metadata
                    self.manifest['errors'].pop(algo, None)
            except Exception:
                error = traceback.format_exc()
                with self.lock:
                    self.manifest['errors'][algo] = error
                print(error, flush=True)
            atomic_json(self.out / 'manifest.json', self.manifest)
            self.refresh()

    def loop(self):
        while True:
            try:
                self.refresh()
                self.update_replays()
            except Exception:
                print(traceback.format_exc(), flush=True)
            time.sleep(15)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, dashboard, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, directory=str(dashboard.out), **kwargs)

    def do_GET(self):
        if self.path.split('?')[0] == '/api/status':
            with self.dashboard.lock:
                self.dashboard.snapshot = collect_status(self.dashboard.runs, self.dashboard.out)
                atomic_json(self.dashboard.out / 'live_status.json', self.dashboard.snapshot)
                body = json.dumps(dict(status=self.dashboard.snapshot, manifest=self.dashboard.manifest), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--serve', action='store_true')
    p.add_argument('--port', type=int, default=8873)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    os.nice(5)
    a.out.mkdir(parents=True, exist_ok=True)
    dashboard = Dashboard(a.runs.resolve(), a.out.resolve())
    dashboard.refresh(plot=True)
    if a.serve:
        server = ThreadingHTTPServer(('127.0.0.1', a.port), partial(Handler, dashboard=dashboard))
        worker = threading.Thread(target=dashboard.loop, daemon=True)
        worker.start()
        atomic_json(a.out / 'service.json', dict(pid=os.getpid(), url=f'http://127.0.0.1:{a.port}/', started=timestamp()))
        print(f'http://127.0.0.1:{a.port}/', flush=True)
        server.serve_forever()
    else:
        dashboard.update_replays()
        dashboard.refresh(plot=True)


if __name__ == '__main__':
    main()
