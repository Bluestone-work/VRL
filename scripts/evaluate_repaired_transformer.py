"""Paired development evaluation for the repaired Transformer and stateful classical baselines."""
import argparse, hashlib, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np

def job(spec):
    method, anatomy, n, seed, flow, latency, ckpt = spec
    import torch
    torch.set_num_threads(1)
    from scripts.benchmark_lysis import LysisEpisode, SettleGuard, AdaptiveSettleGuard, WallGuard
    from scripts.train_lysis_nav import NavController
    from marl.deployable_sensing import DeployableConfig
    ep = LysisEpisode(n, anatomy, seed, flow_inlet_mm_s=flow,
                      sense_cfg=DeployableConfig(latency_steps=latency))
    try:
        if method == 'transformer':
            controller = NavController(ep, Path(ckpt))
        elif method == 'adaptive_settle':
            controller = AdaptiveSettleGuard(ep)
        elif method == 'settle':
            controller = SettleGuard(ep)
        else:
            controller = WallGuard(ep)
        entered, last_near = {}, {}; departures = 0; first_entry = None
        near_time = eligible_time = command_sum = drift_sum = 0.0
        near_count = drift_count = 0; previous = None
        while True:
            est = ep.observe(); tgt = ep.plan_targets_now(est)
            rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
            local = controller(ep, est, tgt, rule, hold)
            near = np.zeros(n, bool); now = float(ep.env.elapsed_s)
            for i, t in enumerate(tgt):
                if t < 0 or not est.active[i]: continue
                key = (i, int(t)); near[i] = np.linalg.norm(ep.env.clot_positions_mm[t]-est.pos[i]) < .3
                if near[i] and key not in entered:
                    entered[key] = now
                    if first_entry is None: first_entry = now
                if last_near.get(key, False) and not near[i]: departures += 1
                last_near[key] = bool(near[i])
            if previous is not None:
                old_pos, old_near, old_active, stamp = previous
                valid = old_near & old_active & est.active
                if valid.any():
                    drift_sum += float(np.linalg.norm(est.pos[valid]-old_pos[valid], axis=1).sum()) / max(now-stamp, 1e-9)
                    drift_count += int(valid.sum())
            previous = (est.pos.copy(), near.copy(), est.active.copy(), now)
            done, _ = ep.step(est, local, hold)
            dt = float(ep.env.elapsed_s) - now
            eligible_time += float(((np.asarray(tgt) >= 0) & est.active).sum()) * dt
            near_time += float(near.sum()) * dt
            command_sum += float(np.linalg.norm(ep.prev_local[near], axis=1).sum()); near_count += int(near.sum())
            if done: break
        row = ep.row(method)
        row.update(flow_inlet_mm_s=flow, latency_steps=latency,
                   first_neighborhood_entry_s=first_entry,
                   neighborhood_residence_ratio=near_time/max(eligible_time, 1e-9),
                   departures_after_entry=departures,
                   near_command_norm=command_sum/near_count if near_count else None,
                   observed_near_motion_mm_s=drift_sum/drift_count if drift_count else None,
                   failure_stage=('success' if row['strict_success'] else 'safety_or_exit' if row['task_success'] or row['lost'] else 'approach' if first_entry is None else 'residence_or_clearance'))
        row['T90_300'] = row['t90_s'] if row['t90_s'] is not None else 300.0
        return row
    except Exception as exc:
        return {'method': method, 'anatomy': anatomy, 'clusters': n, 'seed': seed, 'error': repr(exc)}
    finally:
        ep.close()

def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--flow', type=float, default=.05); p.add_argument('--latencies', default='1,2,3')
    p.add_argument('--methods', default='transformer,adaptive_settle,settle,no_settle')
    a = p.parse_args(); order = json.loads(Path('configs/evaluation_splits.json').read_text())['anatomy_order']
    # Development-only offset, disjoint from training scenes and the retired 2700000000 test range.
    jobs = [(m, an, n, 2600000000 + order.index(an)*100000 + 20, a.flow, int(lat), str(a.checkpoint))
            for lat in a.latencies.split(',') for an in order for n in (1,2,3) for m in a.methods.split(',')]
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(jobs=jobs, flow=a.flow, latencies=a.latencies, scene_seed_offset=20,
                    purpose='repaired Transformer paired development evaluation',
                    checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=20) as pool, (a.out/'episodes.jsonl').open('w') as out:
        for i, row in enumerate(pool.imap_unordered(job, jobs), 1):
            out.write(json.dumps(row)+'\n'); out.flush(); print(f'{i}/{len(jobs)} {row.get("method")} {row.get("latency_steps")} {row.get("strict_success", "ERR")}', flush=True)

if __name__ == '__main__': main()
