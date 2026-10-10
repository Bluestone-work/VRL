"""Bounded, isolated, paired EXP0073 classical/paper-adaptation screening."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

PANELS = {'low_delay': (.05, 1, 0.), 'high_delay': (.05, 3, 0.),
          'strong_flow': (.1, 2, .625), 'variable_response': (.05, 2, 1.25)}
ALL_PANELS = dict(PANELS, moderate_delay=(.05, 2, 0.), ood_flow_delay=(.075, 3, .625))

def child(spec, connection, policy=None, lateral_scale=None, lateral_near_radius=None):
    try:
        from scripts.benchmark_lysis import (LysisEpisode, SettleGuard,
                                             AdaptiveSettleGuard, WallGuard, METHODS)
        from marl.deployable_sensing import DeployableConfig
        method, panel, job = spec
        _, _, split, suite, anatomy, n, seed, flow, latency, variation = job
        kw = dict(flow_inlet_mm_s=flow, sense_cfg=DeployableConfig(latency_steps=latency),
                  variation=variation if variation > 0 else None)
        if method in ('pac_nmpc', 'stpg', 'oracle_flow'):
            row, ep = METHODS[method](n, anatomy, seed, **kw)
        else:
            ep = LysisEpisode(n, anatomy, seed, **kw)
            if policy is not None and method == 'sched_settle':
                from scripts.train_sched_settle import SchedController
                controller = SchedController(ep, Path(policy))
            elif policy is not None:
                from scripts.train_lysis_nav import NavController
                controller = NavController(ep, Path(policy))
                if lateral_scale is not None:
                    controller.cfg['lateral_residual_scale'] = lateral_scale
                if lateral_near_radius is not None:
                    controller.cfg['lateral_near_radius'] = lateral_near_radius
            else:
                from scripts.benchmark_lysis import SwitchSettle
                controller = {'settle': SettleGuard, 'adaptive_settle': AdaptiveSettleGuard,
                              'no_settle': WallGuard,
                              'switch_settle_015': lambda e: SwitchSettle(e, .15),
                              'switch_settle_025': lambda e: SwitchSettle(e, .25)}[method](ep)
            while True:
                est = ep.observe(); tgt = ep.plan_targets_now(est)
                rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
                done, _ = ep.step(est, controller(ep, est, tgt, rule, hold), hold)
                if done: break
            row = ep.row(method)
        row.update(panel=panel, split=split, suite=suite, flow_inlet_mm_s=flow,
                   latency_steps=latency, variation=variation)
        if policy is not None:
            row.update(checkpoint=str(policy), lateral_scale_intervention=lateral_scale,
                       lateral_near_radius_intervention=lateral_near_radius)
        row['T90_300'] = row['t90_s'] if row['t90_s'] is not None else 300.
        ep.close(); connection.send(row)
    except Exception:
        connection.send(dict(error=traceback.format_exc(), spec=spec))
    finally:
        connection.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--methods', default='settle,adaptive_settle,no_settle,stpg,pac_nmpc')
    p.add_argument('--policy', type=Path)
    p.add_argument('--lateral-scale', type=float)
    p.add_argument('--lateral-near-radius', type=float)
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--panels', default=','.join(PANELS))
    a = p.parse_args()
    source = Path('research/validation/EXP0073_LONG1M_20261009/nav_tf_v3_s7101/manifest.json')
    jobs = json.loads(source.read_text())['jobs']
    panels = {name: ALL_PANELS[name] for name in a.panels.split(',')}
    specs = [(m, panel, j) for panel, condition in panels.items() for j in jobs
             if tuple(j[-3:]) == condition and (not a.train_only or j[2]=='seen_topology')
             for m in a.methods.split(',')]
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'manifest.json').write_text(json.dumps(dict(specs=specs, panels=panels,
        timeout_s=a.timeout, workers=a.workers, source=str(source),
        train_only=a.train_only, policy=str(a.policy) if a.policy else None,
        checkpoint_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest() if a.policy else None,
        lateral_scale_intervention=a.lateral_scale,
        lateral_near_radius_intervention=a.lateral_near_radius,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
        ['scripts/evaluate_hard_baselines.py','scripts/benchmark_lysis.py','marl/lysis_baselines.py','marl/sched_settle.py','scripts/train_sched_settle.py']},
        scope='development screening; original repository defaults, no tuning; paper methods are adaptations'), indent=2))
    ctx = mp.get_context('fork'); running = {}; cursor = completed = 0
    with (a.out/'episodes.jsonl').open('w') as out, (a.out/'failures.jsonl').open('w') as failures:
        while cursor < len(specs) or running:
            while cursor < len(specs) and len(running) < a.workers:
                spec = specs[cursor]; cursor += 1
                receiver, sender = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=child, args=(spec, sender, a.policy, a.lateral_scale, a.lateral_near_radius)); proc.start(); sender.close()
                running[proc.pid] = (proc, receiver, spec, time.monotonic())
            for pid, (proc, receiver, spec, started) in list(running.items()):
                row = None
                if receiver.poll():
                    try: row = receiver.recv()
                    except EOFError: pass
                expired = time.monotonic()-started > a.timeout
                if row is None and proc.is_alive() and not expired: continue
                if row is None:
                    row = dict(spec=spec, error='wall-clock timeout' if expired else 'worker exited without result',
                               exitcode=proc.exitcode, infrastructure_failure=True)
                if proc.is_alive():
                    proc.join(timeout=.1)
                    if proc.is_alive(): proc.terminate()
                proc.join(); receiver.close(); del running[pid]
                target = failures if 'error' in row else out
                target.write(json.dumps(row)+'\n'); target.flush(); completed += 1
                if completed % 10 == 0 or completed == len(specs):
                    print(f'{completed}/{len(specs)} elapsed workers={len(running)}', flush=True)
            time.sleep(.1)

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    main()
