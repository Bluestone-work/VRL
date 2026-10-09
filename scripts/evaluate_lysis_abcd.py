"""Paired EXP0062 screening. Metrics come unchanged from LysisEpisode.row."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import numpy as np

RUNS = dict(A='EXP0062_A_fixed_run', B='EXP0062_B_multiflow_run',
            C='EXP0062_C_history_run', D='EXP0062_D_history_aux_run')
SUITES = dict(in025=(.025, 1), in050=(.05, 1), unseen0375=(.0375, 1),
              stress100=(.1, 1), unseen_combination=(.0375, 3))


def job(spec):
    import torch
    torch.set_num_threads(1)
    from scripts.benchmark_lysis import LysisEpisode, SettleGuard, WallGuard
    from scripts.train_lysis_nav import NavController
    from marl.deployable_sensing import DeployableConfig
    from marl.physio_variation import Variation, VENOUS
    method, suite, anatomy, n, seed = spec
    flow, latency = SUITES[suite]
    ep = None
    try:
        ep = LysisEpisode(n, anatomy, seed, flow_inlet_mm_s=flow,
                         sense_cfg=DeployableConfig(latency_steps=latency))
        ep.var = Variation(ep, 0., np.random.default_rng([seed, 5150]))
        ep.var.amp = .12 if anatomy in VENOUS else .4
        initial_hash = hashlib.sha256(ep.env.positions_mm.tobytes()+ep.env.masses.tobytes()).hexdigest()
        if method in RUNS or method == 'A_legacy':
            arm = 'A' if method == 'A_legacy' else method
            low = NavController(ep, Path('research/runs')/RUNS[arm]/'policy.pt')
            if method == 'A_legacy':
                low.cfg['prior_residual_scale'] = .5
        else:
            low = SettleGuard(ep) if method == 'settle' else WallGuard(ep)
        entered = {}; last_near = {}; departures = 0
        near_time = eligible_time = command_sum = drift_sum = 0.
        near_count = drift_count = 0
        previous = None
        first_entry = None
        while True:
            est = ep.observe(); tgt = ep.plan_targets_now(est)
            rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
            local = low(ep, est, tgt, rule, hold)
            near = np.zeros(n, bool)
            now = float(ep.env.elapsed_s)
            for i, t in enumerate(tgt):
                if t < 0 or not est.active[i]:
                    continue
                key = (i, int(t))
                near[i] = np.linalg.norm(ep.env.clot_positions_mm[t]-est.pos[i]) < .3
                if near[i] and key not in entered:
                    entered[key] = now
                    if first_entry is None: first_entry = now
                if last_near.get(key, False) and not near[i]: departures += 1
                last_near[key] = bool(near[i])
            if previous is not None:
                old_pos, old_near, old_active, stamp = previous
                valid = old_near & old_active & est.active
                if valid.any():
                    drift_sum += float(np.linalg.norm(est.pos[valid]-old_pos[valid], axis=1).sum())/max(now-stamp, 1e-9)
                    drift_count += int(valid.sum())
            previous = (est.pos.copy(), near.copy(), est.active.copy(), now)
            done, _ = ep.step(est, local, hold)
            dt = float(ep.env.elapsed_s)-now
            eligible_time += float(((np.asarray(tgt)>=0)&est.active).sum())*dt
            near_time += float(near.sum())*dt
            command_sum += float(np.linalg.norm(ep.prev_local[near], axis=1).sum())
            near_count += int(near.sum())
            if done: break
        row = ep.row(method)
        row.update(suite=suite, flow_inlet_mm_s=flow, latency_steps=latency,
                   actual_healthy_mean_mm_s=flow*ep.env.episode_flow_multiplier/2.,
                   training_seed=6201 if method in RUNS or method=='A_legacy' else None,
                   initial_state_hash=initial_hash, first_neighborhood_entry_s=first_entry,
                   neighborhood_residence_ratio=near_time/max(eligible_time,1e-9),
                   departures_after_entry=departures,
                   near_command_norm=command_sum/near_count if near_count else None,
                   observed_near_motion_mm_s=drift_sum/drift_count if drift_count else None,
                   failure_stage=('success' if row['strict_success'] else 'safety_or_exit' if row['task_success'] or row['lost']
                                  else 'approach' if first_entry is None else 'residence_or_clearance'))
        return row
    except Exception:
        return dict(method=method, suite=suite, anatomy=anatomy, clusters=n, seed=seed, error=traceback.format_exc())
    finally:
        if ep is not None: ep.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--suites',default=','.join(SUITES))
    p.add_argument('--methods',default='A,B,C,D,settle,no_settle,A_legacy')
    p.add_argument('--anatomies',default='all')
    p.add_argument('--clusters',default='1,2,3')
    a=p.parse_args()
    order=json.loads(Path('configs/evaluation_splits.json').read_text())['anatomy_order']
    anatomies=order if a.anatomies=='all' else a.anatomies.split(',')
    # New development validation seeds, disjoint from 2.1e9 training scenes.
    jobs=[(m,s,an,int(n),2700000000+order.index(an)*100000+11)
          for s in a.suites.split(',') for an in anatomies for n in a.clusters.split(',') for m in a.methods.split(',')]
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=dict(jobs=jobs, suites=SUITES, horizon_s=300, neighborhood_mm=.3,
                  label='development screening; one scene per anatomy/N/suite',
                  checkpoint_sha256={k:hashlib.sha256((Path('research/runs')/v/'policy.pt').read_bytes()).hexdigest() for k,v in RUNS.items()},
                  source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                 ['scripts/evaluate_lysis_abcd.py','scripts/train_lysis_nav.py','marl/lysis_abcd.py']})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    os.environ['OMP_NUM_THREADS']='1'
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=20) as pool, (a.out/'episodes.jsonl').open('w') as f:
        for i,row in enumerate(pool.imap_unordered(job,jobs),1):
            f.write(json.dumps(row)+'\n'); f.flush()
            print(f'{i}/{len(jobs)} {row["method"]} {row["suite"]} '+('ERROR' if 'error' in row else str(row['strict_success'])),flush=True)


if __name__=='__main__': main()
