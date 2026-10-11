"""EXP0090: collect predictor training / validation data (marl.dyn_predictor, behaviour marl.dyn_select.Behaviour).

Per control step and active cluster one row: deployable history fields exactly as DynFeatures pushes them (image
velocity, sent world command of the previous step, delayed position estimate, frame age, hold), the local frame F_t,
the context vector, the intended world command of this step, and - as supervision targets only - the true position
before the step, the true flow at the cluster and the effective response gain (marl.lysis_matrix labels).
Domain: inlet speed {0.025, 0.05, 0.1} mm/s, latency U{1,2,3}, v5 strength 0 (p 0.3) or U[0, 1.25], N ~ U{1,2,3}.
usage: collect_dyn_data.py --out DIR --split train|val_train|val_heldout --episodes 360 --workers 20
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

SEED_BASE = {'train': 2121000000, 'val_train': 2122000000, 'val_heldout': 2123000000}
OFFSET = {'v': 0}     # independent predictor seeds: train-split scene seeds shifted by --seed-offset (B: 5e6, C: 1e7)


def run(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    split, k, out, offset = job
    from marl.deployable_sensing import DeployableConfig
    from marl.dyn_predictor import DynFeatures
    from marl.dyn_select import Behaviour
    from marl.lysis_matrix import training_dynamics_labels
    from scripts.benchmark_lysis import LysisEpisode
    sp = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']
    anats = sp['held_out'] if split == 'val_heldout' else sp['train']
    base = SEED_BASE[split]+offset
    rng = np.random.default_rng(base+k)
    for attempt in range(20):
        seed = base+1000*k+attempt
        an = anats[rng.integers(len(anats))]; n = int(rng.integers(1, 4))
        s = 0. if rng.random() < .3 else float(rng.uniform(0, 1.25)); flow = float(rng.choice([.025, .05, .1]))
        lat = int(rng.integers(1, 4))
        try:
            ep = LysisEpisode(n, an, seed, sense_cfg=DeployableConfig(latency_steps=lat), variation=s if s > 0 else None,
                              flow_inlet_mm_s=flow)
            break
        except (ValueError, RuntimeError):
            continue
    else:
        return None
    feat = DynFeatures(ep); beh = Behaviour(ep, np.random.default_rng(seed+1)); rows = []
    try:
        while True:
            est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
            feat.push(ep, est, hold)
            lab = training_dynamics_labels(ep); truth = ep.env.positions_mm[:n].astype(float).copy()
            F = ep.ctl.frames(est); active = ep.env.active[:n].copy()
            ctxs = [feat.context(ep, est, tgt, i, F[i]) for i in range(n)]
            local = beh(ep, est, tgt, rule, hold)
            for i in range(n):
                v, c, p, age, h = feat.buf[i][-1]
                rows.append(np.concatenate([[i, float(active[i] and est.active[i]), float(tgt[i] >= 0)], v, c, p, [age, h],
                                            F[i].ravel(), ctxs[i], beh.intended[i], truth[i], lab[i]]).astype(np.float32))
            done, _ = ep.step(est, local, hold)
            if done:
                break
    finally:
        ep.close()
    R = np.stack(rows)
    meta = dict(split=split, k=k, seed=seed, anatomy=an, n=n, s=s, flow=flow, latency=lat, steps=len(R)//n)
    np.savez_compressed(Path(out)/f'{split}_{k:04d}.npz', rows=R, meta=json.dumps(meta))
    return meta


# row layout (columns)
COL = dict(cluster=0, active=1, live=2, vel=slice(3, 6), sent=slice(6, 9), pos=slice(9, 12), age=12, hold=13,
           F=slice(14, 23), ctx=slice(23, 42), intended=slice(42, 45), truth=slice(45, 48), flow=slice(48, 51), gain=51)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--split', required=True, choices=list(SEED_BASE))
    ap.add_argument('--episodes', type=int, default=360); ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--seed-offset', type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.split, k, str(a.out), a.seed_offset) for k in range(a.episodes) if not (a.out/f'{a.split}_{k:04d}.npz').exists()]
    metas = []
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=4) as pool:
        for j, m in enumerate(pool.imap_unordered(run, jobs)):
            if m:
                metas.append(m)
            if j % 20 == 0:
                print(f'{j+1}/{len(jobs)}', flush=True)
    with (a.out/f'{a.split}_meta.jsonl').open('a') as f:
        for m in metas:
            f.write(json.dumps(m)+'\n')


if __name__ == '__main__':
    main()
