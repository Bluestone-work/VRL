"""Evaluate long-run lysis navigation checkpoints on paired seen/unseen splits.

Seen means a topology family present during training (new scene seeds). Unseen
means a held-out anatomy/topology family. Flow suites are also separated so
interpolation and intermediate-flow generalisation are reported explicitly.
"""
from __future__ import annotations

import argparse, hashlib, json, multiprocessing as mp, os, traceback
from pathlib import Path
import numpy as np

SPLITS = json.loads(Path('configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']
ORDER = json.loads(Path('configs/evaluation_splits.json').read_text())['anatomy_order']
SUITES = {
    'seen_flow': ((.025, .05, .1), (1, 2, 3)),
    'unseen_flow': ((.0375, .075), (2, 3)),
}

def job(spec):
    import torch
    torch.set_num_threads(1)
    from scripts.benchmark_lysis import LysisEpisode
    from scripts.train_lysis_nav import NavController
    from marl.deployable_sensing import DeployableConfig
    from marl.lysis_baselines import SwitchableTPG
    ckpt, method, split, suite, anatomy, n, seed, flow, latency, variation = spec
    ep = None
    try:
        ep = LysisEpisode(int(n), anatomy, int(seed), flow_inlet_mm_s=float(flow),
                         sense_cfg=DeployableConfig(latency_steps=int(latency)),
                         variation=float(variation) if variation > 0 else None)
        ctl = NavController(ep, Path(ckpt))
        rolling = None
        if method == 'belief_rolling_tpg' and int(n) > 1:
            rolling = SwitchableTPG(ep.coord, period_s=2.)
        while True:
            est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est)
            if rolling is not None:
                rolling.update(ep.env.elapsed_s); hold = rolling.c.gate(est.pos, est.active)
            else:
                hold = ep.hold(est)
            local = ctl(ep, est, tgt, rule, hold)
            done, _ = ep.step(est, local, hold)
            if done: break
        row = ep.row(method)
        row.update(split=split, suite=suite, flow_inlet_mm_s=float(flow),
                   latency_steps=int(latency), variation=float(variation),
                   topology_seen=(split == 'seen_topology'), checkpoint=str(ckpt))
        return row
    except Exception:
        return dict(method=method, split=split, suite=suite, anatomy=anatomy,
                    clusters=int(n), seed=int(seed), flow_inlet_mm_s=float(flow),
                    latency_steps=int(latency), variation=float(variation),
                    error=traceback.format_exc())
    finally:
        if ep is not None: ep.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--method',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--episodes-per-cell',type=int,default=1)
    ap.add_argument('--variation',default='0,0.625,1.25')
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    variations=[float(x) for x in a.variation.split(',')]
    jobs=[]
    # Seed ranges are disjoint from training (1.8e9) and from the old EXP0072
    # development evaluation (2.1e9).
    for split, anatomies, base in [('seen_topology',SPLITS['train'],2400000000),
                                   ('unseen_topology',SPLITS['held_out'],2500000000)]:
        for suite,(flows,lats) in SUITES.items():
            for ai,an in enumerate(anatomies):
                for n in (1,2,3):
                    for fi,flow in enumerate(flows):
                        for li,lat in enumerate(lats):
                            for vi,var in enumerate(variations):
                                seed=base+ORDER.index(an)*100000+fi*10000+li*1000+vi*100+n*10
                                for rep in range(a.episodes_per_cell):
                                    jobs.append((str(a.policy),a.method,split,suite,an,n,seed+rep,flow,lat,var))
    manifest=dict(policy=str(a.policy), method=a.method, jobs=jobs,
                  training_anatomies=SPLITS['train'], heldout_anatomies=SPLITS['held_out'],
                  suites=SUITES, variations=variations,
                  scope='paired seen-topology/new-scene and unseen-topology evaluation; no sealed test access',
                  checkpoint_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    os.environ['OMP_NUM_THREADS']='1'
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=8) as pool, (a.out/'episodes.jsonl').open('w') as f:
        for i,row in enumerate(pool.imap_unordered(job,jobs),1):
            f.write(json.dumps(row)+'\n'); f.flush()
            if i % 20 == 0 or i == len(jobs): print(f'{i}/{len(jobs)}',flush=True)
    print(a.out)

if __name__=='__main__': main()
