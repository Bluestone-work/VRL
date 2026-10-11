"""Paired V5 ablation: nav_tf_v3 with WallGuard and spacing shield disabled."""
from __future__ import annotations
import json, multiprocessing as mp
import argparse
from pathlib import Path
import torch

CKPT = 'research/runs/EXP0073_LONG1M_20261009/nav_tf_v3_s7101/policy.pt'
ROOT = Path('research/validation/V5_PHYSIO_20261008')

VARIANT = 'none'

def job(spec):
    row, out_path = spec
    from scripts.benchmark_lysis import LysisEpisode
    from scripts.train_lysis_nav import NavController
    n, anatomy, seed = int(row['clusters']), row['anatomy'], int(row['seed'])
    s = float(row['variation']['strength']) if isinstance(row.get('variation'), dict) else 1.0
    ep = LysisEpisode(n, anatomy, seed, variation=s if s > 0 else None)
    ctl = NavController(ep, Path(CKPT))
    if VARIANT in ('no_guard', 'none'):
        ctl.guard = lambda _ep, _est, _tgt, command, _hold: command
    if VARIANT in ('no_shield', 'none'):
        ep.shield = None
    try:
        while True:
            est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est)
            hold = ep.hold(est)
            local = ctl(ep, est, tgt, rule, hold)
            done, _ = ep.step(est, local, hold)
            if done: break
        z = ep.row('nav_tf_v3_no_wallguard_no_spacing_shield')
        z['paired_method'] = 'nav_tf_v3'; z['safety_variant'] = VARIANT
        return z
    finally:
        ep.close()

def main():
    global VARIANT
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', choices=('no_guard','no_shield','none'), default='none'); a = ap.parse_args(); VARIANT = a.variant
    src = [json.loads(x) for x in (ROOT/'s1_nav_tf_v3.jsonl').read_text().splitlines() if x.strip()]
    jobs = [(r, None) for r in src]
    out = ROOT/f's1_nav_tf_v3_{VARIANT}.jsonl'
    with mp.get_context('spawn').Pool(18, maxtasksperchild=4) as pool, open(out,'w') as f:
        for i, z in enumerate(pool.imap_unordered(job, jobs), 1):
            f.write(json.dumps(z)+'\n'); f.flush()
            if i % 20 == 0: print(i, flush=True)

if __name__ == '__main__': main()
