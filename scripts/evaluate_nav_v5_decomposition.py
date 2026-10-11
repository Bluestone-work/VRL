"""Paired V5 decomposition for nav_tf_v3 or an auxiliary-head checkpoint.

The episode registry is copied from the registered V5 nav_tf_v3 cells, so each
variant sees the same anatomy/seed/cluster/variation cell. Variants retain the
learned policy unless explicitly disabled at the requested safety/coordinator
layer.
"""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing as mp, traceback
from pathlib import Path

ROOT = Path('research/validation/V5_PHYSIO_20261008')

def job(spec):
    row, ckpt, variant = spec
    try:
        import torch
        torch.set_num_threads(1)
        from scripts.benchmark_lysis import LysisEpisode
        from scripts.train_lysis_nav import NavController
        n, anatomy, seed = int(row['clusters']), row['anatomy'], int(row['seed'])
        v = row.get('variation')
        s = float(v.get('strength', 0.)) if isinstance(v, dict) else 0.
        kw = dict(variation=s if s > 0 else None)
        if variant == 'no_tpg':
            kw.update(tpg=False, fallback='park')
        ep = LysisEpisode(n, anatomy, seed, **kw)
        ctl = NavController(ep, Path(ckpt))
        if variant in ('no_guard', 'no_safety'):
            ctl.guard = lambda _ep, _est, _tgt, command, _hold: command
        if variant in ('no_shield', 'no_safety'):
            ep.shield = None
        try:
            while True:
                est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est)
                hold = ep.hold(est)
                local = ctl(ep, est, tgt, rule, hold)
                done, _ = ep.step(est, local, hold)
                if done: break
            z = ep.row(f'nav_{variant}')
            z.update(source_strength=f's{int(round(10*s))}', variant=variant,
                     checkpoint=str(ckpt), paired_method='nav_tf_v3')
            return z
        finally:
            ep.close()
    except Exception:
        return dict(method=f'nav_{variant}', anatomy=row.get('anatomy'), clusters=row.get('clusters'),
                    seed=row.get('seed'), source_strength=row.get('variation', {}).get('strength', 0.) if isinstance(row.get('variation'), dict) else 0.,
                    variant=variant, checkpoint=str(ckpt), error=traceback.format_exc())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variants', default='full,no_guard,no_shield,no_safety,no_tpg')
    ap.add_argument('--strengths', default='s0,s1,s15')
    ap.add_argument('--workers', type=int, default=18)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    variants = [x.strip() for x in a.variants.split(',') if x.strip()]
    rows = []
    for strength in [x.strip() for x in a.strengths.split(',') if x.strip()]:
        rows.extend(json.loads(line) for line in (ROOT/f'{strength}_nav_tf_v3.jsonl').read_text().splitlines() if line.strip())
    jobs = [(r, str(a.policy), v) for v in variants for r in rows]
    manifest = dict(policy=str(a.policy), variants=variants, strengths=a.strengths,
                    episodes=len(rows), jobs=len(jobs), source='registered V5 nav_tf_v3 cells',
                    checkpoint_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=4) as pool, (a.out/'episodes.jsonl').open('w') as f:
        for i, z in enumerate(pool.imap_unordered(job, jobs), 1):
            f.write(json.dumps(z, ensure_ascii=False)+'\n'); f.flush()
            if i % 50 == 0 or i == len(jobs): print(f'{i}/{len(jobs)}', flush=True)
    print(a.out)

if __name__ == '__main__': main()
