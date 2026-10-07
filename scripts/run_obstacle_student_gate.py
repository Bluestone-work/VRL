"""G2: deployable student on the 84 registered G1 development scenes (14 anatomies x seeds 2600000000-5, N=1).
APF/teacher rows are reused from the G1 file; pairing is checked by scenario_hash in analyze_obstacle_gate."""
from __future__ import annotations
import argparse, json, multiprocessing as mp
from pathlib import Path
from scripts.run_obstacle_gate import ANATOMIES


def job(x):
    anatomy, seed, ckpt, method, shield, perc, sens = x
    import torch; torch.set_num_threads(1)
    from scripts.benchmark_obstacles import run
    try:
        return dict(anatomy=anatomy, seed=seed, row=run(method, 1, anatomy, seed, ckpt=ckpt, shield=shield, perception=perc, sensing=sens), error=None)
    except Exception as e:  # keep failures as records
        return dict(anatomy=anatomy, seed=seed, row=None, error=repr(e))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt'); p.add_argument('--method', default='student')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--workers', type=int, default=20); p.add_argument('--seeds', type=int, default=6); p.add_argument('--shield', type=float); p.add_argument('--perception', default='detector', choices=('detector', 'truth')); p.add_argument('--sensing', default='noise', choices=('noise', 'image'))
    a = p.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        done = {(r['anatomy'], r['seed']) for r in map(json.loads, a.out.read_text().splitlines())}
    jobs = [(an, 2600000000+k, a.ckpt, a.method, a.shield, a.perception, a.sensing) for an in ANATOMIES for k in range(a.seeds) if (an, 2600000000+k) not in done]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=4) as pool, a.out.open('a') as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r, default=str)+'\n'); f.flush()


if __name__ == '__main__':
    main()
