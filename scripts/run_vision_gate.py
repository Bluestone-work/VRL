"""G3: deterministic evaluation of a VP-PPO checkpoint on the 84 registered dev scenes (image sensing, N=1).
Baselines under the same image sensing: gates/G3img_rule_apf.jsonl, gates/G3img_rule_switch.jsonl."""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
from scripts.run_obstacle_gate import ANATOMIES


def job(x):
    ckpt, anatomy, seed = x
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    from scripts.train_vision_ppo import load, rollout
    try:
        net, _ = load(ckpt); _, row = rollout(net, anatomy, seed, sample=False, record=False)
        row['ckpt'] = ckpt; return dict(anatomy=anatomy, seed=seed, row=row, error=None)
    except Exception as e:
        return dict(anatomy=anatomy, seed=seed, row=None, error=repr(e))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8); a = p.parse_args()
    done = {(r['anatomy'], r['seed']) for r in map(json.loads, a.out.read_text().splitlines())} if a.out.exists() else set()
    jobs = [(a.ckpt, an, 2600000000+k) for an in ANATOMIES for k in range(6) if (an, 2600000000+k) not in done]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=4) as pool, a.out.open('a') as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r, default=str)+'\n'); f.flush()


if __name__ == '__main__':
    main()
