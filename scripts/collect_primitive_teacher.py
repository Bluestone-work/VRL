"""Privileged lookahead teacher over the motion primitives, recorded with the VP-PPO observation (biplane image
frames + token history), for behaviour-cloning initialisation of VP-PPO. The teacher (simulator copies) only
produces labels; the student input is the deployable observation. Train anatomies, seeds 2621000000+."""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np


def job(x):
    anatomy, seed, out = x
    os.environ['OMP_NUM_THREADS'] = '1'
    from marl.lookahead_teacher import label_primitives, primitive_local
    from marl.obstacle_control import History, token, token_dim
    from scripts.benchmark_obstacles import Episode
    from scripts.train_vision_ppo import Frames, PERIOD
    f = Path(out)/f'{anatomy}_{seed}.npz'
    if f.exists():
        return dict(skipped=str(f))
    ep = Episode(1, anatomy, seed, horizon=300., sensing='image'); hist = History(1, dim=token_dim(False)); fr = Frames()
    I, S, M, Y, C = [], [], [], [], []; k = 0; act = 0
    try:
        while True:
            est, tgt, rule, hold = ep.observe(); nominal = np.array(ep.ctl.nominal, float)
            T, _ = token(ep.env, ep.sensor, est, ep.ctl, nominal, hold, tgt, ep.prev_local, False)
            seq, mask = hist.push(T); img = fr.push(ep.sensor.crops)
            if k % PERIOD == 0:
                act, costs = label_primitives(ep, est, nominal, hold, hold_steps=PERIOD)
                if est.active[0] and tgt[0] >= 0:
                    I.append(img[0].astype(np.float16)); S.append(seq[0]); M.append(mask[0]); Y.append(act); C.append(costs)
            done, _ = ep.step(est, primitive_local(ep, est, nominal, act), hold); k += 1
            if done:
                break
        row = ep.row('primitive_teacher')
    finally:
        ep.close()
    if not Y:
        return dict(anatomy=anatomy, seed=seed, samples=0)
    np.savez_compressed(f, img=np.stack(I), seq=np.stack(S), mask=np.stack(M), label=np.array(Y), costs=np.stack(C).astype(np.float32))
    return dict(anatomy=anatomy, seed=seed, samples=len(Y), safe=row['cluster_safe_success'], task=row['task_success'],
                obstacle_events=row['obstacle_events'], wall=row['wall_contact_s'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--episodes', type=int, default=36)
    p.add_argument('--workers', type=int, default=9); p.add_argument('--first-seed', type=int, default=2621000000); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    jobs = [(train[k % len(train)], a.first_seed+k, str(a.out)) for k in range(a.episodes)]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, (a.out/'log.jsonl').open('a') as log:
        for r in pool.imap_unordered(job, jobs):
            log.write(json.dumps(r, default=str)+'\n'); log.flush()


if __name__ == '__main__':
    main()
