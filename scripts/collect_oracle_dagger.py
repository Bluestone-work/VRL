"""Primitive-oracle labels for Oracle -> BC -> DAgger (VP observation, topology-fixed pursuit, image sensing).

Oracle (privileged, training only): every 0.5 s each of the 7 primitives is rolled out on a simulator copy for
0.5 s and then held for 1.5 s more (tail='hold': pure primitives, no heuristic); the lowest teacher cost wins.
Executed action: the oracle's (BC, --beta 1) or, with probability 1-beta per decision, the current student's
argmax (DAgger: states from the student's own distribution, always labelled by the oracle).
Recorded per decision: VP observation (biplane frames + token history), oracle label, 7 oracle costs.
Train anatomies only, seeds 2630000000+ (disjoint from dev 2600000000+ and the sealed pool).
usage: collect_oracle_dagger.py --out DIR --first-seed S --episodes N [--ckpt student.pt --beta 0.5]
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np

HOLD, TAIL = 5, 15


def job(x):
    anatomy, seed, out, ckpt, beta = x
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    from marl.lookahead_teacher import label_primitives, primitive_local
    from marl.obstacle_control import History, token, token_dim
    from scripts.benchmark_obstacles import Episode
    from scripts.train_vision_ppo import Frames, PERIOD, load
    f = Path(out)/f'{anatomy}_{seed}.npz'
    if f.exists():
        return dict(skipped=str(f))
    net = load(ckpt)[0] if ckpt else None; rng = np.random.default_rng(seed+31)
    try:
        ep = Episode(1, anatomy, seed, horizon=300., sensing='image', topo_pursuit=True)
    except (ValueError, RuntimeError) as e:
        return dict(anatomy=anatomy, seed=seed, error=repr(e))
    hist = History(1, dim=token_dim(False)); fr = Frames()
    I, S, M, Y, C = [], [], [], [], []; k = 0; act = 0; n_student = 0
    try:
        while True:
            est, tgt, rule, hold = ep.observe(); nominal = np.array(ep.ctl.nominal, float)
            T, _ = token(ep.env, ep.sensor, est, ep.ctl, nominal, hold, tgt, ep.prev_local, False)
            seq, mask = hist.push(T); img = fr.push(ep.sensor.crops)
            if k % PERIOD == 0 and est.active[0] and tgt[0] >= 0:
                y, costs = label_primitives(ep, est, nominal, hold, hold_steps=HOLD, tail=TAIL, tail_policy='hold')
                I.append(img[0].astype(np.float16)); S.append(seq[0]); M.append(mask[0]); Y.append(y); C.append(costs)
                act = y
                if net is not None and rng.random() > beta:
                    with torch.no_grad():
                        act = int(net(torch.as_tensor(img), torch.as_tensor(seq), torch.as_tensor(mask))[0].argmax())
                    n_student += 1
            done, _ = ep.step(est, primitive_local(ep, est, nominal, act), hold); k += 1
            if done:
                break
        row = ep.row('oracle_collect')
    finally:
        ep.close()
    if not Y:
        return dict(anatomy=anatomy, seed=seed, samples=0)
    np.savez_compressed(f, img=np.stack(I), seq=np.stack(S), mask=np.stack(M), label=np.array(Y), costs=np.stack(C).astype(np.float32))
    return dict(anatomy=anatomy, seed=seed, samples=len(Y), student_frac=n_student/len(Y), safe=row['cluster_safe_success'],
                task=row['task_success'], obstacle_events=row['obstacle_events'], wall=row['wall_contact_s'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--first-seed', type=int, required=True)
    p.add_argument('--episodes', type=int, default=44); p.add_argument('--workers', type=int, default=22)
    p.add_argument('--ckpt'); p.add_argument('--beta', type=float, default=1.)
    a = p.parse_args()
    assert not (2700000000 <= a.first_seed < 2800000000), 'sealed seed range'
    a.out.mkdir(parents=True, exist_ok=True); train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    jobs = [(train[k % len(train)], a.first_seed+k, str(a.out), a.ckpt, a.beta) for k in range(a.episodes)]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, (a.out/'log.jsonl').open('a') as log:
        for r in pool.imap_unordered(job, jobs):
            log.write(json.dumps(r, default=str)+'\n'); log.flush()


if __name__ == '__main__':
    main()
