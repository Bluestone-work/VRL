"""Parallel privileged-teacher labelling for the deployable obstacle student (BC and DAgger rounds).

Each job is one (anatomy, seed) episode. The executed option is the teacher's (BC, --beta 1) or, with
probability 1-beta per decision, the current student's argmax (DAgger: states come from the student's own
distribution). The student only ever sees token(); the teacher's cost vector over the 8 options is stored
for soft-target distillation. Training anatomies and non-sealed seeds only.
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp
from pathlib import Path
import numpy as np


def job(a):
    anatomy, seed, out, ckpt, beta, th, v2 = a
    import torch
    torch.set_num_threads(1)
    from marl.lookahead_teacher import label, label_v2, option_local
    from marl.obstacle_control import History, load_discrete_checkpoint, token, token_dim
    from scripts.benchmark_obstacles import Episode
    f = Path(out)/f'{anatomy}_{seed}.npz'
    if f.exists():
        return dict(anatomy=anatomy, seed=seed, skipped=True)
    net = None
    if ckpt:
        cfg = json.loads(Path(ckpt+'.json').read_text()); net = load_discrete_checkpoint(ckpt, cfg)
    rng = np.random.default_rng(seed+29)
    ep = Episode(1, anatomy, seed, horizon=300., avoid=True); hist = History(1, dim=token_dim(False))
    S, M, Y, C = [], [], [], []; used_student = 0; steps = 0; period = 5 if v2 else 1; act = 8 if v2 else 0
    try:
        while True:
            est, tgt, rule, hold = ep.observe()
            T, _ = token(ep.env, ep.sensor, est, ep.ctl, rule, hold, tgt, ep.prev_local, False)
            seq, mask = hist.push(T)
            live = bool(est.active[0] and tgt[0] >= 0 and not hold[0])
            decide = steps % period == 0
            if decide:
                y, costs = label_v2(ep, est, rule, hold, hold_steps=period) if v2 else label(ep, est, rule, hold, horizon=th)
                act = y
            if live and decide:
                S.append(seq[0]); M.append(mask[0]); Y.append(y); C.append(costs)
                if net is not None and rng.random() > beta:
                    with torch.no_grad():
                        act = int(net.logits(torch.as_tensor(seq), torch.as_tensor(mask)).argmax(-1)[0])
                    used_student += 1
            done, _ = ep.step(est, option_local(ep, est, rule, act), hold); steps += 1
            if done:
                break
        r = ep.row('collect')
    finally:
        ep.close()
    if not Y:
        return dict(anatomy=anatomy, seed=seed, samples=0)
    np.savez_compressed(f, seq=np.stack(S), mask=np.stack(M), label=np.asarray(Y, np.int64),
                        costs=np.stack(C).astype(np.float32), period=np.int64(period),
                        meta=json.dumps(dict(anatomy=anatomy, seed=seed, beta=beta, ckpt=ckpt, teacher_horizon=th, teacher='v2' if v2 else 'v1')))
    return dict(anatomy=anatomy, seed=seed, samples=len(Y), student_frac=used_student/max(len(Y), 1),
                safe=r['cluster_safe_success'], obstacle_events=r['obstacle_events'], wall=r['wall_contact_s'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--first-seed', type=int, required=True)
    p.add_argument('--episodes', type=int, required=True); p.add_argument('--ckpt'); p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--teacher-horizon', type=int, default=10); p.add_argument('--workers', type=int, default=18); p.add_argument('--v2', action='store_true')
    a = p.parse_args()
    split = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']
    anat = split['train']
    assert not (2700000000 <= a.first_seed < 2800000000) and not (2700000000 <= a.first_seed+a.episodes < 2800000000), 'sealed seed range'
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(anat[k % len(anat)], a.first_seed+k, str(a.out), a.ckpt, a.beta, a.teacher_horizon, a.v2) for k in range(a.episodes)]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, (a.out/'log.jsonl').open('a') as log:
        for r in pool.imap_unordered(job, jobs):
            log.write(json.dumps(r, default=str)+'\n'); log.flush(); print(json.dumps(r, default=str), flush=True)


if __name__ == '__main__':
    main()
