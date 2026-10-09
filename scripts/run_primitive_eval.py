"""Primitive-level diagnostics on the 84 registered dev scenes (image sensing, N=1), gate-format output.

  --controller away      always the 'away' primitive (fixed control: VP-PPO uses it in 80-92 % of decisions)
  --controller advance   always the 'advance' primitive (route following through the primitive frame)
               vp        frozen VP-PPO checkpoint (no retraining)
               oracle    privileged primitive oracle: every 0.5 s each primitive is rolled out on a simulator
                         copy (hold 0.5 s, then --tail: 'hold' = same primitive, 'switch' = rule_switch) and the
                         lowest teacher cost is executed (upper bound of what the primitive set can do)
               switch    rule_switch baseline (projection applies to its command too)
  --frame carrot|route|velocity|fused   primitive reference direction (marl.lookahead_teacher._direction)
  --safety none|sdf                     wall-safety projection from the map SDF (marl.vessel_sdf)
usage: run_primitive_eval.py --controller vp --ckpt C --frame carrot --safety sdf --out FILE
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np


def job(x):
    controller, ckpt, frame, safety, tail, topo, anatomy, seed = x
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    from marl.lookahead_teacher import label_primitives, option_local, primitive_local
    from marl.vessel_sdf import VesselSDF, wall_safe_projection
    from scripts.benchmark_obstacles import Episode
    try:
        ep = Episode(1, anatomy, seed, sensing='image', avoid=(controller != 'noavoid'), topo_pursuit=topo)
        ep.prim_cfg = dict(frame=frame, safety=None if safety == 'none' else safety)
        if controller == 'vp':
            from marl.obstacle_control import History, token, token_dim
            from scripts.train_vision_ppo import Frames, PERIOD, load
            net, _ = load(ckpt); hist = History(1, dim=token_dim(False)); fr = Frames()
        sdf = VesselSDF(ep.sensor); k = 0; act = 0; counts = np.zeros(7, int)
        try:
            while True:
                est, tgt, rule, hold = ep.observe(); nominal = np.array(ep.ctl.nominal, float)
                if controller == 'vp':
                    T, _ = token(ep.env, ep.sensor, est, ep.ctl, nominal, hold, tgt, ep.prev_local, False)
                    seq, mask = hist.push(T); img = fr.push(ep.sensor.crops)
                if k % 5 == 0 and est.active[0] and tgt[0] >= 0:
                    if controller == 'vp':
                        with torch.no_grad():
                            act = int(net(torch.as_tensor(img), torch.as_tensor(seq), torch.as_tensor(mask))[0].argmax())
                    elif controller == 'oracle':
                        act, _ = label_primitives(ep, est, nominal, hold, hold_steps=5, tail=15, tail_policy=tail)
                    else:
                        act = 3 if controller == 'away' else 0      # fixed-primitive controls
                    counts[act] += 1
                if controller in ('apf', 'noavoid'):
                    local = rule
                elif controller == 'switch':
                    local = option_local(ep, est, rule, 8)
                    if safety == 'sdf':
                        F = ep.ctl.frames(est)
                        for i in range(ep.n):
                            if est.active[i] and np.any(local[i]):
                                local[i] = F[i]@wall_safe_projection(F[i].T@local[i], est.pos[i], sdf, float(ep.ctl.body), edge=int(est.edge[i]))
                else:
                    local = primitive_local(ep, est, nominal, act)
                done, _ = ep.step(est, local, hold); k += 1
                if done:
                    break
            row = ep.row(f'{controller}_{frame}_{safety}'+('_topo' if topo else '')); row['sensing_model'] = 'image'; row['primitive_counts'] = counts.tolist()
        finally:
            ep.close()
        return dict(anatomy=anatomy, seed=seed, row=row, error=None)
    except Exception as e:
        return dict(anatomy=anatomy, seed=seed, row=None, error=repr(e))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', required=True, choices=('advance', 'away', 'vp', 'oracle', 'switch', 'apf', 'noavoid')); p.add_argument('--ckpt')
    p.add_argument('--frame', default='carrot', choices=('carrot', 'route', 'velocity', 'fused'))
    p.add_argument('--safety', default='none', choices=('none', 'sdf')); p.add_argument('--tail', default='hold', choices=('hold', 'switch')); p.add_argument('--topo', action='store_true', help='topology-aware route progress (fix)')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    from scripts.run_obstacle_gate import ANATOMIES
    done = {(r['anatomy'], r['seed']) for r in map(json.loads, a.out.read_text().splitlines())} if a.out.exists() else set()
    jobs = [(a.controller, a.ckpt, a.frame, a.safety, a.tail, a.topo, an, 2600000000+k) for an in ANATOMIES for k in range(6) if (an, 2600000000+k) not in done]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=4) as pool, a.out.open('a') as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r, default=str)+'\n'); f.flush()


if __name__ == '__main__':
    main()
