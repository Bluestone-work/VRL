"""Where does wall contact come from? Per-step attribution on the 84 registered dev scenes (N=1).

Controllers: noavoid (route pursuit), switch (rule_switch 0.3 mm), vp (VP-PPO checkpoint, image sensing only)
Sensing:     perfect (cluster position exact, no latency/dropout), noise (default deployable sensor), image (biplane)
For every control step with wall contact (env projected the body back into the lumen) we record:
  phase       clearing (mass removed this step) / at_clot (< 1 mm from own target) / transit
  junction    true position within 1.5 mm of a pre-operative branching node
  outward     command component along the true outward wall normal (> 0: commanded into the wall)
  est_err     estimate error (mm) and its radial part (estimate further from the wall than the truth > 0)
Output: one JSON line per episode with totals and wall-seconds split by each attribute.
usage: diagnose_wall_contact.py --controller noavoid --sensing perfect --out FILE [--ckpt C]
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np


def episode(x):
    controller, sensing, anatomy, seed, ckpt = x
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    from marl.deployable_sensing import DeployableConfig
    from marl.lookahead_teacher import option_local
    from scripts.benchmark_obstacles import Episode
    sc = DeployableConfig(position_sigma_mm=0., latency_steps=0, dropout_prob=0.) if sensing == 'perfect' else DeployableConfig()
    ep = Episode(1, anatomy, seed, sensing='image' if sensing == 'image' else 'noise', sense_cfg=sc, avoid=(controller != 'noavoid'))
    vp = None
    if controller == 'vp':
        from marl.obstacle_control import History, token, token_dim
        from scripts.train_vision_ppo import Frames, PERIOD, load
        from marl.lookahead_teacher import primitive_local
        net, cfg = load(ckpt); vp = (net, History(1, dim=token_dim(False)), Frames(), token, primitive_local, PERIOD)
    ctl = ep.ctl; J = ctl.pts[np.asarray(ctl.deg) >= 3] if np.any(np.asarray(ctl.deg) >= 3) else np.zeros((0, 3))
    acc = dict(wall=0., clearing=0., at_clot=0., transit=0., junction=0., outward=0., inward_or_zero=0., est_bias=0.,
               est_err_sum=0., steps_wall=0, steps=0, est_err_all=0., stop_cmd=0.)
    k = 0; act = 0
    try:
        while True:
            est, tgt, rule, hold = ep.observe()
            if controller == 'noavoid':
                local = rule
            elif controller == 'switch':
                local = option_local(ep, est, rule, 8)
            else:
                net, hist, fr, token, prim, period = vp; nominal = np.array(ctl.nominal, float)
                T, _ = token(ep.env, ep.sensor, est, ctl, nominal, hold, tgt, ep.prev_local, False)
                seq, mask = hist.push(T); img = fr.push(ep.sensor.crops)
                if k % period == 0 and est.active[0] and tgt[0] >= 0:
                    with torch.no_grad():
                        act = int(net(torch.as_tensor(img), torch.as_tensor(seq), torch.as_tensor(mask))[0].argmax())
                local = prim(ep, est, nominal, act)
            pos = ep.env.positions_mm[0].astype(float).copy(); u = ctl.to_world(np.where(hold[:, None], 0., local), est)[0]
            # true outward normal from the map axis of the matched edge, evaluated at the TRUE position
            e = int(est.edge[0]); a_, ab = ep.sensor.a[e], ep.sensor.ab[e]
            t = float(np.clip(((pos-a_)@ab)/max(ab@ab, 1e-12), 0, 1)); ax = a_+t*ab; nrm = pos-ax; rad_t = np.linalg.norm(nrm)
            nrm = nrm/max(rad_t, 1e-9); rad_e = float(np.linalg.norm(est.pos[0]-ax))
            err = float(np.linalg.norm(est.pos[0]-pos)); mass0 = float(ep.env.masses.sum())
            done, out = ep.step(est, local, hold); k += 1
            w = float(np.sum(out['wall'])); acc['steps'] += 1; acc['est_err_all'] += err
            if w > 0:
                acc['wall'] += w; acc['steps_wall'] += 1; acc['est_err_sum'] += err
                removed = mass0-float(ep.env.masses.sum()) > 1e-12
                dt_ = float(np.linalg.norm(ep.env.clot_positions_mm[tgt[0]]-pos)) if tgt[0] >= 0 else 99.
                acc['clearing' if removed else 'at_clot' if dt_ < 1. else 'transit'] += w
                if len(J) and np.min(np.linalg.norm(J-pos, axis=1)) < 1.5:
                    acc['junction'] += w
                acc['outward' if float(u@nrm) > .2 else 'inward_or_zero'] += w
                if not np.any(u):
                    acc['stop_cmd'] += w
                if rad_t-rad_e > .03:          # estimate thinks the body is closer to the axis than it is
                    acc['est_bias'] += w
            if done:
                break
        row = ep.row(controller)
    finally:
        ep.close()
    acc['est_err_mean_at_wall'] = acc.pop('est_err_sum')/max(acc['steps_wall'], 1); acc['est_err_mean'] = acc.pop('est_err_all')/max(acc['steps'], 1)
    return dict(controller=controller, sensing=sensing, anatomy=anatomy, seed=seed, wall_total=row['wall_contact_s'],
                task=row['task_success'], safe=row['cluster_safe_success'], obstacle_events=row['obstacle_events'], **acc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', required=True, choices=('noavoid', 'switch', 'vp')); p.add_argument('--sensing', required=True, choices=('perfect', 'noise', 'image'))
    p.add_argument('--ckpt'); p.add_argument('--out', type=Path, required=True); p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    from scripts.run_obstacle_gate import ANATOMIES
    jobs = [(a.controller, a.sensing, an, 2600000000+k, a.ckpt) for an in ANATOMIES for k in range(6)]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=6) as pool, a.out.open('w') as f:
        for r in pool.imap_unordered(episode, jobs):
            f.write(json.dumps(r, default=float)+'\n'); f.flush()


if __name__ == '__main__':
    main()
