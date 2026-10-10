"""EXP0090 paired evaluation of control arms with execution diagnostics.

Arms (--methods, comma separated):
  fixed_settle | adaptive_settle | switch_settle | stpg | pac_nmpc        rules / local paper adaptations
  nav=<ckpt>, nav_off=<ckpt>            nav_tf_v3 policy / same NavController with the residual zeroed (learner-off)
  sel_learned, sel_learned_fb, sel_online   prediction-assisted selection (D), + uncertainty fallback (E), with the
                                            non-learned online affine estimator (F); --predictors, --fallback-std
Protocols (--protocol):
  hard   EXP0073 registered scenes (research/validation/EXP0073_LONG1M_20261009/nav_tf_v3_s7101/manifest.json),
         panels from scripts.evaluate_hard_baselines.ALL_PANELS (flow_inlet_mm_s, latency_steps, variation)
  v5     V5 registered development scenes (14 anatomies x 10 x N=1/2/3), legacy flow, latency 1, --strengths 0,1,1.5
Rows: every LysisEpisode.row metric + T90_300 (and T50/T100), diagnostics, inference time, scenario_hash.
usage: eval_exp0090.py --out DIR --protocol hard --panels low_delay,... --methods ... --workers 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

SOURCE = 'research/validation/EXP0073_LONG1M_20261009/nav_tf_v3_s7101/manifest.json'


def make_episode(spec):
    from marl.deployable_sensing import DeployableConfig
    from scripts.benchmark_lysis import LysisEpisode
    method, cell, (anatomy, n, seed, flow, latency, variation), opts = spec
    kind = method.split('=')[0]
    kw = dict(sense_cfg=DeployableConfig(latency_steps=latency), variation=variation if variation > 0 else None)
    if flow is not None:
        kw['flow_inlet_mm_s'] = flow
    if kind == 'pac_nmpc' or opts.get('no_tpg'):
        kw['tpg'] = False
    if opts.get('no_tpg') and kind != 'pac_nmpc':
        kw['fallback'] = 'park'          # same allocation fallback as with TPG (EXP0080 convention); only the TPG is removed
    return LysisEpisode(n, anatomy, seed, **kw), kind


def controller(method, ep, opts):
    from scripts.benchmark_lysis import AdaptiveSettleGuard, SettleGuard, SwitchSettle
    kind, _, arg = method.partition('=')
    if kind == 'fixed_settle':
        return SettleGuard(ep)
    if kind == 'adaptive_settle':
        return AdaptiveSettleGuard(ep)
    if kind == 'switch_settle':
        return SwitchSettle(ep, .15)
    if kind in ('nav', 'nav_off'):
        from scripts.train_lysis_nav import NavController
        return NavController(ep, arg, zero_residual=(kind == 'nav_off'), diagnostics=True)
    if kind.startswith('sel_'):
        from marl.dyn_predictor import Ensemble
        from marl.dyn_select import PredictiveSelector
        if kind in ('sel_online', 'sel_online_inv'):
            return PredictiveSelector(ep, 'online', weights=opts.get('weights'), belief_candidates=kind.endswith('_inv'))
        ens = Ensemble(arg.split(';') if arg else opts['predictors'].split(','))
        return PredictiveSelector(ep, 'learned', ens, opts.get('fallback_std') if kind.startswith('sel_learned_fb') else None,
                                  opts.get('weights'), belief_candidates=kind.endswith('_inv'))
    if kind == 'pac_nmpc':
        from marl.lysis_baselines import PacNMPC
        return PacNMPC(ep)
    raise ValueError(method)


def nav_diag(ctl, post):
    D = ctl.diag
    if not D:
        return {}
    live = np.array([d['live'] for d in D], bool); hold = np.array([d['hold'] for d in D], bool)
    m = np.array([d['mapped'] for d in D]); g = np.array([d['guarded'] for d in D]); z = np.array([d['zero_cmd'] for d in D])
    P = np.array(post[:len(D)]); a = np.abs(np.array([d['a_raw'] for d in D]))
    L = max(int(live.sum()), 1); dg = np.linalg.norm(g-m, axis=-1); ds = np.linalg.norm(P-g, axis=-1); dl = np.linalg.norm(m-z, axis=-1)
    return dict(diag_live_steps=int(live.sum()), diag_hold_frac=float(hold.mean()),
                diag_guard_override_frac=float((dg[live] > 1e-6).sum()/L), diag_guard_override_mag=float(dg[live].mean()) if live.any() else 0.,
                diag_shield_override_frac=float((ds[live] > 1e-6).sum()/L), diag_shield_override_mag=float(ds[live].mean()) if live.any() else 0.,
                diag_learn_dev_mag=float(dl[live].mean()) if live.any() else 0., diag_learn_dev_frac=float((dl[live] > .05).sum()/L),
                diag_raw_action_abs=a[live].mean(0).tolist() if live.any() else None)


def child(spec, conn):
    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        import torch; torch.set_num_threads(1)
        method, cell, job, opts = spec
        if method in ('stpg',):
            from scripts.benchmark_lysis import METHODS
            from marl.deployable_sensing import DeployableConfig
            anatomy, n, seed, flow, latency, variation = job
            kw = dict(sense_cfg=DeployableConfig(latency_steps=latency), variation=variation if variation > 0 else None)
            if flow is not None:
                kw['flow_inlet_mm_s'] = flow
            row, ep = METHODS['stpg'](n, anatomy, seed, **kw); extra = {}
        else:
            ep, kind = make_episode(spec); ctl = controller(method, ep, opts); post = []; t0 = time.perf_counter(); k = 0
            abl = opts.get('ablation')
            if abl in ('no_guard', 'no_safety'):            # remove WallGuard (re-planning + near-wall projection)
                class _NoGuard:
                    class exec:
                        @staticmethod
                        def act(t, est, ctl_, local, hold):
                            return local
                    def __call__(self, ep_, est, tgt, command, hold):
                        return command
                if hasattr(ctl, 'guard'):
                    ctl.guard = _NoGuard()
            if abl in ('no_shield', 'no_safety'):
                ep.shield = None
            while True:
                est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
                done, _ = ep.step(est, ctl(ep, est, tgt, rule, hold), hold); k += 1
                if kind in ('nav', 'nav_off'):
                    post.append(ep.ctl.to_world(ep.prev_local, est))
                if done:
                    break
            row = ep.row(method.split('=')[0]); extra = dict(ctrl_ms_per_step=1000*(time.perf_counter()-t0)/max(k, 1))
            if kind in ('nav', 'nav_off'):
                extra.update(nav_diag(ctl, post))
            if hasattr(ctl, 'summary'):
                extra.update(ctl.summary())
        for h in ('t50_s', 't90_s', 't100_s'):
            row[h.replace('_s', '_300')] = row[h] if row[h] is not None else float(row['horizon_s'])
        row.update(extra, method=method.split('=')[0], method_arg=method.partition('=')[2] or None, cell=cell,
                   flow_inlet_mm_s=job[3], latency_steps=job[4], variation_s=job[5], opts={k: v for k, v in opts.items() if k != 'weights'})
        ep.close(); conn.send(row)
    except Exception:
        conn.send(dict(error=traceback.format_exc(), spec=[spec[0], spec[1], list(spec[2])]))
    finally:
        conn.close()


def jobs_for(a):
    out = []
    if a.protocol == 'hard':
        from scripts.evaluate_hard_baselines import ALL_PANELS
        J = json.loads(Path(SOURCE).read_text())['jobs']
        for panel in a.panels.split(','):
            cond = ALL_PANELS[panel]
            for j in J:
                if tuple(j[-3:]) == cond and (not a.train_only or j[2] == 'seen_topology'):
                    out.append((panel+('|'+j[2]), (j[4], int(j[5]), int(j[6]), float(j[7]), int(j[8]), float(j[9]))))
    elif a.protocol == 'tune':
        # selector tuning scenes: training anatomies only, seeds 2124000000+ (disjoint from EXP0073 jobs, V5 dev and data)
        from scripts.evaluate_hard_baselines import ALL_PANELS
        train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
        for p, panel in enumerate(a.panels.split(',')):
            for k, an in enumerate(train):
                for n in (1, 2, 3):
                    f, L, v = ALL_PANELS[panel]
                    out.append((panel+'|tune', (an, n, 2124000000+p*100000+k*1000+n, f, L, v)))
    else:
        from scripts.benchmark_lysis import dev_seeds
        order = json.load(open('configs/evaluation_splits.json'))['anatomy_order']
        for s in [float(x) for x in a.strengths.split(',')]:
            for n in (1, 2, 3):
                for an in order:
                    for seed in dev_seeds(an, a.count):
                        out.append((f'v5_s{s:g}', (an, n, seed, None, 1, s)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--protocol', default='hard', choices=('hard', 'v5', 'tune'))
    ap.add_argument('--panels', default='low_delay,moderate_delay,high_delay,strong_flow,variable_response,ood_flow_delay')
    ap.add_argument('--strengths', default='0,1,1.5'); ap.add_argument('--count', type=int, default=10)
    ap.add_argument('--methods', required=True); ap.add_argument('--predictors'); ap.add_argument('--fallback-std', type=float)
    ap.add_argument('--weights', default='{}'); ap.add_argument('--no-tpg', action='store_true')
    ap.add_argument('--ablation', choices=('no_guard', 'no_shield', 'no_safety'))
    ap.add_argument('--train-only', action='store_true'); ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--timeout', type=float, default=1200); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    opts = dict(predictors=a.predictors, fallback_std=a.fallback_std, weights=json.loads(a.weights), no_tpg=a.no_tpg,
                ablation=a.ablation)
    J = jobs_for(a)
    if a.limit:
        seen = {}; J = [x for x in J if seen.setdefault(x[0], 0) < a.limit and not seen.__setitem__(x[0], seen[x[0]]+1)]
    specs = [(m, cell, job, opts) for cell, job in J for m in a.methods.split(',')]
    done = set()
    if (a.out/'episodes.jsonl').exists():
        for l in open(a.out/'episodes.jsonl'):
            r = json.loads(l); done.add((r['method']+('='+r['method_arg'] if r.get('method_arg') else ''), r['cell'], r['anatomy'], r['clusters'], r['seed']))
    specs = [s for s in specs if (s[0], s[1], s[2][0], s[2][1], s[2][2]) not in done]
    hashes = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
              ['scripts/eval_exp0090.py', 'scripts/benchmark_lysis.py', 'scripts/train_lysis_nav.py', 'marl/dyn_select.py',
               'marl/dyn_predictor.py', 'marl/lysis_baselines.py']}
    ck = {}
    for m in a.methods.split(','):
        if '=' in m:
            for q in m.split('=')[1].split(';'):
                ck[q] = hashlib.sha256(Path(q).read_bytes()).hexdigest()
    for p in (a.predictors or '').split(','):
        if p:
            ck[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    with (a.out/'manifest.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.strftime('%F %T'), args=vars(a), n_specs=len(specs), source_hashes=hashes,
                                checkpoint_sha256=ck), default=str)+'\n')
    ctx = mp.get_context('fork'); running = {}; cursor = completed = 0
    with (a.out/'episodes.jsonl').open('a') as out, (a.out/'failures.jsonl').open('a') as fail:
        while cursor < len(specs) or running:
            while cursor < len(specs) and len(running) < a.workers:
                spec = specs[cursor]; cursor += 1
                r_, s_ = ctx.Pipe(duplex=False); pr = ctx.Process(target=child, args=(spec, s_)); pr.start(); s_.close()
                running[pr.pid] = (pr, r_, spec, time.monotonic())
            for pid, (pr, r_, spec, t0) in list(running.items()):
                row = None
                if r_.poll():
                    try:
                        row = r_.recv()
                    except EOFError:
                        pass
                late = time.monotonic()-t0 > a.timeout
                if row is None and pr.is_alive() and not late:
                    continue
                if row is None:
                    row = dict(error='timeout' if late else f'worker exited ({pr.exitcode})', spec=[spec[0], spec[1], list(spec[2])],
                               infrastructure_failure=True)
                if pr.is_alive():
                    pr.terminate()
                pr.join(); r_.close(); del running[pid]
                (fail if 'error' in row else out).write(json.dumps(row, default=float)+'\n'); (fail if 'error' in row else out).flush()
                completed += 1
                if completed % 25 == 0 or completed == len(specs):
                    print(f'{time.strftime("%T")} {completed}/{len(specs)}', flush=True)
            time.sleep(.05)


if __name__ == '__main__':
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    main()
