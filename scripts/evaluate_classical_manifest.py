"""Run the classical local controllers on navigation_dataset_v1 rows.

The learned policy is evaluated through CurriculumVascularEnv, so the classical
controllers are driven on the episode that same environment builds: identical
anatomy, scene seed, flow, obstacle count, clot initialisation, variation,
sensing latency and frozen target assignment (including the geodesic fallback
when a makespan-optimal plan leaves a cluster unassigned). Only the local
command differs, which is the comparison we want. Arrival uses the environment's
own navigation tolerance, not a separate criterion.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.vessel_sdf import project_local_commands

METHODS = ('pursuit', 'stpg', 'pac')


def build_controller(method, ep):
    """Classical local controller on an already-constructed episode."""
    from scripts.benchmark_lysis import SettleGuard
    if method in ('pursuit', 'stpg'):
        ctl = SettleGuard(ep)
        switcher = None
        if method == 'stpg' and getattr(ep, 'coord', None) is not None:
            from marl.lysis_baselines import SwitchableTPG
            switcher = SwitchableTPG(ep.coord)
        return ctl, switcher
    if method == 'pac':
        from marl.lysis_baselines import PacNMPC
        return PacNMPC(ep), None
    raise ValueError(method)


def run_row(phys, method, row):
    _, info = phys.reset(options=dict(dataset_row=row))
    scene = info["scene"]
    ep = phys.ep
    n = phys.clusters
    tol = float(phys.nav_goal_tolerance_mm)
    targets = np.asarray(phys.nav_targets, int)
    ctl, switcher = build_controller(method, ep)
    reached = np.zeros(n, bool)
    wall = 0.
    wall_after_reach = 0.
    started = time.monotonic()
    est, _, hold, _ = phys.current
    rule = phys.current_rule
    while True:
        if switcher is not None:
            switcher.update(ep.env.elapsed_s)
            hold = ep.hold(est)
        local = ctl(ep, est, targets, rule, hold)
        local, _ = project_local_commands(local, est, targets, hold, ep.ctl.frames(est), phys.sdf, ep.ctl.body)
        done, res = ep.step(est, local, hold)
        wall += float(np.sum(res['wall']))
        wall_after_reach += float(res['wall'][reached].sum())
        reached |= phys._navigation_distances() <= tol
        if done or reached.all() or not ep.env.active[:n].all():
            break
        est = ep.observe()
        rule = ep.ctl.act(targets, est)
        if switcher is None:
            hold = ep.hold(est)
    lost = int(np.count_nonzero(~ep.env.active[:n]))
    return dict(method=method, anatomy=row['anatomy'], clusters=n,
                scene_seed=int(row['scene_seed']), flow_mode=row['flow_mode'],
                obstacle_density=row['obstacle_density'],
                obstacle_layout=row['obstacle_layout'],
                clot_pattern=row['clot_pattern'],
                condition_id=int(row.get('condition_id', -1)),
                scenario_hash=row.get('scenario_hash'), realized_scenario_hash=scene['scenario_hash'],
                accepted_seed=scene['accepted_seed'], navigation_targets=targets.tolist(),
                particle_collision_events=int(ep.info['episode_particle_collision_events']),
                particle_contact_s=float(ep.info['episode_particle_contact_s']),
                navigation_success=bool(reached.all() and lost == 0),
                navigation_reached=int(reached.sum()),
                elapsed_s=float(ep.env.elapsed_s),
                wall_contact_s=float(wall), wall_after_reach_s=wall_after_reach, lost=lost,
                strict_success=bool(reached.all() and lost == 0 and wall < 1.),
                seconds=float(time.monotonic() - started))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--split', choices=('train', 'validation', 'holdout_test'), required=True)
    ap.add_argument('--clusters', type=int, choices=(1, 2, 3), default=3)
    ap.add_argument('--centerline-match', action='store_true')
    ap.add_argument('--max-episodes', type=int, default=180)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.navigation_dataset import iter_rows, MANIFEST
    rows = list(iter_rows(split=a.split, clusters=a.clusters))
    if not rows or a.max_episodes < 0:
        ap.error('evaluation needs matching rows and a nonnegative max-episodes')
    if a.max_episodes:
        # Must match evaluate_navigation_dataset.py exactly: it subsamples with
        # linspace over the whole split, so taking the first N rows instead would
        # compare the controllers on a different set of scenes (the leading rows
        # are all one flow regime) and break the pairing.
        idx = np.linspace(0, len(rows) - 1, min(a.max_episodes, len(rows)), dtype=int)
        rows = [rows[int(i)] for i in idx]
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        ap.error(f"specify distinct methods from {METHODS}")
    phys = CurriculumVascularEnv(seed=1041, clusters=a.clusters, curriculum=False,
                                 reward_mode='navigation', zero_latency=True,
                                 direct_action=True, centerline_match=a.centerline_match)
    # The development adapter refuses anatomies outside its training pool. Holdout
    # rows are permitted here for the same reason evaluate_navigation_dataset.py
    # permits them: this is an explicit evaluation on frozen manifest rows, never
    # a training path.
    phys.train_anatomies = sorted({r['anatomy'] for r in rows})
    out = []
    for i, row in enumerate(rows):
        for m in methods:
            try:
                r = run_row(phys, m, row)
            except Exception as exc:
                import traceback
                out.append(dict(method=m, row_index=i, anatomy=row["anatomy"], flow_mode=row["flow_mode"],
                                scene_seed=row["scene_seed"], scenario_hash=row["scenario_hash"], error=repr(exc)[:200],
                                trace=traceback.format_exc()[-600:]))
                print(json.dumps({'row': i, 'method': m, 'error': repr(exc)[:110]}), flush=True)
                continue
            r['row_index'] = i
            out.append(r)
            print(json.dumps({'row': i, 'method': m, 'succ': r['navigation_success'],
                              'reached': r['navigation_reached'],
                              't': round(r['elapsed_s'], 1),
                              'wall': round(r['wall_contact_s'], 2)}), flush=True)
    phys.close()
    from scripts.evaluate_navigation_dataset import aggregate
    overall = {}
    by_anatomy = {}
    by_flow = {}
    for method in methods:
        group = [r for r in out if r["method"] == method]
        overall[method] = aggregate(group)
        for field, table in (("anatomy", by_anatomy), ("flow_mode", by_flow)):
            table[method] = {key: aggregate([r for r in group if r[field] == key])
                             for key in sorted({r[field] for r in group})}
    result = dict(protocol='classical controllers on navigation_dataset_v1 rows via CurriculumVascularEnv',
                  split=a.split, clusters=a.clusters, methods=methods, centerline_match=a.centerline_match,
                  manifest=str(MANIFEST.relative_to(ROOT)),
                  manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  overall=overall, by_anatomy=by_anatomy, by_flow_mode=by_flow,
                  n_errors=sum(group["errors"] for group in overall.values()), episodes=out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    if result["n_errors"]:
        raise SystemExit("Incomplete classical evaluation. See error rows in the output.")


if __name__ == '__main__':
    main()
