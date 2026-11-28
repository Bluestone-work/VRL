"""Phase B: global-configuration arms executed with one fixed local controller.

Every arm runs on the same base tasks (base_task_hash) with the same release candidates, cluster
resources, safety layer, flow, obstacles, patient-level variation and deadline; only the global
configuration (and, for abl_no_tpg, the coordination switch) differs.  Rows are appended to a JSONL
file keyed by (base_task_hash, arm); reruns skip finished keys; failures are written as error rows.
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from marl.v3.configuration import jsonable  # noqa: E402


def arm_spec(arm, fixed):
    """Planner arguments for an arm. fixed = {'root': k, 'opt': k} chosen on the validation set."""
    if arm.startswith('fixedN') and arm[6].isdigit():          # fixedN{k}_root / fixedN{k}_opt (validation sweep)
        k = int(arm[6]); dep = 'root' if arm.endswith('_root') else 'optimized'
        return dict(n_fixed=k, deployment=dep, search='local', objective='full'), {}
    specs = {
        'fixedN_root': (dict(n_fixed=fixed.get('root'), deployment='root', search='local', objective='full'), {}),
        'fixedN_random': (dict(n_fixed=fixed.get('root'), deployment='random', search='local', objective='full'), {}),
        'fixedN_opt': (dict(n_fixed=fixed.get('opt'), deployment='optimized', search='local', objective='full'), {}),
        'adaptiveN_root': (dict(n_fixed=None, deployment='root', search='local', objective='full'), {}),
        'adaptiveN_opt_time': (dict(n_fixed=None, deployment='optimized', search='local', objective='time'), {}),
        'proposed': (dict(n_fixed=None, deployment='optimized', search='local', objective='full'), {}),
        'exact': (dict(n_fixed=None, deployment='optimized', search='exact', objective='full'), {}),
        'abl_euclid': (dict(n_fixed=None, deployment='optimized', search='local', objective='full', topology='euclidean'), {}),
        'abl_no_tpg': (dict(n_fixed=None, deployment='optimized', search='local', objective='full'), dict(tpg=False)),
    }
    return specs[arm]


def run_task(job):
    task_doc, arms, fixed, commit = job
    rows = []
    try:
        from marl.v3.scene import BaseTask, SceneBuilder
        from marl.v3.vessel_graph import VesselGraph
        from marl.v3.configuration import Planner
        from marl.v3.execute import run_configuration
        from marl.v3 import assumptions as A
        task = BaseTask(**task_doc['task'])
        sb = SceneBuilder(task)
        assert sb.base_task_hash == task_doc['base_task_hash'], 'base task not reproduced'
        g = VesselGraph(sb.probe)
    except Exception:
        return [dict(base_task_hash=task_doc['base_task_hash'], arm=a, error=traceback.format_exc()) for a in arms]
    for arm in arms:
        t0 = time.monotonic()
        try:
            pspec, xspec = arm_spec(arm, fixed)
            pspec = dict(pspec)
            topology = pspec.pop('topology', 'geodesic')
            if pspec.get('n_fixed'):
                pspec['n_fixed'] = min(int(pspec['n_fixed']), sb.task.n_targets)
            planner = Planner(g, sb.targets, sb.root_station, objective=pspec['objective'], topology=topology,
                              seed=int(task.seed) % 100000)
            cfg = planner.solve(method=arm, n_fixed=pspec['n_fixed'], deployment=pspec['deployment'],
                                search=pspec['search'], objective=pspec['objective'],
                                random_slots_seed=int(task.seed) if pspec['deployment'] == 'random' else None)
            summary, _ = run_configuration(sb, cfg, **xspec)
            rows.append(dict(split=task.split, task=task.key(), anatomy_group=task_doc['anatomy_group'],
                             base_task_hash=sb.base_task_hash, arm=arm, arm_spec=dict(planner=pspec, topology=topology, execution=xspec),
                             configuration=cfg.to_dict(), configuration_hash=cfg.hash(sb.base_task_hash),
                             executed=summary, wall_seconds=float(time.monotonic() - t0), commit=commit,
                             assumptions_version=A.load()['version']))
        except Exception:
            rows.append(dict(split=task.split, task=task.key(), base_task_hash=task_doc['base_task_hash'], arm=arm,
                             error=traceback.format_exc(), commit=commit))
    sb.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=Path, required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fixed-n-root', type=int)
    ap.add_argument('--fixed-n-opt', type=int)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    doc = json.loads(a.tasks.read_text())
    arms = [x for x in a.arms.split(',') if x]
    fixed = dict(root=a.fixed_n_root, opt=a.fixed_n_opt)
    done = set()
    if a.out.exists():
        for l in a.out.read_text().splitlines():
            r = json.loads(l)
            if 'error' not in r:
                done.add((r['base_task_hash'], r['arm']))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    commit = commit + ('+dirty' if dirty else '')
    jobs = []
    for t in doc['tasks'][:a.limit]:
        todo = [arm for arm in arms if (t['base_task_hash'], arm) not in done]
        if todo:
            jobs.append((t, todo, fixed, commit))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(f'{len(jobs)} tasks to run, {sum(len(j[1]) for j in jobs)} arm-episodes, commit {commit}', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=4) as pool, open(a.out, 'a') as fh:
        for k, rows in enumerate(pool.imap_unordered(run_task, jobs)):
            for r in rows:
                fh.write(json.dumps(r, default=jsonable) + '\n')
            fh.flush()
            err = sum('error' in r for r in rows)
            print(f'[{k+1}/{len(jobs)}] {rows[0].get("task", {}).get("anatomy")} rows={len(rows)} errors={err}', flush=True)


if __name__ == '__main__':
    main()
