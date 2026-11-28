"""Planner-only sensitivity on the validation set (no execution): how N*, release and predicted
completion change with epsilon, catheter speed, release interval and the waiting weight."""
from __future__ import annotations

import argparse, json, multiprocessing as mp, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GRID = [dict(name='registered')] + \
       [dict(name=f'eps_{e}', eps=e) for e in (0.0, 0.05, 0.2)] + \
       [dict(name=f'vcath_{v}', v_c=v) for v in (0.5, 1.0, 4.0)] + \
       [dict(name=f'release_interval_{t}', tau_rel=t) for t in (1.5, 6.0)] + \
       [dict(name=f'w_wait_{w}', w_wait=w) for w in (0.0, 3.0)]


def one(task_doc):
    from marl.v3.scene import BaseTask, SceneBuilder
    from marl.v3.vessel_graph import VesselGraph
    from marl.v3.configuration import Planner
    sb = SceneBuilder(BaseTask(**task_doc['task'])); g = VesselGraph(sb.probe)
    out = []
    for p in GRID:
        pl = Planner(g, sb.targets, sb.root_station)
        for k in ('eps', 'v_c', 'tau_rel', 'w_wait'):
            if k in p:
                setattr(pl, k, p[k])
        c = pl.solve(search='exact')
        out.append(dict(base_task_hash=task_doc['base_task_hash'], setting=p['name'], n=c.n,
                        T_pred=c.predicted['T_pred'], deployment_time=c.predicted['deployment_time_s'],
                        wait_s=c.predicted['wait_s'], J_by_n=c.search['J_by_n'], release=c.release_stations,
                        certificate=c.certificate, anatomy=task_doc['task']['anatomy'], m=task_doc['task']['n_targets']))
    sb.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    tasks = json.loads(a.tasks.read_text())['tasks']
    with mp.get_context('spawn').Pool(a.workers) as pool, open(a.out, 'w') as fh:
        for rows in pool.imap_unordered(one, tasks):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
