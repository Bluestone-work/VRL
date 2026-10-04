"""Frontier-choice supervision for the learned branch selector (training anatomies only).

Runs the fair tabu follower (local_tabu) and, at every frontier decision, stores the fair features of
each candidate and a privileged label: the candidate whose geodesic distance to the target clot
(on the known map) is smallest. The label is used for training only; the selector never sees the map.
usage: collect_frontier_data.py --anatomy A --clusters N --first-seed S --episodes E --out NPZ
"""
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
from environments.mca_physical_env import DynamicsConfig
from marl.teacher import TEACHER_CONFIG
import scripts.benchmark_multicluster as bm
from marl.edge_follower import LocalFollower
from marl.multicluster import MultiClusterConfig
from marl.multicluster_observation import ClusterSensorAdapter
p = argparse.ArgumentParser(); p.add_argument('--anatomy'); p.add_argument('--clusters', type=int); p.add_argument('--first-seed', type=int)
p.add_argument('--episodes', type=int); p.add_argument('--out', type=Path); a = p.parse_args()
reg = json.load(open('configs/evaluation_splits.json')); assert a.anatomy in reg['anatomy_holdout_v1']['train']
r = next(x for x in reg['reserved_ranges'] if x['name'].startswith('teacher dataset')); assert r['start'] <= a.first_seed and a.first_seed+a.episodes <= r['stop']
X, Y, G = [], [], []
for seed in range(a.first_seed, a.first_seed+a.episodes):
    try:
        env, _ = bm.paired_environment(replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=a.anatomy, episode_duration_s=300.), a.clusters, seed)
    except (ValueError, RuntimeError):
        continue
    Gs = bm.station_geodesic(env)
    def rec(env, i, t, frontier, F):
        d = [Gs[int(u), int(env.clot_stations[t])] for u in frontier]
        X.append(F); Y.append(int(np.argmin(d))); G.append(len(X))
    mc = MultiClusterConfig(method='multi_parallel' if a.clusters > 1 else 'single_sequential', clusters=a.clusters, min_spacing_mm=2. if a.clusters > 1 else 0.)
    sensor = ClusterSensorAdapter(env, mc); sensor.reset(seed)
    shield = bm.Shield(mc, robot_speed_mm_s=1., control_dt_s=.1)
    plan, _ = bm.preoperative_plan(env); pt = bm.PlanTargets(plan); lf = LocalFollower(env, tabu_s=30., recorder=rec)
    while True:
        pk = sensor.observe(); tgt = pt.targets(env, env.positions_mm[:a.clusters])
        _, _, te, tr, _ = env.step(sensor.execute(shield.filtered(lf.act(tgt, pk.navigation), pk)))
        if te or tr:
            break
    env.close()
n = max(len(x) for x in X) if X else 1
Xp = np.zeros((len(X), n, len(LocalFollower.FEATURES)), np.float32); M = np.zeros((len(X), n), bool)
for k, x in enumerate(X):
    Xp[k, :len(x)] = x; M[k, :len(x)] = True
np.savez_compressed(a.out, X=Xp, mask=M, y=np.array(Y), anatomy=a.anatomy, clusters=a.clusters)
print(a.anatomy, a.clusters, len(X), 'decisions')
