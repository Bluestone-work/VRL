"""Action selection statistics for a trained checkpoint, on frozen manifest rows.

The training script no longer writes action_stats.csv, and usage is the evidence
that decides whether an action slot earns its place in the categorical head: a
slot that is valid but never selected costs capacity and enlarges the search
space without adding behaviour.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--action-set', default='semantic_minimal')
    ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import (IPPOAgentVecEnv, SEMANTIC_MINIMAL_INDICES,
                                        SEMANTIC_REDUCED_INDICES)
    from scripts.navigation_dataset import iter_rows
    from stable_baselines3 import PPO
    LABEL = {0: 'zero', 1: 'rule', 2: 'route_full', 3: '.75_route', 4: 'route_050',
             5: 'target_full', 6: 'target_050', 7: 'target_025', 8: 'settle',
             9: 'reverse', 10: 'normal+035', 11: 'normal-035',
             12: 'binormal+035', 13: 'binormal-035'}
    orig = (SEMANTIC_MINIMAL_INDICES if a.action_set == 'semantic_minimal'
            else SEMANTIC_REDUCED_INDICES)
    rows = list(iter_rows(split='train', clusters=a.clusters))
    idx = np.linspace(0, len(rows) - 1, min(a.episodes, len(rows)), dtype=int)
    rows = [rows[int(i)] for i in idx]
    phys = CurriculumVascularEnv(seed=4101, clusters=a.clusters, curriculum=False,
                                 reward_mode='navigation', zero_latency=True,
                                 direct_action=True)
    phys.train_anatomies = sorted({r['anatomy'] for r in rows})
    env = IPPOAgentVecEnv(phys, action_set=a.action_set)
    model = PPO.load(str(a.model), env=env, device='cpu')
    sel, valid_steps, steps = Counter(), Counter(), 0
    try:
        for row in rows:
            obs = env.reset(options=dict(dataset_row=row))
            while True:
                act, _ = model.predict(obs, deterministic=True)
                for k in np.asarray(act).reshape(-1):
                    sel[int(k)] += 1
                v = env._candidate_valid
                for k in range(v.shape[1]):
                    valid_steps[k] += int((v[:, k] > .5).sum())
                steps += int(np.asarray(act).reshape(-1).size)
                obs, _, done, _ = env.step(act)
                if bool(done[0]):
                    break
    finally:
        env.close()
    out = []
    for k in range(len(orig)):
        out.append(dict(index=k, orig_index=orig[k], label=LABEL.get(orig[k], f'slot{orig[k]}'),
                        selections=sel[k], selection_frequency=sel[k] / max(steps, 1),
                        valid_fraction=valid_steps[k] / max(steps, 1)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    for r in sorted(out, key=lambda r: -r['selection_frequency']):
        print(f"  {r['index']:2d} ({r['orig_index']:2d}) {r['label']:<14}"
              f"{r['selection_frequency']*100:6.2f}%  valid {r['valid_fraction']*100:5.1f}%")
    print(f'steps={steps}  零使用槽位 {sum(1 for r in out if r["selections"] == 0)}/{len(out)}')


if __name__ == '__main__':
    main()
