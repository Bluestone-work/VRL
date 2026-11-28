"""Build the V3 validation and frozen test task sets (base tasks independent of N).

Seed namespaces (new, disjoint from every earlier VRL set):
  dev 3100000000+k (development, reusable)   validation 3700000000+k   test 3800000000+k
Tasks whose targets cannot be placed are listed under `excluded` with the reason; they are
never silently replaced.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from marl.v3.scene import BaseTask, SceneBuilder

ANAT = json.loads((ROOT / 'configs/evaluation_splits.json').read_text())['navigation_dataset_v1']
TRAIN, HOLD = list(ANAT['train_anatomies']), list(ANAT['holdout_test_anatomies'])
ALL = TRAIN + HOLD + ['hcc_portal_multifocal']
FLOWS = ('weak', 'strong', 'pulsatile')
BASE = {'dev': 3100000000, 'validation': 3700000000, 'test': 3800000000}


def specs(split):
    out = []
    k = 0
    for ai, an in enumerate(ALL):
        if split == 'validation':
            grid = [(m, 'dispersed' if (ai + m) % 2 == 0 else 'clustered') for m in (3, 4, 5)]
        else:
            grid = [(m, p) for m in (2, 3, 4, 5) for p in ('dispersed', 'clustered')]
        for m, p in grid:
            out.append(BaseTask(an, BASE[split] + k, m, p, FLOWS[k % 3], obstacles=8, variation=0.75, split=split))
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('dev', 'validation', 'test'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    tasks, excluded = [], []
    for t in specs(a.split):
        try:
            sb = SceneBuilder(t)
        except ValueError as exc:
            excluded.append(dict(task=t.key(), reason=str(exc)))
            continue
        tasks.append(dict(task=t.key(), base_task_hash=sb.base_task_hash, targets=sb.targets.tolist(),
                          pattern_realised=sb.pattern_realised, root_station=sb.root_station,
                          anatomy_group='hcc' if t.anatomy.startswith('hcc') else
                          ('rl_train_anatomy' if t.anatomy in TRAIN else 'rl_holdout_anatomy')))
        sb.close()
    from scripts.multicluster_protocol import digest
    doc = dict(split=a.split, n_tasks=len(tasks), excluded=excluded, tasks=tasks,
               set_hash=digest([x['base_task_hash'] for x in tasks]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))
    print(a.split, len(tasks), 'tasks', len(excluded), 'excluded', doc['set_hash'][:16])


if __name__ == '__main__':
    main()
