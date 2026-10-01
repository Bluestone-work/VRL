"""Tabulate a finished paired MCA study from its raw per-episode evaluations.

Success rates are recomputed from episode records and checked against the
trainer's own summary; layouts must match across every arm and seed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
STEPS = 500000


def load(study, arm, seed):
    path = study/arm/f'seed_{seed}'/f'evaluation_{STEPS}.json'
    data = json.loads(path.read_text())
    episodes = data['episodes']
    if len(episodes) != 100:
        raise ValueError(f'{path}: expected 100 episodes')
    rates = dict(success_rate=np.mean([e['success'] for e in episodes]),
                 collision_free_success_rate=np.mean([e['collision_free_success'] for e in episodes]))
    for key, value in rates.items():
        if abs(value-data[key]) > 1e-12:
            raise ValueError(f'{path}: recomputed {key} differs')
    return episodes, hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(episodes):
    return dict(success=100*np.mean([e['success'] for e in episodes]),
                clean=100*np.mean([e['collision_free_success'] for e in episodes]),
                collision=100*np.mean([e['particle_contact_s'] > 1e-12 for e in episodes]),
                removal=100*np.mean([e['removal_fraction'] for e in episodes]),
                events=np.mean([e['particle_collision_events'] for e in episodes]),
                contact_s=np.mean([e['particle_contact_s'] for e in episodes]),
                duration=np.mean([e['elapsed_s'] for e in episodes]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True)
    p.add_argument('--arms', required=True, help='comma separated, control first')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    study, out = ROOT/args.study, ROOT/args.out
    arms = args.arms.split(',')
    if json.loads((study/'study_status.json').read_text())['phase'] != 'completed':
        raise ValueError('Study has not completed')
    out.mkdir(parents=True, exist_ok=False)
    rows, table, hashes, layouts = [], {}, {}, None
    for arm in arms:
        table[arm] = {}
        for seed in SEEDS:
            episodes, digest = load(study, arm, seed)
            hashes[f'{arm}/seed_{seed}'] = digest
            current = [(e['seed'], e['reset_info']) for e in episodes]
            layouts = layouts or current
            if current != layouts:
                raise ValueError('Unpaired validation layouts')
            table[arm][seed] = metrics(episodes)
            rows += [dict(arm=arm, training_seed=seed, layout_seed=e['seed'], success=e['success'],
                          collision_free_success=e['collision_free_success'],
                          particle_collision_events=e['particle_collision_events'],
                          particle_contact_s=e['particle_contact_s'], removal_fraction=e['removal_fraction'],
                          lost_robots=e['lost_robots'], elapsed_s=e['elapsed_s'], reason=e['reason'])
                     for e in episodes]
        table[arm]['mean'] = {k: float(np.mean([table[arm][s][k] for s in SEEDS])) for k in table[arm][SEEDS[0]]}
    with (out/'episodes.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(study=args.study, arms=arms, control=arms[0], seeds=list(SEEDS), steps_per_arm_seed=STEPS,
                   evaluation_layouts=100, evaluation_sha256=hashes,
                   results={a: {str(k): v for k, v in t.items()} for a, t in table.items()},
                   scope='Paired development validation on 100 shared layouts; not a sealed test', world_model=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False)+'\n')
    lines = ['| 组 | 训练seed | 完整清栓 | 无接触且全员保留清栓 | 碰撞回合 | 平均清除质量 |', '|---|---|---:|---:|---:|---:|']
    for arm in arms:
        for seed in (*SEEDS, 'mean'):
            m = table[arm][seed]
            label = '均值' if seed == 'mean' else str(seed)
            lines.append(f"| {arm} | {label} | {m['success']:.1f}% | {m['clean']:.1f}% | {m['collision']:.1f}% | {m['removal']:.1f}% |")
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
