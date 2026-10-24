"""Summarize the bounded paired screen, retaining failures in denominators."""
import json
from collections import defaultdict
from pathlib import Path
import time
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait', action='store_true')
    a = p.parse_args()
    folder = Path(__file__).resolve().parents[1] / 'research/validation/EXP0108_SAC_3D_OBSTACLE_20261013'
    if a.wait:
        deadline = time.monotonic() + 7200
        while not (folder / 'PIPELINE_COMPLETE.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Pipeline incomplete; no final report generated')
            time.sleep(10)
    rows = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], 'all')].append(row)
        groups[(row['method'], row['cell'])].append(row)
    names = ['task_success', 'strict_success', 'removal', 'removal_auc',
             'wall_contact_s', 'spacing_violation_pair_s', 'lost',
             'particle_contact_s', 'particle_collision_events']
    output = []
    for (method, cell), group in sorted(groups.items()):
        n = len(group)
        record = dict(method=method, cell=cell, episodes=n)
        record.update({name: sum(float(r[name]) for r in group) / n for name in names})
        reached = [r['t90_s'] for r in group if r['t90_s'] is not None]
        record.update(t90_reached=len(reached)/n,
                      T90_300=sum(r['t90_s'] if r['t90_s'] is not None else 300. for r in group)/n,
                      conditional_T90=sum(reached)/len(reached) if reached else None)
        output.append(record)
    (folder / 'summary.json').write_text(json.dumps(output, indent=2))
    text = ['# EXP0108 development smoke', '',
            'Single training seed, 20 iterations. Archived scalar SAC/TD3 budgets differ; not a matched training ablation.',
            'All episodes enter task, AUC and T90_300 denominators. No stability or significance claim.', '',
            '| Method | n | Full clear | Strict | Removal | AUC | T90_300 | Wall s | Spacing pair-s | Lost | Particle s | Collision events |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in output:
        if r['cell'] != 'all':
            continue
        method = r['method']
        if 'EXP0108' in method:
            method = '3D SAC (ours, smoke)' if 'policy.pt' in method else '3D SAC initialization'
        else:
            method = method.partition('=')[0]
        fields = [r[k] for k in ['task_success', 'strict_success', 'removal', 'removal_auc',
                  'T90_300', 'wall_contact_s', 'spacing_violation_pair_s', 'lost',
                  'particle_contact_s', 'particle_collision_events']]
        text.append(f'| {method} | {r["episodes"]} | ' + ' | '.join(f'{x:.4f}' for x in fields) + ' |')
    (folder / 'RESULTS.md').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
