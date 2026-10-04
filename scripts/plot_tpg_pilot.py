"""Plot an admitted EXP0053 pilot, retaining all seeds and paired scene means."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    audit = json.loads((args.root / 'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:
        raise ValueError('No comparison figure from an incomplete or failed matrix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['memory_n1', 'balanced_1s', 'tpg', 'high', 'low', 'both']
    labels = ['Sequential\nN=1', 'Balanced\n1 s', 'V-CTPG\nrule', 'High\nonly', 'Low\nonly', 'V-CTPG\nHRL']
    by_arm = defaultdict(list)
    for path in sorted((args.root / 'results').glob('*.jsonl')):
        name = path.stem
        arm = next((x for x in ['balanced_1s', 'memory_n1', 'untrained', 'tpg', 'high', 'low', 'both'] if name.startswith(x + '_')), None)
        if arm is None:
            raise ValueError(f'Unexpected result {path}')
        for line in path.read_text().splitlines():
            by_arm[arm].append(json.loads(line))
    colors = ['#B8BDC5', '#7E91A9', '#4C78A8', '#E6A254', '#67A88B', '#AF668C']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [('cluster_safe_success', 'Registered safe success (%)', 100.),
               ('removal', 'Clot mass removed (%)', 100.),
               ('removal_auc_180', 'Removal AUC / 180 s (%)', 100.),
               ('wall_contact_s', 'Wall contact (cluster-s)', 1.),
               ('spacing_violation_pair_s', 'Spacing violation (pair-s)', 1.),
               ('scheduler_calls', 'High-level calls / episode', 1.)]
    numeric = []
    for ax, (metric, title, factor) in zip(axes.flat, metrics):
        for index, arm in enumerate(names):
            if metric == 'spacing_violation_pair_s' and arm == 'memory_n1':
                ax.text(index, .03, 'N/A', ha='center', va='bottom',
                        transform=ax.get_xaxis_transform(), fontsize=9, color='#555555')
                continue
            seeds = defaultdict(list)
            for row in by_arm[arm]:
                value = row.get(metric)
                if value is None and metric == 'scheduler_calls':
                    value = row.get('macro_steps')
                if value is None:
                    continue
                seeds[row.get('training_seed')].append(float(value)*factor)
            points = [float(np.mean(values)) for values in seeds.values()]
            if not points:
                continue
            mean = float(np.mean(points))
            ax.bar(index, mean, color=colors[index], alpha=.8, width=.65)
            offsets = np.linspace(-.13, .13, len(points)) if len(points) > 1 else [0.]
            ax.scatter(index+np.asarray(offsets), points, c='#222222', s=22, zorder=3)
            numeric.append(dict(arm=arm, metric=metric, mean=mean, per_training_seed_means=points))
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        if metric in ('cluster_safe_success', 'removal', 'removal_auc_180'):
            ax.set_ylim(0, 105)
    fig.suptitle('EXP0053 development pilot | 4 paired scenes | 3 training seeds per learned variant', fontsize=13)
    fig.text(.5, .015, 'Dots: each training seed averaged over the SAME 4 scenes; controls: one fixed policy. No confidence-interval or superiority claim.\n'
             'N=1 matches catalytic-rate proxy only. Shared heuristics/safety projection are not learning gains. Generic NumPy runtime only.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .07, 1, .95])
    folder = args.root / 'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'pilot_comparison.png', dpi=160)
    fig.savefig(folder / 'pilot_comparison.svg')
    plt.close(fig)
    (folder / 'plotted_values.json').write_text(json.dumps(numeric, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(png=str(folder/'pilot_comparison.png'), cells=len(numeric), exploratory_only=True)))


if __name__ == '__main__':
    main()
