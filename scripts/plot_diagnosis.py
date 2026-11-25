"""Diagnostic view of where the learned policy actually fails.

Three questions, three panels: which scene factor drives failure (a), whether the
"no degradation" result is a floor effect (b), and how the methods rank on the
hardest territory (c). Every number is read from a result file; the validated
categorical palette is used because the previous one failed CVD checks.
"""
from __future__ import annotations
import json, os, sys, collections
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
# Validated categorical slots (dataviz reference palette, passes all six checks).
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8a86', '#fcfcfb'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
    'savefig.dpi': 200, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
})
G = ROOT / 'research/runs/NAV_GEN_20261028'
H = ROOT / 'research/runs/HCC_SEEDS12_FIXED_20261028'
S = ROOT / 'research/runs/NAV_SCALE_FIXED_20261029'


def ippo_factors():
    runs = [json.loads((G / f's{s}_train.json').read_text()) for s in (41, 42, 43)]
    eps = [e for r in runs for e in r['episodes']]
    out = {}
    out['Flow regime'] = {k: np.mean([e['navigation_success'] for e in eps if e['flow_mode'] == k])
                          for k in ('weak', 'strong', 'pulsatile')}
    for field, label, order in (('obstacle_density', 'Obstacle density', ('low', 'medium', 'high', 'stress')),
                                ('obstacle_layout', 'Obstacle layout', ('uniform', 'route_encounter')),
                                ('clot_pattern', 'Target distribution', ('historical_sites', 'mixed_pattern', 'random_branches'))):
        by = collections.defaultdict(list)
        for e in eps:
            by[e[field]].append(e['navigation_success'])
        out[label] = {k: float(np.mean(by[k])) for k in order if k in by}
    return out, len(eps)


def scale_points():
    pts = {}
    for n in (1, 2, 3):
        row = {}
        for sp in ('train', 'holdout_test'):
            fs = [(G / f's{s}_{sp}.json') if n == 3 else (S / f'n{n}_s{s}_{sp}.json') for s in (41, 42, 43)]
            v = [json.loads(f.read_text())['overall']['navigation_success_rate'] for f in fs if f.exists()]
            if v: row[sp] = (float(np.mean(v)) * 100, float(np.std(v)) * 100)
        if len(row) == 2: pts[n] = row
    return pts


def hcc_methods():
    rows = []
    for f in ('shard_off0', 'shard_off3', 'shard_off6', 'shard_off9'):
        rows += [r for r in json.loads((H / f'{f}.json').read_text())['rows'] if 'error' not in r]
    by = collections.defaultdict(list)
    for r in rows:
        by[r['method']].append(r['navigation_success'])
    return {k: float(np.mean(v)) * 100 for k, v in by.items()}, len(by['pursuit'])


LABEL = {'weak': 'weak', 'strong': 'strong', 'pulsatile': 'pulsatile',
         'low': 'low', 'medium': 'medium', 'high': 'high', 'stress': 'stress',
         'uniform': 'uniform', 'route_encounter': 'route encounter',
         'historical_sites': 'registered lesion sites', 'mixed_pattern': 'mixed',
         'random_branches': 'random branches'}
MNAME = {'IPPO_S41': 'IPPO (learned)', 'ours_hcc': 'Predictive selector',
         'pursuit': 'Pursuit + settle', 'stpg': 'Switchable TPG', 'pac': 'PAC-NMPC',
         'mppi': 'MPPI', 'cbf': 'CBF-QP', 'online': 'Online selector', 'turbo': 'Transformer PPO'}


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    factors, n_eps = ippo_factors()
    pts = scale_points()
    meth, n_hcc = hcc_methods()
    fig = plt.figure(figsize=(13.2, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1.15], wspace=.42,
                          left=.055, right=.985, top=.84, bottom=.13)

    # (a) which factor drives failure: each row is one factor, dot per level
    ax = fig.add_subplot(gs[0, 0])
    names = list(factors)[::-1]
    for i, fname in enumerate(names):
        lv = factors[fname]
        vals = np.array([v * 100 for v in lv.values()])
        span = vals.max() - vals.min()
        hot = span > 20                        # the one factor that matters
        col = C2 if hot else MUTED
        ax.plot([vals.min(), vals.max()], [i, i], '-', lw=3.2 if hot else 2.0,
                color=col, alpha=.95 if hot else .5, solid_capstyle='round', zorder=2)
        ax.plot(vals, [i] * len(vals), 'o', ms=8, mfc=col if hot else '#ffffff',
                mec=col, mew=1.5, zorder=3)
        # Levels sit close together on the axis, so stagger the labels above and
        # below the line in x order instead of letting them collide.
        for j, (k, v) in enumerate(sorted(lv.items(), key=lambda kv: kv[1])):
            above = j % 2 == 0
            ax.annotate(LABEL.get(k, k), (v * 100, i), textcoords='offset points',
                        xytext=(0, 10 if above else -15), ha='center', fontsize=6.4,
                        color=INK if hot else INK2, zorder=4)
        ax.annotate(f'span {span:.1f} pp', (vals.max(), i), textcoords='offset points',
                    xytext=(9, -1), va='center', fontsize=7.2,
                    color=col, fontweight='bold' if hot else 'normal', zorder=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('IPPO navigation success (%)')
    ax.set_xlim(28, 76)
    ax.set_ylim(-.62, len(names) - .28)
    ax.grid(axis='x', color='#e9e9e5', lw=.6); ax.set_axisbelow(True)
    ax.set_title(f'a  Failure tracks the target distribution,\n   not scene difficulty  (n={n_eps} episodes)',
                 loc='left', fontsize=8.6, color=INK, pad=7)

    # (b) is the "no degradation" result a floor effect?
    ax = fig.add_subplot(gs[0, 1])
    for n, row in pts.items():
        tr, ho = row['train'][0], row['holdout_test'][0]
        col = C1 if n < 3 else C2
        ax.plot([0, 1], [tr, ho], '-o', lw=2.0, ms=8, color=col, mfc=col,
                mec='#ffffff', mew=1.2, zorder=3)
        ax.annotate(f'N={n}', (0, tr), textcoords='offset points', xytext=(-10, 0),
                    ha='right', va='center', fontsize=7.6, color=col, fontweight='bold')
        ax.annotate(f'{tr:.1f}', (0, tr), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=6.8, color=INK2)
        ax.annotate(f'{ho:.1f}', (1, ho), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=6.8, color=INK2)
        ax.annotate(f'{ho - tr:+.1f} pp', (1, ho), textcoords='offset points', xytext=(11, 0),
                    va='center', fontsize=7.2, color=col,
                    fontweight='bold' if abs(ho - tr) < 1 else 'normal')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Trained\nanatomies', 'Held-out\nclasses'], fontsize=7.4)
    ax.set_xlim(-.42, 1.42); ax.set_ylim(40, 104)
    ax.set_ylabel('Navigation success (%)')
    ax.grid(axis='y', color='#e9e9e5', lw=.6); ax.set_axisbelow(True)
    ax.set_title('b  Only the worst configuration\n   fails to degrade', loc='left',
                 fontsize=8.6, color=INK, pad=7)

    # (c) method ranking on the hardest territory
    ax = fig.add_subplot(gs[0, 2])
    order = sorted(meth, key=meth.get)
    y = np.arange(len(order))
    cols = [C2 if m == 'IPPO_S41' else MUTED for m in order]
    ax.barh(y, [meth[m] for m in order], height=.62, color=cols,
            edgecolor=SURF, lw=1.2, zorder=2)
    for i, m in enumerate(order):
        ax.annotate(f'{meth[m]:.1f}', (meth[m], i), textcoords='offset points',
                    xytext=(4, 0), va='center', fontsize=7,
                    color=C2 if m == 'IPPO_S41' else INK2,
                    fontweight='bold' if m == 'IPPO_S41' else 'normal')
    ax.set_yticks(y); ax.set_yticklabels([MNAME.get(m, m) for m in order], fontsize=7.2)
    ax.set_xlabel('Task completion (%)')
    ax.set_xlim(0, 82)
    ax.grid(axis='x', color='#e9e9e5', lw=.6); ax.set_axisbelow(True)
    ax.set_title(f'c  Hardest territory (HCC),\n   {n_hcc} paired scene seeds', loc='left',
                 fontsize=8.6, color=INK, pad=7)
    p = out_dir / 'diagnosis_overview.png'
    fig.savefig(p, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main(ROOT / 'research/figures/DIAGNOSIS_20261029')
