"""Tables and figures for benchmark v3 (obstacles): training curves, main comparison, ablations, trade-offs.

usage: analyze_obstacles.py [--out research/figures/V3_20261006] [--table research/validation/OBST_BENCH_20261006/SUMMARY.md]
Reads research/runs/OBST_DRL_20261006/*/log.jsonl and research/validation/OBST_BENCH_20261006/*.jsonl.
Method tags in the benchmark files: rule_noavoid, rule_apf, <variant>_s<k> (variant = tres, gru, mlp, tvel, tdir, tnodr).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '.')
from scripts.make_manuscript_figures import C  # noqa: E402  (shared palette and CJK font setup)

RUNS = Path('research/runs/OBST_DRL_20261006'); BENCH = Path('research/validation/OBST_BENCH_20261006')
NAMES = {'rule_noavoid': '路线追踪（无避障）', 'rule_apf': '路线追踪 + APF', 'mlp': 'MLP 残差 PPO',
         'gru': 'GRU 残差 PPO', 'tres': 'T-IRPPO（本文）', 'tdir': 'Transformer PPO（无规则先验）',
         'tnodr': 'T-IRPPO 无域随机化', 'tvel': 'T-IRPPO + 障碍速度输入'}
ORDER = ['rule_noavoid', 'rule_apf', 'tdir', 'mlp', 'gru', 'tvel', 'tnodr', 'tres']
COLS = {'rule_noavoid': '#cdd4db', 'rule_apf': C['priv'], 'tdir': C['rl'], 'mlp': '#e0a33c', 'gru': '#8a5fa8',
        'tnodr': '#7fb3c8', 'tvel': '#2f8f6b', 'tres': C['ours']}


def variant(tag):
    return tag if tag.startswith('rule') else tag.rsplit('_s', 1)[0]


def load_bench():
    rows = []
    for f in sorted(BENCH.glob('*.jsonl')):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return [r for r in rows if 'error' not in r and r.get('sensing_model', 'noise') == 'noise']


def boot(d, n=4000):
    d = np.asarray(d, float)
    b = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def metrics(X):
    f = lambda k: np.array([float(r[k] or 0) for r in X])
    t100 = [r['t100_s'] for r in X if r['t100_s'] is not None]
    return dict(n=len(X), safe=100*f('cluster_safe_success').mean(), raw=100*f('task_success').mean(),
                removal=100*f('removal').mean(), obs=100*np.mean([r['obstacle_events'] > 0 for r in X]),
                obs_s=f('obstacle_events_static').mean(), obs_d=f('obstacle_events_dynamic').mean(),
                wall=100*np.mean([r['wall_contact_s'] >= 1 for r in X]), lost=100*np.mean([r['lost'] > 0 for r in X]),
                t100=float(np.mean(t100)) if t100 else np.nan, auc=100*f('removal_auc').mean(),
                path=f('path_mm').mean(), clear=np.nanmedian([r['obstacle_min_clearance_mm'] if r['obstacle_min_clearance_mm'] is not None else np.nan for r in X]))


def table(rows, out):
    by = defaultdict(list)
    for r in rows:
        by[(variant(r['method']), r['clusters'])].append(r)
    seeds = defaultdict(set)
    for r in rows:
        seeds[variant(r['method'])].add(r['method'])
    L = ['# Benchmark v3 (microscope-visible obstacles): dev scenes, 14 anatomies x 30 seeds', '',
         'Learned variants pool all seeds (number of seeds in brackets). Safe Success: all clots cleared, wall < 1 robot-s, '
         'no obstacle collision, no lost cluster, no cluster contact, spacing compliant.', '',
         '| N | method | episodes | Safe % | Raw % | removal % | obstacle-hit eps % | static hits/ep | moving hits/ep | wall>=1 s % | lost % | T100 s | AUC % | path mm |',
         '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in (1, 2, 3):
        for v in ORDER:
            X = by.get((v, n))
            if not X:
                continue
            m = metrics(X)
            L.append(f"| {n} | {NAMES[v]} [{len(seeds[v])}] | {m['n']} | {m['safe']:.1f} | {m['raw']:.1f} | {m['removal']:.1f} | "
                     f"{m['obs']:.1f} | {m['obs_s']:.2f} | {m['obs_d']:.2f} | {m['wall']:.1f} | {m['lost']:.1f} | {m['t100']:.0f} | {m['auc']:.1f} | {m['path']:.0f} |")
    L += ['', '## Paired differences in Safe Success vs route pursuit + APF (pp, 95 % bootstrap CI over scenes; learned = seed mean)', '',
          '| method | N=1 | N=2 | N=3 | all |', '|---|---|---|---|---|']
    base = {(r['anatomy'], r['seed'], r['clusters']): float(r['cluster_safe_success']) for r in rows if r['method'] == 'rule_apf'}
    for v in ORDER:
        if v == 'rule_apf':
            continue
        acc = defaultdict(list)
        for r in rows:
            if variant(r['method']) == v:
                acc[(r['anatomy'], r['seed'], r['clusters'])].append(float(r['cluster_safe_success']))
        cells = []
        for N in (1, 2, 3, None):
            k = [x for x in acc if x in base and (N is None or x[2] == N)]
            if not k:
                cells.append('—'); continue
            m, lo, hi = boot([100*(np.mean(acc[x])-base[x]) for x in k])
            cells.append(f'{m:+.1f} [{lo:+.1f}, {hi:+.1f}]')
        L.append(f'| {NAMES[v]} | '+' | '.join(cells)+' |')
    Path(out).write_text('\n'.join(L)+'\n')
    return '\n'.join(L)


def fig_training(out):
    logs = defaultdict(list)
    for d in sorted(RUNS.glob('*_s*')):
        f = d/'log.jsonl'
        if f.exists():
            logs[variant(d.name)].append([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for v, runs in logs.items():
        grid = np.linspace(0, min(r[-1]['agent_steps'] for r in runs), 60)
        for ax, key in zip(axes, ('safe', 'collided', 'success')):
            Y = []
            for r in runs:
                x = np.array([e['agent_steps'] for e in r if e[key] is not None]); y = np.array([e[key] for e in r if e[key] is not None])
                Y.append(np.interp(grid, x, y))
            Y = 100*np.array(Y); m = Y.mean(0)
            ax.plot(grid/1e6, m, color=COLS[v], lw=1.3, label=f'{NAMES[v]} ({len(runs)})')
            if len(runs) > 1:
                ax.fill_between(grid/1e6, Y.min(0), Y.max(0), color=COLS[v], alpha=.15, lw=0)
    for ax, t in zip(axes, ('训练回合安全成功率 (%)', '训练回合撞障碍率 (%)', '训练回合完成率 (%)')):
        ax.set_xlabel('agent steps (M)'); ax.set_title(t, fontsize=8)
    axes[0].legend(frameon=False, fontsize=6, loc='lower right')
    fig.tight_layout(); fig.savefig(out/'v3_training_curves.png', bbox_inches='tight'); plt.close(fig)


def fig_main(rows, out):
    by = defaultdict(list)
    for r in rows:
        by[(variant(r['method']), r['clusters'])].append(r)
    present = [v for v in ORDER if any((v, n) in by for n in (1, 2, 3))]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    w = .8/len(present)
    for k, (key, lab) in enumerate((('safe', '安全成功率 (%)'), ('obs', '撞障碍回合 (%)'), ('wall', '壁接触 ≥1 s 回合 (%)'))):
        ax = axes[k]
        for s, v in enumerate(present):
            y = [metrics(by[(v, n)])[key] if (v, n) in by else np.nan for n in (1, 2, 3)]
            ax.bar(np.arange(3)+(s-(len(present)-1)/2)*w, y, w, color=COLS[v], label=NAMES[v], edgecolor='w', lw=.3)
        ax.set_xticks(range(3)); ax.set_xticklabels(['N=1', 'N=2', 'N=3']); ax.set_title(lab, fontsize=8)
    axes[0].legend(frameon=False, fontsize=5.8, loc='upper left', bbox_to_anchor=(0, 1.42), ncol=4)
    fig.tight_layout(); fig.savefig(out/'v3_main_results.png', bbox_inches='tight'); plt.close(fig)


def fig_tradeoff(rows, out):
    by = defaultdict(list)
    for r in rows:
        by[variant(r['method'])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for v in ORDER:
        if v not in by:
            continue
        m = metrics(by[v])
        axes[0].scatter(m['obs'], m['wall'], s=40, color=COLS[v], edgecolor='k', lw=.4, zorder=3)
        axes[0].annotate(NAMES[v], (m['obs'], m['wall']), fontsize=6, xytext=(3, 3), textcoords='offset points')
        axes[1].scatter(m['t100'], m['safe'], s=40, color=COLS[v], edgecolor='k', lw=.4, zorder=3)
        axes[1].annotate(NAMES[v], (m['t100'], m['safe']), fontsize=6, xytext=(3, 3), textcoords='offset points')
    axes[0].set_xlabel('撞障碍回合 (%)'); axes[0].set_ylabel('壁接触 ≥1 s 回合 (%)'); axes[0].set_title('避障与贴壁的权衡（越靠左下越好）', fontsize=8)
    axes[1].set_xlabel('完全清除用时 T100 (s)'); axes[1].set_ylabel('安全成功率 (%)'); axes[1].set_title('安全与效率（越靠左上越好）', fontsize=8)
    fig.tight_layout(); fig.savefig(out/'v3_tradeoffs.png', bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('research/figures/V3_20261006'))
    ap.add_argument('--table', type=Path, default=BENCH/'SUMMARY.md')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = load_bench()
    print(table(rows, a.table))
    fig_training(a.out)
    if rows:
        fig_main(rows, a.out); fig_tradeoff(rows, a.out)


if __name__ == '__main__':
    main()
