"""Summarize verified fixed-target navigation, CCBS and paired planar evaluations."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_table(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def scores(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(20261127).choice(values, (10000, len(values)), replace=True)
    return [round(float(x), 4) for x in (values.mean(), *np.quantile(draws.mean(1), [.025, .975]))]


def report(research):
    root = research / 'navigation_budget_20261127'
    arms = {}
    budget_rows, pair_rows = [], []
    for arm in ('geodesic_tpg', 'geodesic_off', 'geodesic_ccbs', 'linear_sum_tpg', 'geodesic_portal'):
        rows = json.loads((root / f'{arm}.json').read_text())['rows']
        indexed = {(r['seed'], r['clusters']): r for r in rows}
        required = {(s, n) for s in range(2944200000, 2944200006) for n in (1, 2, 3)}
        if set(indexed) != required or len(rows) != len(required):
            raise ValueError(f'{arm}: incomplete fixed-target matrix')
        reference = indexed if arm == 'geodesic_portal' else arms.get('geodesic_tpg', indexed)
        for (seed, n), row in indexed.items():
            first = indexed[seed, 3]
            if (row['scenario_hash'] != first['scenario_hash'] or
                row['selected_start_indices'] != first['selected_start_indices'][:n] or
                row['scenario_hash'] != reference[seed, n]['scenario_hash']):
                raise ValueError(f'{arm}: scene or start mismatch {seed}, {n}')
            if arm in ('geodesic_off', 'geodesic_ccbs') and row['assignment'] != reference[seed, n]['assignment']:
                raise ValueError('coordinator changed assigned routes')
        arms[arm] = indexed
        for n in (1, 2, 3):
            group = [r for r in rows if r['clusters'] == n]
            for budget, limit in [('common_deadline', 300.), ('common_cluster_seconds', 300. / n)]:
                # This is administrative censoring of the same navigation
                # trajectories. Each deployed cluster reserves one slot.
                times = [r['last_arrival_s'] for r in group]
                reached = [t is not None and t <= limit + 1e-9 for t in times]
                sync = [a and r['synchronized_success'] for a, r in zip(reached, group)]
                budget_rows.append(dict(arm=arm, clusters=n, budget=budget, deadline_s=limit,
                    available_cluster_s=n*limit, scenes=len(group), covered=sum(reached), sync30=sum(sync),
                    mean_deadline_scored_s=np.mean([t if ok else limit for t, ok in zip(times, reached)])))
    for key, a in sorted(arms['geodesic_tpg'].items()):
        b = arms['linear_sum_tpg'][key]
        pair_rows.append(dict(seed=key[0], clusters=key[1], scenario_hash=a['scenario_hash'],
            starts=a['selected_start_indices'], different=a['assignment'] != b['assignment'],
            geodesic_tours=a['assignment'], linear_sum_tours=b['assignment'],
            geodesic_max_mm=max(a['route_lengths_mm']), linear_sum_max_mm=max(b['route_lengths_mm']),
            geodesic_sum_mm=sum(a['route_lengths_mm']), linear_sum_sum_mm=sum(b['route_lengths_mm']),
            geodesic_scored_s=a['deadline_scored_last_arrival_s'],
            linear_sum_scored_s=b['deadline_scored_last_arrival_s']))
    write_table(root/'budgets.csv', budget_rows)
    write_table(root/'allocations.csv', pair_rows)
    groups = [r for r in budget_rows if r['arm']=='geodesic_tpg' and r['budget']=='common_deadline']
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), layout='constrained')
    ns = [1, 2, 3]
    for field, label, marker in [('covered','All three reached','o'),('sync30','Within 30 s','s')]:
        axes[0].plot(ns, [r[field]/r['scenes'] for r in groups], marker=marker, label=label)
    axes[0].set(ylabel='Fraction of six layouts', ylim=(-.05,1.05))
    axes[0].legend()
    for s in range(2944200000, 2944200006):
        axes[1].plot(ns,[arms['geodesic_tpg'][s,n]['deadline_scored_last_arrival_s'] for n in ns],
                     color='0.75',linewidth=.8)
    axes[1].plot(ns,[r['mean_deadline_scored_s'] for r in groups], color='black',marker='o',label='Mean')
    axes[1].set_ylabel('Deadline-scored time (s)')
    axes[1].legend()
    for ax in axes:
        ax.set(xlabel='Deployed clusters',xticks=ns)
    fig.savefig(root/'cluster_comparison.png',dpi=180)
    plt.close(fig)

    runs = research/'ccbs_verified_20261127'
    specs = json.loads((runs/'manifest.json').read_text())['specs']
    actual = [json.loads(line) for line in (runs/'episodes.jsonl').read_text().splitlines()]
    expected_keys = Counter((m,p,j[2],j[3],j[4],j[5],j[6]) for m,p,j in specs)
    actual_keys = Counter(tuple(r[k] for k in ('method','panel','split','suite','anatomy','clusters','seed')) for r in actual)
    if expected_keys != actual_keys or (runs/'failures.jsonl').read_text().strip():
        raise ValueError('CCBS matrix has missing, extra or failed runs')
    panels=[]
    for panel in sorted({r['panel'] for r in actual}):
        sub = [r for r in actual if r['panel']==panel]
        panels.append(dict(panel=panel, episodes=len(sub), lysis_success=sum(r['task_success'] for r in sub),
            infeasible=sum(r['infeasible'] for r in sub), node_limit=sum(r['node_limit_reached'] for r in sub),
            mean_hold_s=np.mean([r['tpg_hold_s'] for r in sub]),
            mean_spacing_pair_s=np.mean([r['spacing_violation_pair_s'] for r in sub])))
    write_table(runs/'panels.csv',panels)
    planar = json.loads((research/'mr_nav_2d_paired_20261127.json').read_text())
    methods = {m:{r['lane']:r for r in planar['episodes'] if r['method']==m} for m in ('mr_nav','direct_pursuit')}
    a,b=methods.values()
    if len(planar['episodes']) != 2*planar['maps'] or set(a)!=set(b) or len(a)!=planar['maps']:
        raise ValueError('Incomplete planar pair set')
    for i in a:
        if any(a[i][k] != b[i][k] for k in ('map','start','target')):
            raise ValueError('Planar pair mismatch')
    delta=[int(a[i]['success'])-int(b[i]['success']) for i in a]
    geo = arms['geodesic_tpg']
    diff=[geo[s,3]['deadline_scored_last_arrival_s']-geo[s,1]['deadline_scored_last_arrival_s']
          for s in sorted({s for s,n in geo})]
    summary=dict(mca_n3_minus_n1_time_mean_and_paired_ci95_s=scores(diff),
        mr_nav_wins_ties_losses=[delta.count(v) for v in (1,0,-1)],
        mr_nav_success_delta_and_paired_ci95=scores(delta),
        note='Percentile bootstrap over paired layouts, without resampling training seeds.')
    (root/'paired_effects.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research',type=Path,default=Path('research'))
    report(parser.parse_args().research)
