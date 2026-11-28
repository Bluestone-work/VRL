"""Analyse Phase B configuration runs: four metrics, paired effects, planner quality, verdicts."""
from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PRIMARY = [('success', 'Navigation success', 'rate'), ('safe_success', 'Safe (contact-free) success', 'rate'),
           ('T_last_deadline_scored_s', 'Last target arrival, deadline-scored (s)', 'mean'),
           ('conflict', 'Tasks with a spacing violation', 'rate')]
AUX = ['lost_any', 'obstacle_collision', 'wall_contact_robot_s', 'mean_target_arrival_s', 'deployment_time_s',
       'total_cluster_time_s', 'spacing_violation_pair_s', 'violation_per_exposed_pair_s', 'min_inter_cluster_distance_mm',
       'tpg_hold_robot_s', 'shield_intervention_steps', 'wall_projection_rewrite_steps', 'target_coverage', 'deadlock']
HYP = {'H1': ('success', +1, 0.05), 'H2': ('T_last_deadline_scored_s', -1, 10.), 'H3': ('conflict', -1, 0.05),
       'H4': ('safe_success', +1, 0.05)}


def load(path):
    rows, errors = [], []
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        (errors if 'error' in r else rows).append(r)
    return rows, errors


def value(r, key):
    if key == 'n_clusters':
        return float(r['configuration']['n'])
    v = r['executed'].get(key)
    return np.nan if v is None else float(v)


def two_level(values_by_group, stat=np.nanmean, B=10000, seed=0):
    """Bootstrap: resample groups (territories), then items within each sampled group."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(v, float) for v in values_by_group.values() if len(v)]
    point = stat(np.concatenate(groups)) if groups else np.nan
    if len(groups) < 2:
        return point, np.nan, np.nan
    out = np.empty(B)
    G = len(groups)
    for b in range(B):
        pick = rng.integers(0, G, G)
        out[b] = stat(np.concatenate([groups[g][rng.integers(0, len(groups[g]), len(groups[g]))] for g in pick]))
    return float(point), float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n10, n01
    k = min(n01, n10)
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, n10, n01


def by_arm(rows):
    out = defaultdict(dict)
    for r in rows:
        out[r['arm']][r['base_task_hash']] = r
    return out


def arm_table(arms, keys):
    table = {}
    for arm, d in arms.items():
        rec = {'n_tasks': len(d)}
        for key in keys + ['n_clusters']:
            groups = defaultdict(list)
            for r in d.values():
                groups[r['task']['anatomy']].append(value(r, key))
            m, lo, hi = two_level(groups)
            rec[key] = dict(mean=m, lo=lo, hi=hi)
        rec['planning_time_s_median'] = float(np.median([r['configuration']['planning_time_s'] for r in d.values()]))
        rec['n_distribution'] = {str(k): int(sum(r['configuration']['n'] == k for r in d.values())) for k in range(1, 5)}
        table[arm] = rec
    return table


def paired(arms, arm, ref, key):
    common = sorted(set(arms[arm]) & set(arms[ref]))
    groups = defaultdict(list)
    for h in common:
        a, b = arms[arm][h], arms[ref][h]
        groups[a['task']['anatomy']].append(value(a, key) - value(b, key))
    m, lo, hi = two_level(groups)
    out = dict(n=len(common), diff=m, lo=lo, hi=hi)
    if key in ('success', 'safe_success', 'conflict', 'lost_any'):
        p, w, l = mcnemar([value(arms[arm][h], key) > .5 for h in common], [value(arms[ref][h], key) > .5 for h in common])
        out.update(mcnemar_p=p, arm_only=w, ref_only=l)
    return out


def verdict(d, sign, minimal):
    if not np.isfinite(d['lo']):
        return 'INCONCLUSIVE'
    if sign > 0:
        if d['lo'] > 0 and d['diff'] >= minimal:
            return 'PASS'
        if d['hi'] < 0:
            return 'FAIL'
    else:
        if d['hi'] < 0 and d['diff'] <= -minimal:
            return 'PASS'
        if d['lo'] > 0:
            return 'FAIL'
    return 'INCONCLUSIVE'


def planner_quality(arms):
    if 'proposed' not in arms or 'exact' not in arms:
        return None
    gaps, agree, times = [], 0, []
    common = sorted(set(arms['proposed']) & set(arms['exact']))
    for h in common:
        jl = arms['proposed'][h]['configuration']['predicted']['J']; je = arms['exact'][h]['configuration']['predicted']['J']
        gaps.append((jl - je) / max(je, 1e-9))
        agree += arms['proposed'][h]['configuration']['n'] == arms['exact'][h]['configuration']['n']
        times.append((arms['proposed'][h]['configuration']['planning_time_s'], arms['exact'][h]['configuration']['planning_time_s']))
    gaps = np.asarray(gaps)
    certs = [arms['exact'][h]['configuration']['certificate'] for h in common]
    return dict(n=len(common), gap_median=float(np.median(gaps)), gap_max=float(gaps.max()),
                within_5pct=float(np.mean(gaps <= .05 + 1e-12)), negative_gap_tasks=int((gaps < -1e-9).sum()),
                n_agreement=float(agree / max(len(common), 1)), exact_certificates={c: certs.count(c) for c in set(certs)},
                planning_time_s_median=dict(local=float(np.median([t[0] for t in times])), exact=float(np.median([t[1] for t in times]))))


def prediction_fidelity(arms, arm='proposed'):
    if arm not in arms:
        return None
    p, e = [], []
    for r in arms[arm].values():
        if r['executed']['success']:
            p.append(r['configuration']['predicted']['T_pred']); e.append(r['executed']['T_last_success_s'])
    if len(p) < 3:
        return None
    p, e = np.asarray(p), np.asarray(e)
    return dict(n=len(p), pearson_r=float(np.corrcoef(p, e)[0, 1]), mean_executed_minus_predicted_s=float(np.mean(e - p)),
                median_ratio=float(np.median(e / p)))


def choose_fixed_n(arms, dep):
    best = None
    for k in range(1, 5):
        arm = f'fixedN{k}_{dep}'
        if arm not in arms:
            continue
        s = np.mean([value(r, 'success') for r in arms[arm].values()])
        t = np.mean([value(r, 'T_last_deadline_scored_s') for r in arms[arm].values()])
        key = (-s, t)
        if best is None or key < best[0]:
            best = (key, k, s, t)
    return None if best is None else dict(n=best[1], success=float(best[2]), T_last=float(best[3]),
                                          rule='max success, then min mean deadline-scored T_last')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', default='fixedN_root')
    a = ap.parse_args()
    rows, errors = load(a.runs)
    arms = by_arm(rows)
    keys = [k for k, _, _ in PRIMARY] + AUX
    res = dict(runs=str(a.runs), rows=len(rows), errors=len(errors),
               error_examples=[e['error'][-300:] for e in errors[:3]], arms=arm_table(arms, keys))
    ref = a.reference if a.reference in arms else None
    if ref:
        res['paired_vs_reference'] = {arm: {k: paired(arms, arm, ref, k) for k, _, _ in PRIMARY + [('lost_any', '', '')]}
                                      for arm in arms if arm != ref}
        if 'proposed' in arms:
            res['verdicts'] = {h: dict(metric=k, **res['paired_vs_reference']['proposed'][k],
                                       verdict=verdict(res['paired_vs_reference']['proposed'][k], s, mn))
                               for h, (k, s, mn) in HYP.items()}
    if 'proposed' in arms:
        res['paired_vs_proposed'] = {arm: {k: paired(arms, arm, 'proposed', k) for k, _, _ in PRIMARY}
                                     for arm in arms if arm != 'proposed'}
    pq = planner_quality(arms)
    if pq:
        res['planner_quality'] = pq
        res.setdefault('verdicts', {})['H5'] = dict(metric='J gap local vs exact', **pq,
                                                    verdict='PASS' if pq['within_5pct'] >= .95 else 'FAIL')
    res['prediction_fidelity'] = prediction_fidelity(arms)
    res['fixed_n_selection'] = {dep: choose_fixed_n(arms, dep) for dep in ('root', 'opt')}
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ('rows', 'errors', 'fixed_n_selection')}, default=float))
    for arm, rec in res['arms'].items():
        print(f"{arm:20s} n={rec['n_tasks']:3d} succ={rec['success']['mean']:.3f} safe={rec['safe_success']['mean']:.3f} "
              f"T={rec['T_last_deadline_scored_s']['mean']:6.1f} conf={rec['conflict']['mean']:.3f} "
              f"lost={rec['lost_any']['mean']:.3f} N={rec['n_clusters']['mean']:.2f} dep={rec['deployment_time_s']['mean']:5.1f}")
    for h, v in res.get('verdicts', {}).items():
        print(h, v.get('metric'), {k: (round(v[k], 4) if isinstance(v.get(k), float) else v.get(k)) for k in ('diff', 'lo', 'hi', 'within_5pct', 'verdict') if k in v})


if __name__ == '__main__':
    main()
