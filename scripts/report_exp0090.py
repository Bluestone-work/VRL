"""EXP0090 report: per-panel tables of every registered metric, seed statistics, paired bootstrap differences, figures.

usage: report_exp0090.py --runs DIR [DIR ...] --out DIR
Arms are labelled from (method, checkpoint): A_nav_off, B_nav_s71xx, C_belief_s71xx, D_sel_ens, D_sel_s{0,1,2},
E_sel_fb, F_sel_online, rules. T50/T90/T100 are reported both as *_300 (not reached -> 300 s) and as reached fraction +
conditional mean over reached episodes. Pairing key = (panel cell, anatomy, N, scene seed); scenario_hash must match.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MAIN = ['fixed_settle', 'adaptive_settle', 'switch_settle', 'stpg', 'pac_nmpc', 'A_nav_off', 'B_nav', 'C_belief',
        'F_sel_online', 'D_sel', 'D_sel_ens', 'E_sel_fb', 'I_sel_online_inv', 'G_sel_inv', 'H_sel_inv_fb']
METRICS = ['strict_success', 'task_success', 'relaxed_success', 'removal', 'clot_frac', 'removal_auc', 'removal_rate_per_min',
           't50_300', 't90_300', 't100_300', 'reached90', 'cond_t90', 'wall_contact_s', 'max_continuous_wall_contact_s',
           'wall_ge_1s', 'wall_ge_5s', 'spacing_violation_pair_s', 'min_pair_distance_mm', 'robot_pair_contact_s',
           'coupling_exposure', 'tpg_hold_s', 'lost', 'path_mm', 'timeout', 'ctrl_ms_per_step',
           'diag_guard_override_frac', 'diag_shield_override_frac', 'diag_hold_frac', 'diag_learn_dev_frac', 'diag_learn_dev_mag',
           'sel_learned_takeover_frac', 'sel_fallback_frac', 'sel_stop_frac', 'infer_ms_per_step']
PCT = {'strict_success', 'task_success', 'relaxed_success', 'removal', 'clot_frac', 'reached90', 'wall_ge_1s', 'wall_ge_5s',
       'timeout', 'diag_guard_override_frac', 'diag_shield_override_frac', 'diag_hold_frac', 'diag_learn_dev_frac',
       'sel_learned_takeover_frac', 'sel_fallback_frac', 'sel_stop_frac', 'removal_auc'}


def label(r):
    m, a = r['method'], r.get('method_arg') or ''
    seed = re.search(r'_s(\d+)', a)
    if m == 'nav_off':
        return 'A_nav_off', None
    if m == 'nav':
        return ('C_belief' if 'belief' in a else 'B_nav'), (seed.group(1) if seed else a)
    if m == 'sel_learned':
        return ('D_sel', 'p'+re.search(r'predictor_s(\d)', a).group(1)) if a else ('D_sel_ens', None)
    if m == 'sel_learned_inv':
        return 'G_sel_inv', None
    if m == 'sel_learned_fb_inv':
        return 'H_sel_inv_fb', None
    if m == 'sel_online_inv':
        return 'I_sel_online_inv', None
    if m == 'sel_learned_fb':
        return 'E_sel_fb', None
    if m == 'sel_online':
        return 'F_sel_online', None
    return m, None


def value(r, k):
    if k == 'clot_frac':
        return r['clots_cleared']/max(r['clots'], 1)
    if k == 'reached90':
        return float(r['t90_s'] is not None)
    if k == 'cond_t90':
        return np.nan if r['t90_s'] is None else float(r['t90_s'])
    if k == 'timeout':
        return float(r.get('termination') == 'time_limit')
    v = r.get(k)
    if v is None:
        return np.nan
    return float(v)


def load(dirs):
    rows = []
    for d in dirs:
        for l in open(Path(d)/'episodes.jsonl'):
            r = json.loads(l); r['arm'], r['arm_seed'] = label(r); r['panel'] = r['cell'].split('|')[0]
            r['topology'] = r['cell'].split('|')[1] if '|' in r['cell'] else 'v5'
            r['key'] = (r['cell'], r['anatomy'], r['clusters'], r['seed']); rows.append(r)
    return rows


def boot(d, B=4000, seed=0):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(k, v):
    if not np.isfinite(v):
        return '-'
    return f'{100*v:.1f}' if k in PCT else (f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = load(a.runs)
    # pairing / scenario-hash audit
    by_key = defaultdict(dict); hash_bad = []
    for r in rows:
        by_key[r['key']][(r['arm'], r['arm_seed'])] = r
    for k, d in by_key.items():
        hs = {x['scenario_hash'] for x in d.values()}
        if len(hs) > 1:
            hash_bad.append(k)
    arms = sorted({(r['arm'], r['arm_seed']) for r in rows}, key=lambda x: (MAIN.index(x[0]) if x[0] in MAIN else 99, str(x[1])))
    panels = sorted({r['panel'] for r in rows}, key=lambda p: ['low_delay', 'moderate_delay', 'high_delay', 'strong_flow',
                                                                'variable_response', 'ood_flow_delay'].index(p) if p in
                    ['low_delay', 'moderate_delay', 'high_delay', 'strong_flow', 'variable_response', 'ood_flow_delay'] else 9)
    keys_all = [k for k, d in by_key.items() if all(x in d for x in arms)]
    agg = {}
    for p in panels+['ALL']:
        for topo in ('all', 'seen_topology', 'unseen_topology'):
            ks = [k for k in keys_all if (p == 'ALL' or k[0].split('|')[0] == p) and (topo == 'all' or k[0].split('|')[1] == topo)]
            if not ks:
                continue
            for arm in arms:
                rs = [by_key[k][arm] for k in ks]
                agg[(p, topo, arm)] = {m: float(np.nanmean([value(r, m) for r in rs])) if any(np.isfinite(value(r, m)) for r in rs) else np.nan
                                       for m in METRICS} | dict(n=len(rs))
    # seed-level stats for multi-seed arms
    seed_stats = {}
    for base in ('B_nav', 'C_belief', 'D_sel'):
        seeds = sorted({s for (b, s) in arms if b == base})
        if not seeds:
            continue
        for p in panels+['ALL']:
            vals = {m: [agg[(p, 'all', (base, s))][m] for s in seeds if (p, 'all', (base, s)) in agg] for m in ('strict_success', 't90_300', 'wall_contact_s', 'removal_auc')}
            seed_stats[(base, p)] = {m: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0., v) for m, v in vals.items()}
    # paired differences vs references (per-scene mean over seeds for seed arms)
    def scene_val(k, arm_base, m):
        d = by_key[k]; xs = [value(r, m) for (b, s), r in d.items() if b == arm_base]
        return float(np.mean(xs)) if xs else np.nan
    paired = {}
    bases = sorted({b for b, _ in arms}, key=lambda b: MAIN.index(b) if b in MAIN else 99)
    for ref in ('A_nav_off', 'switch_settle'):
        if ref not in bases:
            continue
        for p in panels+['ALL']:
            ks = [k for k in keys_all if p == 'ALL' or k[0].split('|')[0] == p]
            for b in bases:
                if b == ref:
                    continue
                paired[(ref, p, b)] = {m: boot([scene_val(k, b, m)-scene_val(k, ref, m) for k in ks]) for m in ('strict_success', 't90_300', 'wall_contact_s', 'removal_auc', 'removal')}
    # ---------------- write
    name = lambda arm: arm[0]+(f'_{arm[1]}' if arm[1] else '')
    import csv
    with (a.out/'aggregate.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['panel', 'topology', 'arm', 'n']+METRICS)
        for (p, topo, arm), v in agg.items():
            w.writerow([p, topo, name(arm), v['n']]+[v[m] for m in METRICS])
    (a.out/'aggregate.json').write_text(json.dumps({f'{p}|{t}|{name(arm)}': v for (p, t, arm), v in agg.items()}, indent=1, default=float))
    (a.out/'paired.json').write_text(json.dumps({f'{r}|{p}|{b}': v for (r, p, b), v in paired.items()}, indent=1))
    (a.out/'seed_stats.json').write_text(json.dumps({f'{b}|{p}': v for (b, p), v in seed_stats.items()}, indent=1))
    L = ['# EXP0090 第一轮困难矩阵配对 pilot（开发集，非密封测试）', '',
         f'共同配对场景 {len(keys_all)} 个；scenario_hash 不一致 {len(hash_bad)} 个；臂：{", ".join(name(x) for x in arms)}', '',
         'T90_300 = 未达 90 % 记 300 s；reached90 = 达到比例；cond_t90 = 仅对达到者求均值。PAC-NMPC、STPG 为本地适配实现。', '']
    show = ['strict_success', 'task_success', 'removal', 'removal_auc', 't90_300', 'reached90', 'cond_t90', 'wall_contact_s',
            'max_continuous_wall_contact_s', 'wall_ge_1s', 'spacing_violation_pair_s', 'coupling_exposure', 'tpg_hold_s', 'path_mm', 'ctrl_ms_per_step']
    for p in panels+['ALL']:
        for topo in ('all', 'seen_topology', 'unseen_topology'):
            if (p, topo, arms[0]) not in agg:
                continue
            L += [f'## {p} / {topo} (n={agg[(p, topo, arms[0])]["n"]})', '', '| arm | '+' | '.join(show)+' |', '|---|'+'---|'*len(show)]
            for arm in arms:
                v = agg[(p, topo, arm)]; L.append(f'| {name(arm)} | '+' | '.join(fmt(m, v[m]) for m in show)+' |')
            L.append('')
    L += ['## 种子统计（均值 ± 标准差；逐种子值）', '']
    for (b, p), v in seed_stats.items():
        L.append(f"- {b} {p}: strict {100*v['strict_success'][0]:.1f} ± {100*v['strict_success'][1]:.1f} ({', '.join(f'{100*x:.1f}' for x in v['strict_success'][2])}); "
                 f"T90_300 {v['t90_300'][0]:.1f} ± {v['t90_300'][1]:.1f}; wall {v['wall_contact_s'][0]:.2f} ± {v['wall_contact_s'][1]:.2f}")
    L += ['', '## 配对差值（场景 bootstrap 95% CI；多种子臂先按场景对种子取均值）', '']
    for ref in ('A_nav_off', 'switch_settle'):
        L += [f'### 相对 {ref}', '', '| panel | arm | Δstrict pp | ΔT90_300 s | Δwall s | ΔAUC pp |', '|---|---|---|---|---|---|']
        for (r, p, b), v in paired.items():
            if r != ref:
                continue
            c = lambda m, s=1: f'{s*v[m][0]:+.1f} [{s*v[m][1]:+.1f}, {s*v[m][2]:+.1f}]'
            L.append(f'| {p} | {b} | {c("strict_success", 100)} | {c("t90_300")} | {c("wall_contact_s")} | {c("removal_auc", 100)} |')
        L.append('')
    L += ['## 执行诊断（ALL / all）', '', '| arm | guard override % | shield override % | TPG hold % | learn dev % | learn dev mag | takeover % | fallback % | stop % | infer ms | ctrl ms |', '|---|'+'---|'*10]
    for arm in arms:
        v = agg.get(('ALL', 'all', arm))
        if v:
            L.append(f'| {name(arm)} | '+' | '.join(fmt(m, v[m]) for m in ('diag_guard_override_frac', 'diag_shield_override_frac', 'diag_hold_frac',
                     'diag_learn_dev_frac', 'diag_learn_dev_mag', 'sel_learned_takeover_frac', 'sel_fallback_frac', 'sel_stop_frac', 'infer_ms_per_step', 'ctrl_ms_per_step'))+' |')
    # per-anatomy strict
    L += ['', '## 逐解剖 Strict %（ALL panels）', '']
    anats = sorted({k[1] for k in keys_all})
    L += ['| anatomy | '+' | '.join(name(x) for x in arms)+' |', '|---|'+'---|'*len(arms)]
    for an in anats:
        ks = [k for k in keys_all if k[1] == an]
        L.append(f'| {an} | '+' | '.join(f"{100*np.mean([by_key[k][x]['strict_success'] for k in ks]):.0f}" for x in arms)+' |')
    (a.out/'REPORT.md').write_text('\n'.join(L)+'\n')
    with (a.out/'episodes.csv').open('w', newline='') as f:
        cols = ['arm', 'arm_seed', 'panel', 'topology', 'anatomy', 'clusters', 'seed', 'scenario_hash']+[m for m in METRICS]+['termination']
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) if c in ('arm', 'arm_seed', 'panel', 'topology', 'anatomy', 'clusters', 'seed', 'scenario_hash', 'termination') else value(r, c) for c in cols])
    # ---------------- figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show_arms = [b for b in bases]
    def arm_panel(b, p, m):
        vs = [agg[(p, 'all', x)][m] for x in arms if x[0] == b and (p, 'all', x) in agg]
        return (np.mean(vs), np.std(vs) if len(vs) > 1 else 0.) if vs else (np.nan, 0.)
    for m, lab, sc in (('strict_success', 'Strict success (%)', 100), ('t90_300', 'T90 (s, not reached = 300)', 1),
                       ('wall_contact_s', 'Wall contact (robot-s)', 1), ('removal_auc', 'Removal AUC (%)', 100)):
        fig, ax = plt.subplots(figsize=(1.3*len(panels)+3, 3.6)); W = .8/len(show_arms)
        for j, b in enumerate(show_arms):
            mv = [arm_panel(b, p, m) for p in panels]
            ax.bar(np.arange(len(panels))+j*W, [sc*x[0] for x in mv], W, yerr=[sc*x[1] for x in mv], label=b, capsize=1.5)
        ax.set_xticks(np.arange(len(panels))+.4-W/2); ax.set_xticklabels([p.replace('_', '\n') for p in panels], fontsize=8)
        ax.set_ylabel(lab); ax.legend(fontsize=6, ncol=4, frameon=False); fig.tight_layout()
        for e in ('png', 'pdf'):
            fig.savefig(a.out/f'fig_{m}.{e}', dpi=300)
        plt.close(fig)
    print((a.out/'REPORT.md').read_text()[:3000])


if __name__ == '__main__':
    main()
