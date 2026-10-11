"""EXP0093 summary: global x local factorial (hard matrix + V5 subset), merged with EXP0090/0091 references.
Writes research/validation/EXP0093_20261011/summary.json, factorial tables (MD/CSV) and paper figures."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

V = Path('research/validation/EXP0093_20261011'); FIG = Path('research/figures/EXP0091_paper')
G_ORDER = ['A_TPG', 'STPG', 'OT', 'NONE']; L_ORDER = ['ours', 'online', 'pursuit', 'pac', 'mppi', 'cbf', 'turbo']
G_LAB = {'A_TPG': '测地分配+TPG（本文）', 'STPG': '测地分配+STPG [AAAI25]', 'OT': 'OT 分配 [CDC25]', 'NONE': '测地分配，无协调'}
L_LAB = {'ours': '预测辅助选择（本文，学习预测）', 'online': '预测辅助选择（在线估计）', 'pursuit': '路线追踪+减速',
         'pac': 'PAC-NMPC [CASE25]', 'mppi': 'MPPI [ICRA16]', 'cbf': 'CBF-QP [TAC17]', 'turbo': 'Turbo PPO [NMI26]'}
PANELS = ['low_delay', 'moderate_delay', 'high_delay', 'strong_flow', 'variable_response', 'ood_flow_delay']
P_LAB = {'low_delay': '低延迟', 'moderate_delay': '中延迟', 'high_delay': '高延迟', 'strong_flow': '强流',
         'variable_response': '动力学变化', 'ood_flow_delay': '分布外'}
METRICS = ['strict_success', 'task_success', 'removal', 'removal_auc', 't90_300', 'wall_contact_s', 'wall_ge_1s',
           'spacing_violation_pair_s', 'coupling_exposure', 'path_mm', 'tpg_hold_s']


def load(f):
    by = defaultdict(dict)
    if not Path(f).exists():
        return by
    for l in open(f):
        r = json.loads(l)
        if r['method'] != 'fac':
            continue
        by[(r['cell'], r['anatomy'], r['clusters'], r['seed'])][r['method_arg']] = r
    return by


def boot(d, B=4000):
    d = np.asarray(d, float); rng = np.random.default_rng(0); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agg(by, arm, keys):
    rs = [by[k][arm] for k in keys]
    return {m: float(np.mean([float(r[m]) for r in rs])) for m in METRICS} | dict(n=len(rs))


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']; plt.rcParams['axes.unicode_minus'] = False
    S = {}
    for proto, f in (('hard', V/'hard/episodes.jsonl'), ('v5', V/'v5/episodes.jsonl')):
        by = load(f)
        arms = sorted({a for d in by.values() for a in d})
        keys = [k for k, d in by.items() if all(a in d for a in arms)]
        if not keys:
            continue
        cells = sorted({k[0].split('|')[0] for k in keys})
        T = {}
        for a in arms:
            T[a] = dict(ALL=agg(by, a, keys))
            for c in cells:
                T[a][c] = agg(by, a, [k for k in keys if k[0].split('|')[0] == c])
            if proto == 'hard':
                for topo in ('seen_topology', 'unseen_topology'):
                    T[a][topo] = agg(by, a, [k for k in keys if k[0].split('|')[1] == topo])
        ref = 'A_TPG+ours'
        pairs = {a: {c: boot([by[k][ref]['strict_success']-by[k][a]['strict_success'] for k in keys
                              if c == 'ALL' or k[0].split('|')[0] == c]) for c in ['ALL']+cells} for a in arms if a != ref}
        hash_bad = sum(len({r['scenario_hash'] for r in by[k].values()}) > 1 for k in keys)
        S[proto] = dict(arms=arms, n_keys=len(keys), cells=cells, table=T, ours_minus=pairs, hash_mismatch=hash_bad)
        # markdown
        L = [f'# EXP0093 {proto}: 全局层 × 局部层（{len(keys)} 个共同配对场景，scenario_hash 不一致 {hash_bad}）', '',
             '| 全局层 | 局部层 | Strict % | 清除率 % | AUC % | T90_300 s | 壁 robot·s | 壁≥1 s % | 间距 pair·s | 耦合 | '+' | '.join(P_LAB.get(c, c) for c in cells)+' | 本文−该组合 Strict pp [95% CI] |',
             '|---|---|'+'---|'*(10+len(cells))]
        for a in sorted(arms, key=lambda x: (G_ORDER.index(x.split('+')[0]), L_ORDER.index(x.split('+')[1]))):
            g, l = a.split('+'); v = T[a]['ALL']
            pc = ' | '.join(f"{100*T[a][c]['strict_success']:.1f}" for c in cells)
            dd = pairs.get(a, {}).get('ALL'); ds = f'{100*dd[0]:+.1f} [{100*dd[1]:+.1f}, {100*dd[2]:+.1f}]' if dd else '—'
            L.append(f"| {G_LAB[g]} | {L_LAB[l]} | {100*v['strict_success']:.1f} | {100*v['removal']:.1f} | {100*v['removal_auc']:.1f} | "
                     f"{v['t90_300']:.0f} | {v['wall_contact_s']:.2f} | {100*v['wall_ge_1s']:.1f} | {v['spacing_violation_pair_s']:.3f} | "
                     f"{v['coupling_exposure']:.2f} | {pc} | {ds} |")
        (V/f'{proto}_TABLE.md').write_text('\n'.join(L)+'\n')
        # heatmap of strict (global x local)
        gs = [g for g in G_ORDER if any(a.startswith(g+'+') for a in arms)]
        ls = [l for l in L_ORDER if any(a.endswith('+'+l) for a in arms)]
        M = np.full((len(gs), len(ls)), np.nan)
        for i, g in enumerate(gs):
            for j, l in enumerate(ls):
                if f'{g}+{l}' in T:
                    M[i, j] = 100*T[f'{g}+{l}']['ALL']['strict_success']
        fig, ax = plt.subplots(figsize=(1.35*len(ls)+2.6, .62*len(gs)+1.6))
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=max(80, np.nanmax(M)))
        for i in range(len(gs)):
            for j in range(len(ls)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=9, color='w' if M[i, j] < 50 else 'k')
        ax.set_xticks(range(len(ls))); ax.set_xticklabels([L_LAB[l] for l in ls], rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(gs))); ax.set_yticklabels([G_LAB[g] for g in gs], fontsize=8)
        ax.set_title(f"Strict 成功率 %（{'困难矩阵' if proto == 'hard' else 'V5 开发集'}，{len(keys)} 个配对场景）", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=.03); fig.tight_layout()
        for e in ('png', 'pdf'):
            fig.savefig(FIG/f'fig_factorial_{proto}.{e}', dpi=300)
        plt.close(fig)
        # per-panel bars for the A_TPG global row + best external combos
        show = [a for a in ['A_TPG+ours', 'A_TPG+online', 'A_TPG+pursuit', 'STPG+pac', 'A_TPG+pac', 'A_TPG+mppi', 'A_TPG+cbf', 'A_TPG+turbo', 'OT+pursuit'] if a in T]
        fig, ax = plt.subplots(figsize=(11, 3.6)); W = .85/len(show)
        for j, a in enumerate(show):
            g, l = a.split('+')
            ax.bar(np.arange(len(cells))+j*W, [100*T[a][c]['strict_success'] for c in cells], W, label=f'{L_LAB[l]} / {G_LAB[g]}')
        ax.set_xticks(np.arange(len(cells))+.42-W/2); ax.set_xticklabels([P_LAB.get(c, c) for c in cells])
        ax.set_ylabel('Strict 成功率 (%)'); ax.legend(fontsize=6.5, ncol=3, frameon=False); fig.tight_layout()
        for e in ('png', 'pdf'):
            fig.savefig(FIG/f'fig_panels_{proto}.{e}', dpi=300)
        plt.close(fig)
    (V/'summary.json').write_text(json.dumps(S, indent=1))
    print({k: (v['n_keys'], len(v['arms'])) for k, v in S.items()})


if __name__ == '__main__':
    main()
