"""NMI figure set v3.  Phase B provides Fig. 2 (graph-based autonomous configuration).

Every number is read from result files under research/NMI_V3/; every panel writes its source data
as CSV next to the figure.  Panels of phases that have not run are not drawn.

usage: make_nmi_figures_v3.py --out research/figures/NMI_v3 [--only fig2]
"""
from __future__ import annotations

import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path

if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / 'research/NMI_V3/CONFIGURATION_RESULTS'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 6.8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'legend.fontsize': 6, 'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'axes.edgecolor': '#3c4250', 'xtick.color': '#3c4250',
    'ytick.color': '#3c4250', 'axes.labelcolor': '#1d2129', 'text.color': '#1d2129', 'savefig.dpi': 450,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'legend.frameon': False, 'axes.titlepad': 3, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic'})
MM = 1 / 25.4
W2 = 180 * MM
# dataviz-validated categorical hues (all pairs CVD dE >= 9.2, normal dE >= 16.3); baselines are neutral grey
CL = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']
PROPOSED, EXACT, BASE = '#2a78d6', '#1d2129', '#8b929c'
INK, INK2, INK3, GRID, VESSEL, TARGET = '#1d2129', '#4b5361', '#7a8290', '#e9ebef', '#cdd2da', '#c25e00'
ARMS = [('fixedN_root', 'Fixed N, root release'), ('fixedN_random', 'Fixed N, random release'),
        ('fixedN_opt', 'Fixed N, optimised release'), ('adaptiveN_root', 'Adaptive N, root release'),
        ('adaptiveN_opt_time', 'Adaptive N, time-only plan'), ('proposed', 'Proposed joint planning'),
        ('exact', 'Exact reference')]
ABLATIONS = [('fixedN_opt', 'w/o adaptive N'), ('adaptiveN_root', 'w/o release optimisation'),
             ('abl_euclid', 'w/o topology-aware allocation'), ('adaptiveN_opt_time', 'w/o interaction cost'),
             ('abl_no_tpg', 'w/o temporal coordination')]
METRICS = [('success', 'Navigation success (%)', 100.), ('safe_success', 'Safe success (%)', 100.),
           ('T_last_deadline_scored_s', 'Last arrival (s)', 1.), ('conflict', 'Tasks with conflict (%)', 100.)]


def flabel(fig, s, x, y):
    fig.text(x, y, s, fontsize=8, fontweight='bold', va='top', ha='left', color='#000')


def ftitle(fig, s, x, y):
    fig.text(x, y, s, fontsize=6.8, va='top', ha='left', color=INK)


def clean(ax, grid='x'):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.45); ax.set_axisbelow(True)


def write_csv(out, name, header, rows):
    with open(out / f'{name}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def load_runs(path):
    arms = defaultdict(dict)
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if 'error' not in r:
            arms[r['arm']][r['base_task_hash']] = r
    return arms


def two_level(groups, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    gs = [np.asarray(v, float) for v in groups.values() if len(v)]
    point = float(np.mean(np.concatenate(gs)))
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(gs), len(gs))
        out[b] = np.mean(np.concatenate([gs[g][rng.integers(0, len(gs[g]), len(gs[g]))] for g in pick]))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def val(r, key):
    v = r['executed'].get(key)
    return np.nan if v is None else float(v)


def save(fig, out, name, caption):
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', pad_inches=.02)
    fig.savefig(out / f'{name}.png', bbox_inches='tight', pad_inches=.02, dpi=300)
    (out / f'{name}_caption.txt').write_text(caption.strip() + '\n')
    plt.close(fig)


def draw_tree(ax, g, a0, a1, color=VESSEL, lw=2.4, z=0, values=None, cmap=None, norm=None):
    pts, rad = g.points, g.radius
    for i, nb in enumerate(g.env.tree.station_graph):
        for j, _ in nb:
            if j <= i:
                continue
            c = color if values is None else cmap(norm(.5 * (values[i] + values[j])))
            ax.plot(pts[[i, j], a0], pts[[i, j], a1], color=c, lw=lw * .5 * (rad[i] + rad[j]) / rad.max() * 2.2 + .4,
                    solid_capstyle='round', zorder=z)
    ax.set_aspect('equal', adjustable='datalim'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plane(g):
    ext = np.ptp(g.points, axis=0)
    a, b = np.argsort(ext)[::-1][:2]
    return int(min(a, b)), int(max(a, b))


def scene_for(task_doc):
    from marl.v3.scene import BaseTask, SceneBuilder
    from marl.v3.vessel_graph import VesselGraph
    sb = SceneBuilder(BaseTask(**task_doc))
    return sb, VesselGraph(sb.probe)


def config_from(row):
    from marl.v3.configuration import Configuration
    c = dict(row['configuration'])
    return Configuration(**{k: c[k] for k in ('method', 'n', 'tours', 'release_stations', 'release_order',
                                             'release_times', 'routes', 'predicted', 'certificate',
                                             'planning_time_s', 'search')})


def draw_config(ax, g, sb, row, a0, a1, lw=1.6):
    draw_tree(ax, g, a0, a1, lw=1.6)
    cfg = row['configuration']
    for c, route in enumerate(cfg['routes']):
        P = g.points[route]
        ax.plot(P[:, a0], P[:, a1], color=CL[c], lw=lw, zorder=3, solid_capstyle='round')
    for c, s in enumerate(cfg['release_stations']):
        p = g.points[s]
        ax.plot(p[a0], p[a1], '^', ms=5.2, mfc=CL[c], mec='white', mew=.5, zorder=6)
    T = g.points[sb.targets]
    ax.plot(T[:, a0], T[:, a1], 'o', ms=4.2, mfc=TARGET, mec='white', mew=.4, zorder=7)
    r = g.points[sb.root_station]
    ax.plot(r[a0], r[a1], 's', ms=4, mfc=INK, mec='white', mew=.4, zorder=7)


def fig2(out, runs, val_runs, tasks):
    arms = load_runs(runs)
    varms = load_runs(val_runs) if val_runs and Path(val_runs).exists() else {}
    tdoc = {t['base_task_hash']: t for t in json.loads(Path(tasks).read_text())['tasks']}
    fig = plt.figure(figsize=(W2, 205 * MM))
    notes = []
    # ---------------------------------------------------------------- a: weighted graph
    hcc = [h for h, t in tdoc.items() if t['task']['anatomy'] == 'hcc_portal_multifocal' and h in arms.get('proposed', {})]
    hcc.sort(key=lambda h: (tdoc[h]['task']['n_targets'], tdoc[h]['task']['pattern']))
    focus = next((h for h in hcc if tdoc[h]['task']['n_targets'] == 5), hcc[-1])
    sb, g = scene_for(tdoc[focus]['task'])
    a0, a1 = plane(g)
    flabel(fig, 'a', 0, .997); ftitle(fig, 'Weighted vessel graph (HCC portal tree)', .025, .997)
    ax = fig.add_axes([0, .745, .34, .23])
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap('Blues'); norm = Normalize(0, float(np.percentile(g.flow, 99)))
    draw_tree(ax, g, a0, a1, values=g.flow, cmap=lambda x: cmap(.25 + .75 * x), norm=norm, lw=2.6)
    bif = np.flatnonzero(g.degree >= 3)
    ax.plot(g.points[bif, a0], g.points[bif, a1], 'o', ms=2.2, color=INK, zorder=5)
    rel = g.release_candidates(sb.root_station)
    ax.plot(g.points[rel, a0], g.points[rel, a1], '^', ms=5, mfc='none', mec=INK, mew=.7, zorder=6)
    T = g.points[sb.targets]
    ax.plot(T[:, a0], T[:, a1], 'o', ms=4.2, mfc=TARGET, mec='white', mew=.4, zorder=7)
    ax.legend(handles=[Line2D([], [], marker='o', ls='none', color=INK, ms=2.5, label='bifurcation'),
                       Line2D([], [], marker='^', ls='none', mfc='none', mec=INK, ms=5, label='release candidate'),
                       Line2D([], [], marker='o', ls='none', mfc=TARGET, mec='white', ms=4.5, label='target lesion')],
              loc='lower left', fontsize=5.4, handletextpad=.2, borderaxespad=0)
    cax = fig.add_axes([.30, .76, .008, .08])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Blues')), cax=cax)
    cb.set_label('flow (mm/s)', fontsize=5.2); cb.ax.tick_params(labelsize=5, width=.4, length=1.5); cb.outline.set_linewidth(.4)
    K = g.key_graph(sb.targets, rel)
    notes.append(f'a: {K.number_of_nodes()} key nodes, {K.number_of_edges()} segments, {len(rel)} release candidates, '
                 f'{g.n} stations')
    write_csv(out, 'fig2a_key_graph_edges', ['u', 'v', 'length_mm', 'time_down_s', 'time_up_s', 'min_radius_mm', 'mean_flow_mm_s'],
              [[u, v, d['length_mm'], d['time_down_s'], d['time_up_s'], d['min_radius_mm'], d['mean_flow_mm_s']]
               for u, v, d in K.edges(data=True)])
    # ---------------------------------------------------------------- b: N* and release per layout
    flabel(fig, 'b', .37, .997); ftitle(fig, 'Configurations chosen by the planner for different lesion layouts (proposed)', .395, .997)
    picks = []
    for m in (2, 3, 4, 5):
        cand = [h for h in hcc if tdoc[h]['task']['n_targets'] == m]
        if cand:
            picks.append(cand[0] if m % 2 else cand[-1])
    rows_b = []
    for j, h in enumerate(picks[:4]):
        r = arms['proposed'][h]
        sbj, gj = (sb, g) if h == focus else scene_for(tdoc[h]['task'])
        axb = fig.add_axes([.37 + j * .158, .745, .15, .195])
        draw_config(axb, gj, sbj, r, a0, a1)
        e = r['executed']
        axb.set_title(f"m = {tdoc[h]['task']['n_targets']} ({tdoc[h]['pattern_realised'].split('_')[0]})  →  N* = {r['configuration']['n']}\n"
                      f"last arrival {e['T_last_deadline_scored_s']:.0f} s" + ('' if e['success'] else ' (failed)'),
                      fontsize=5.4, color=INK2, pad=1)
        rows_b.append([h, tdoc[h]['task']['n_targets'], tdoc[h]['pattern_realised'], r['configuration']['n'],
                       r['configuration']['release_stations'], r['configuration']['tours'], e['success'], e['T_last_deadline_scored_s']])
        if h != focus:
            sbj.close()
    write_csv(out, 'fig2b_configurations', ['base_task_hash', 'm', 'pattern', 'N_star', 'release_stations', 'tours', 'success', 'T_last_s'], rows_b)
    fig.legend(handles=[Line2D([], [], marker='^', ls='none', mfc=CL[0], mec='white', ms=5, label='release point'),
                        Line2D([], [], color=CL[0], lw=1.4, label='route (cluster colour)'),
                        Line2D([], [], marker='s', ls='none', mfc=INK, mec='white', ms=4, label='catheter entry')],
               loc='upper left', bbox_to_anchor=(.37, .752), ncol=3, fontsize=5.4, handletextpad=.3, columnspacing=1.)
    # ---------------------------------------------------------------- c: timeline of one configuration
    flabel(fig, 'c', 0, .70); ftitle(fig, 'Release, travel, coordination and arrival on one clock (focus task of a; executed)', .025, .70)
    from marl.v3.execute import run_configuration
    r = arms['proposed'][focus]
    summ, trace = run_configuration(sb, config_from(r), record_trace=True)
    axc = fig.add_axes([.06, .45, .42, .22])
    n = r['configuration']['n']
    t = np.asarray([s['t'] for s in trace]); park = np.asarray([s['parked'] for s in trace]); act = np.asarray([s['active'] for s in trace])
    hold = np.asarray([s['hold'] for s in trace]); shield = np.asarray([s['shield'] for s in trace])
    tdt = float(np.median(np.diff(t))) if len(t) > 1 else .1
    owner = {tt: c for c, tour in enumerate(r['configuration']['tours']) for tt in tour}
    arr = summ['arrival_times_s']; pred = r['configuration']['predicted']['arrivals_pred']
    pred_by_target = {}
    k = 0
    for c, tour in enumerate(r['configuration']['tours']):
        for tt in tour:
            pred_by_target[tt] = pred[k]; k += 1
    rows_c = []
    for c in range(n):
        y = n - c
        def spans(mask):
            out, start = [], None
            for i, v in enumerate(mask):
                if v and start is None: start = i
                if not v and start is not None: out.append((t[start] - tdt, t[i - 1])); start = None
            if start is not None: out.append((t[start] - tdt, t[-1]))
            return out
        for s0, s1 in spans(park[:, c]):
            axc.barh(y, s1 - s0, left=s0, height=.5, color='#eceef2', lw=0)
        mine = [tt for tt, cc in owner.items() if cc == c]
        last = max([arr[tt] for tt in mine if arr[tt] is not None], default=None)
        done = (t >= last) if (last is not None and all(arr[tt] is not None for tt in mine)) else np.zeros(len(t), bool)
        for s0, s1 in spans(act[:, c] & ~hold[:, c] & ~done):
            axc.barh(y, s1 - s0, left=s0, height=.5, color=CL[c], lw=0)
        for s0, s1 in spans(act[:, c] & done):
            axc.barh(y, s1 - s0, left=s0, height=.5, color=CL[c], alpha=.25, lw=0)
        for s0, s1 in spans(act[:, c] & hold[:, c]):
            axc.barh(y, s1 - s0, left=s0, height=.5, color='#6b7280', lw=0)
        sh = np.flatnonzero(shield[:, c] & act[:, c])
        if len(sh):
            axc.plot(t[sh], np.full(len(sh), y + .34), '|', ms=2.2, color='#c25e00', mew=.5)
        for tt, cc in owner.items():
            if cc != c:
                continue
            if arr[tt] is not None:
                axc.plot(arr[tt], y, '*', ms=7, mfc=TARGET, mec=INK, mew=.5, zorder=5)
            if pred_by_target.get(tt) is not None:
                axc.plot(pred_by_target[tt], y - .38, 'v', ms=3.2, mfc='none', mec=INK, mew=.5)
            rows_c.append([c, tt, pred_by_target.get(tt), arr[tt]])
    rel_t = summ['release_times_s']
    axc.set_yticks([n - c for c in range(n)]); axc.set_yticklabels([f'cluster {c+1}' for c in range(n)])
    axc.set_ylim(.35, n + .65)
    axc.set_xlim(0, max(t[-1], 1) * 1.03); axc.set_xlabel('Time from catheter insertion (s)'); clean(axc)
    axc.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#eceef2', label='in catheter'),
                        plt.Rectangle((0, 0), 1, 1, color=CL[0], label='moving'),
                        plt.Rectangle((0, 0), 1, 1, color='#6b7280', label='TPG hold'),
                        plt.Rectangle((0, 0), 1, 1, color=CL[0], alpha=.25, label='delivered'),
                        Line2D([], [], marker='|', ls='none', color='#c25e00', ms=4, label='shield rewrite'),
                        Line2D([], [], marker='*', ls='none', mfc=TARGET, mec=INK, ms=7, label='arrival'),
                        Line2D([], [], marker='v', ls='none', mfc='none', mec=INK, ms=4, label='planned arrival')],
               loc='upper center', bbox_to_anchor=(.5, -.2), ncol=7, fontsize=5.2, handlelength=1.1, columnspacing=.7)
    write_csv(out, 'fig2c_arrivals', ['cluster', 'target', 'planned_arrival_s', 'executed_arrival_s'], rows_c)
    notes.append(f"c: releases at {', '.join(f'{x:.1f}' for x in rel_t if x is not None)} s; executed success {summ['success']}, "
                 f"last arrival {summ['T_last_deadline_scored_s']:.1f} s, TPG hold {summ['tpg_hold_robot_s']:.1f} robot-s, "
                 f"shield rewrites {summ['shield_intervention_steps']} cluster-steps")
    sb.close()
    # ---------------------------------------------------------------- d: optimisation relation
    flabel(fig, 'd', .53, .70); ftitle(fig, 'Choosing the number of clusters', .555, .70)
    ex = arms.get('exact', {})
    axd = fig.add_axes([.575, .45, .115, .22])
    ratios = defaultdict(list); rows_d = []
    for h, r in ex.items():
        J = {int(k): v for k, v in r['configuration']['search']['J_by_n'].items()}
        best = min(J.values())
        ks = sorted(J)
        axd.plot(ks, [J[k] / best for k in ks], color='#c3c8d0', lw=.4, alpha=.7)
        for k in ks:
            ratios[k].append(J[k] / best); rows_d.append([h, k, J[k], J[k] / best, r['configuration']['n']])
    if ratios:
        ks = sorted(ratios)
        axd.plot(ks, [np.median(ratios[k]) for k in ks], color=PROPOSED, lw=1.4, marker='o', ms=3, mec='white')
        axd.axhline(1.1, color=INK3, lw=.5, ls=(0, (2, 1.5)))
        axd.text(2.6, 1.16, 'ε = 10% band', fontsize=5, color=INK3, va='bottom')
    axd.set_xticks([1, 2, 3, 4]); axd.set_xlabel('N'); axd.set_ylabel('Predicted J(N) / min J'); axd.set_yscale('log')
    axd.set_yticks([1, 1.5, 2, 3]); axd.set_yticklabels(['1', '1.5', '2', '3']); clean(axd, 'y')
    write_csv(out, 'fig2d_J_by_N_exact', ['base_task_hash', 'N', 'J', 'J_over_min', 'N_star'], rows_d)
    axd2 = fig.add_axes([.725, .45, .095, .22])
    prop = arms.get('proposed', {})
    counts = np.zeros((4, 4))
    for h, r in prop.items():
        counts[tdoc[h]['task']['n_targets'] - 2, r['configuration']['n'] - 1] += 1
    bottom = np.zeros(4)
    for k in range(4):
        axd2.bar(np.arange(2, 6), counts[:, k], bottom=bottom, color=CL[k], width=.7, lw=0, label=f'N* = {k+1}')
        bottom += counts[:, k]
    axd2.set_xticks([2, 3, 4, 5]); axd2.set_xlabel('Targets m'); axd2.set_ylabel('Tasks'); clean(axd2, 'y')
    axd2.set_ylim(0, max(1, bottom.max()) * 1.45)
    axd2.legend(fontsize=5, loc='upper left', ncol=2, handlelength=.8, columnspacing=.6, borderaxespad=.1)
    write_csv(out, 'fig2d_N_star_by_m', ['m', 'N1', 'N2', 'N3', 'N4'], [[m + 2, *counts[m]] for m in range(4)])
    axd3 = fig.add_axes([.885, .45, .105, .22])
    rows_d3 = []
    for dep, mk, col in (('root', 'v', BASE), ('opt', 'o', PROPOSED)):
        xs, ys, lo, hi = [], [], [], []
        for k in range(1, 5):
            arm = f'fixedN{k}_{dep}'
            if arm not in varms:
                continue
            gr = defaultdict(list)
            for r in varms[arm].values():
                gr[r['task']['anatomy']].append(val(r, 'success'))
            m_, l_, h_ = two_level(gr)
            xs.append(k + (-.08 if dep == 'root' else .08)); ys.append(100 * m_); lo.append(100 * l_); hi.append(100 * h_)
            rows_d3.append([dep, k, m_, l_, h_])
        if xs:
            axd3.vlines(xs, lo, hi, color=col, lw=.7); axd3.plot(xs, ys, mk, ms=3.6, mfc=col, mec='white', mew=.3,
                                                                 label='root release' if dep == 'root' else 'optimised release')
    if 'proposed' in varms:
        gr = defaultdict(list)
        for r in varms['proposed'].values():
            gr[r['task']['anatomy']].append(val(r, 'success'))
        m_, l_, h_ = two_level(gr)
        axd3.axhspan(100 * l_, 100 * h_, color=PROPOSED, alpha=.12, lw=0); axd3.axhline(100 * m_, color=PROPOSED, lw=.8)
        rows_d3.append(['proposed_adaptive', None, m_, l_, h_])
    axd3.set_xticks([1, 2, 3, 4]); axd3.set_xlabel('Fixed N (validation)'); axd3.set_ylabel('Navigation success (%)')
    axd3.legend(fontsize=5, loc='lower center', handletextpad=.2); clean(axd3, 'y')
    write_csv(out, 'fig2d_validation_fixed_N', ['deployment', 'N', 'success', 'lo', 'hi'], rows_d3)
    # ---------------------------------------------------------------- e: arms x four metrics
    flabel(fig, 'e', 0, .375); ftitle(fig, 'Fixed versus autonomous configuration on the frozen test set (same local controller)', .025, .375)
    present = [(a, l) for a, l in ARMS if a in arms]
    rows_e, summary_e = [], {}
    for j, (key, lab, sc) in enumerate(METRICS):
        ax = fig.add_axes([.17 + j * .21, .215, .18, .135])
        for i, (arm, al) in enumerate(present):
            gr = defaultdict(list)
            for r in arms[arm].values():
                gr[r['task']['anatomy']].append(val(r, key))
            m_, l_, h_ = two_level(gr)
            y = len(present) - 1 - i
            col = PROPOSED if arm == 'proposed' else (EXACT if arm == 'exact' else BASE)
            ax.hlines(y, sc * l_, sc * h_, color=col, lw=.8)
            ax.plot(sc * m_, y, 'o', ms=3.8, mfc=col if arm != 'exact' else 'white', mec=col, mew=.7, zorder=3)
            rows_e.append([arm, key, m_, l_, h_, len(arms[arm])]); summary_e[(arm, key)] = m_
        ax.set_yticks(range(len(present))); ax.set_yticklabels([l for _, l in present][::-1] if j == 0 else [])
        ax.set_xlabel(lab); clean(ax)
    write_csv(out, 'fig2e_arms_four_metrics', ['arm', 'metric', 'mean', 'lo', 'hi', 'tasks'], rows_e)
    # ---------------------------------------------------------------- f: ablations vs proposed
    flabel(fig, 'f', 0, .165); ftitle(fig, 'Ablations: paired difference to the proposed configuration (ablation − proposed)', .025, .165)
    present_ab = [(a, l) for a, l in ABLATIONS if a in arms and 'proposed' in arms]
    rows_f = []
    for j, (key, lab, sc) in enumerate(METRICS):
        ax = fig.add_axes([.17 + j * .21, .02, .18, .115])
        for i, (arm, al) in enumerate(present_ab):
            gr = defaultdict(list)
            for h in set(arms[arm]) & set(arms['proposed']):
                gr[arms[arm][h]['task']['anatomy']].append(val(arms[arm][h], key) - val(arms['proposed'][h], key))
            m_, l_, h_ = two_level(gr)
            y = len(present_ab) - 1 - i
            ax.hlines(y, sc * l_, sc * h_, color=INK2, lw=.8); ax.plot(sc * m_, y, 'o', ms=3.6, color=INK2, mec='white', mew=.3)
            rows_f.append([arm, key, m_, l_, h_])
        ax.axvline(0, color=PROPOSED, lw=.7)
        ax.set_yticks(range(len(present_ab))); ax.set_yticklabels([l for _, l in present_ab][::-1] if j == 0 else [])
        ax.set_xlabel('Δ ' + lab.replace(' (%)', ' (pp)')); clean(ax)
    write_csv(out, 'fig2f_ablation_paired', ['arm', 'metric', 'diff_mean', 'lo', 'hi'], rows_f)
    n_tasks = len(arms.get('proposed', {}))
    caption = f"""
Fig. 2 | Graph-based autonomous swarm configuration and planning. a, Weighted vessel graph of the portal-hepatic
(HCC) territory built from preoperative information only: station edges coloured by the nominal flow speed used
for planning (healthy tree, 0.05 mm/s inlet), bifurcations, catheter-accessible release candidates and five target
lesions. b, Configurations selected by the planner for HCC test tasks with 2-5 lesions: number of clusters N*,
release points on the catheter path, and routes (one colour per cluster). c, Executed timeline of the configuration
in a: time in the catheter, motion, holds imposed by the release-aware temporal plan graph, steps where the spacing
shield rewrote the command, executed arrivals and planned arrivals. d, Left: predicted objective J(N) relative to the
best N for every test task (exact search); the ε = 10% rule selects the smallest N within the band. Middle: N*
chosen by the proposed planner by number of lesions. Right: executed navigation success of fixed N on the validation
set with root and optimised release, against the adaptive planner (band). e, Four core metrics on the frozen test set
({n_tasks} base tasks, 15 territories): navigation success, contact-free safe success, deadline-scored last target
arrival measured from catheter insertion, and the fraction of tasks with an inter-cluster spacing violation
(d < 2 mm). Every arm uses the same pure-pursuit local controller, safety layer, flow, obstacles, patient-level
variation and deadline; fixed N = 3 was chosen on the validation set. Markers, mean; bars, 95% two-level bootstrap
interval (territories, then tasks). f, Paired differences of each ablation to the proposed configuration on the same
tasks. Notes: {'; '.join(notes)}. d_safe = 2 mm and all deployment parameters are simulation assumptions
(configs/v3/assumptions.json); no magnetic interaction model is used.
"""
    save(fig, out, 'fig2_configuration', caption)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'research/figures/NMI_v3')
    ap.add_argument('--only', default='fig2')
    ap.add_argument('--runs', type=Path, default=RES / 'test_runs.jsonl')
    ap.add_argument('--val-runs', type=Path, default=RES / 'validation_runs.jsonl')
    ap.add_argument('--tasks', type=Path, default=RES / 'tasks_test.json')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if 'fig2' in a.only:
        fig2(a.out, a.runs, a.val_runs, a.tasks)
        print('wrote fig2')


if __name__ == '__main__':
    main()
