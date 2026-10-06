"""Figures for the manuscript draft, all rendered from the simulator and the measured result files.

usage: make_manuscript_figures.py [--out research/figures/MS_20261006] [--only fig2]
fig1 scene: anatomy centreline, clots, clusters, particles, with the planned tours
fig2 perception chain: the actual rendered camera images with detections, and the error distribution
fig3 TPG: pairwise conflict zones on the two tour arclengths, and the resulting hold schedule
fig4 results: Safe Success by N and method (information level annotated)
fig5 robustness: imaging-quality sweep
fig6 trajectories: ours (deployable) vs the privileged planner in one scene
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, '.')
CJK = next((f for f in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'AR PL UMing CN', 'Droid Sans Fallback')
             if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}), 'DejaVu Sans')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [CJK, 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 8, 'axes.linewidth': .6, 'xtick.major.width': .6,
                     'ytick.major.width': .6, 'savefig.dpi': 300, 'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False})
C = dict(vessel='#c9d6e3', clot='#b5293a', cluster='#15607a', particle='#e0a33c',
         ours='#15607a', priv='#9aa7b4', rl='#b5293a', tpg='#2f8f6b')
ANAT, SEED = 'mca_m1_lvo', 2600000000
AN6 = 'ica_terminus_t'          # fig6 scene (see fig6 docstring)


def env_of(n=3, anatomy=ANAT, seed=SEED, horizon=300.):
    from environments.mca_physical_env import DynamicsConfig
    from marl.teacher import TEACHER_CONFIG
    from scripts.multicluster_protocol import paired_environment
    cfg = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy,
                  episode_duration_s=horizon, junction_model='union')
    return paired_environment(cfg, n, seed)[0]


def draw_vessels(ax, env, ax0=0, ax1=1, lw=4.5, alpha=.85, zorder=0):
    t = env.transport; pts = t.points.astype(float)
    rad = np.asarray(env.flow_model.healthy_radius_mm, float)
    for (a, b) in t.ends:
        r = .5*(rad[a]+rad[b])
        ax.plot(pts[[a, b], ax0], pts[[a, b], ax1], color=C['vessel'], solid_capstyle='round',
                lw=lw*r, alpha=alpha, zorder=zorder)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig1(out):
    import scripts.benchmark_multicluster as bm
    env = env_of(3)
    plan, info = bm.preoperative_plan(env)
    _, sp = bm._station_paths(env)
    pts = env.transport.points.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (a0, a1, name) in zip(axes, [(0, 1, 'x–y (俯视)'), (0, 2, 'x–z (侧视)')]):
        draw_vessels(ax, env, a0, a1)
        P = env.positions_mm
        n = env.num_robots
        ax.scatter(P[n:, a0], P[n:, a1], s=4, color=C['particle'], zorder=2, label='碎屑粒子')
        for i, seq in enumerate(plan):
            cur = int(np.asarray(env.robot_stations)[i])
            for c in seq:
                path = sp(cur, int(np.asarray(env.clot_stations)[c]))[::-1]
                ax.plot(pts[path, a0], pts[path, a1], color=C['cluster'], lw=1.1, ls='--', alpha=.75, zorder=3)
                cur = int(np.asarray(env.clot_stations)[c])
        ax.scatter(env.clot_positions_mm[:, a0], env.clot_positions_mm[:, a1], s=55, marker='X',
                   color=C['clot'], edgecolor='w', lw=.6, zorder=5, label='血栓')
        ax.scatter(P[:n, a0], P[:n, a1], s=42, color=C['cluster'], edgecolor='w', lw=.6, zorder=6, label='磁性集群')
        ax.set_title(name, fontsize=8.5)
    h = [Line2D([], [], marker='X', ls='', color=C['clot'], ms=7, label='血栓'),
         Line2D([], [], marker='o', ls='', color=C['cluster'], ms=6, label='磁性集群'),
         Line2D([], [], marker='o', ls='', color=C['particle'], ms=3, label='碎屑粒子'),
         Line2D([], [], ls='--', color=C['cluster'], lw=1.1, label='术前分配的行程')]
    axes[0].legend(handles=h, loc='upper left', frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out/'fig1_scene.png', bbox_inches='tight'); plt.close(fig)
    env.close()
    return 'fig1_scene.png'


def fig2(out):
    """The actual camera images the controller sees, with the detector's output on top."""
    import scripts.benchmark_multicluster as bm
    from marl.deployable_sensing import DeployablePursuit
    from marl.image_sensing import ImageSensor
    env = env_of(1)                                    # the pursuit_dep configuration, exactly as benchmarked
    sensor = ImageSensor(env, seed=SEED)
    plan, _ = bm.preoperative_plan(env)
    tg = bm.PlanTargets(plan)
    ctl = DeployablePursuit(env, sensor)
    best = None
    for step in range(1200):
        est = sensor.observe()
        loc = ctl.act(tg.targets(env, est.pos), est)
        k = len(est.particles[0])                      # keep the richest frame around the cluster
        if best is None or k > best[0]:
            best = (k, sensor.track[0].copy(), sensor._prev_truth.copy(), env.positions_mm.astype(float).copy())
        _, _, term, trunc, _ = env.step(ctl.to_world(loc, est))
        if term or trunc:
            break
    err = np.concatenate(sensor.err_log)
    _, centre, prev, now = best
    HALF = .55                                         # displayed crop half-width (mm)
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=.3)
    for j, (axes_, rax, name) in enumerate([((0, 1), [0, 1], '俯视相机 x–y'), ((0, 2), [0, 2], '侧视相机 x–z')]):
        img, origin = sensor._render(centre[list(axes_)], rax, prev, now)
        ax = fig.add_subplot(gs[0, j]); px = sensor.cam.pixel_mm
        # img[u, v]: rows = first world axis (u), cols = second (v) -> rows on the y axis, cols on x
        ext = [origin[1], origin[1]+img.shape[1]*px, origin[0]+img.shape[0]*px, origin[0]]
        ax.imshow(img, cmap='gray', extent=ext, origin='upper', vmin=sensor.cam.background-.05, vmax=.55)
        for q, a, fl in sensor._detect(img, origin):
            big = sensor._is_cluster(a, fl)
            ax.add_patch(plt.Circle((q[1], q[0]), .085 if big else .045, fill=False, lw=.9,
                                    color=C['cluster'] if big else C['particle']))
        c = centre[list(axes_)]
        ax.set_xlim(c[1]-HALF, c[1]+HALF); ax.set_ylim(c[0]+HALF, c[0]-HALF)
        ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        x0, y0 = c[1]-HALF+.08, c[0]+HALF-.08
        ax.plot([x0, x0+.2], [y0, y0], color='w', lw=2.2)
        ax.text(x0+.1, y0-.04, '0.2 mm', color='w', ha='center', fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(err, bins=40, color=C['cluster'], alpha=.85)
    for v, lab, c, h in [(err.mean(), f'均值 {err.mean():.3f}', C['clot'], .95),
                         (float(np.percentile(err, 95)), f'p95 {np.percentile(err, 95):.3f}', C['tpg'], .8)]:
        ax.axvline(v, color=c, lw=1, ls='--')
        ax.text(v, ax.get_ylim()[1]*h, ' '+lab, fontsize=6.5, color=c, va='top')
    ax.set_xlabel('定位误差 (mm)'); ax.set_ylabel('帧数'); ax.set_title('感知误差分布', fontsize=8)
    fig.savefig(out/'fig2_perception.png', bbox_inches='tight'); plt.close(fig)
    env.close()
    return 'fig2_perception.png'


def fig3(out):
    """TPG: where two tours conflict in 3-D, and the hold the schedule produces."""
    import scripts.benchmark_multicluster as bm
    from marl.deployable_sensing import DeployablePursuit, DeployableSensor
    from marl.tpg_coordinator import TPGCoordinator
    seed = SEED+100000                                 # a scene whose planned tours do conflict
    env = env_of(3, seed=seed)
    plan, _ = bm.preoperative_plan(env)
    _, sp = bm._station_paths(env)
    co = TPGCoordinator(env, plan, sp, d_min_mm=2.)
    pair = max({(z['i'], z['j']) for z in co.zones},
               key=lambda ij: sum((z['s1']-z['s0'])*(z['u1']-z['u0']) for z in co.zones if (z['i'], z['j']) == ij))
    i, j = pair
    A, B = co.paths[i], co.paths[j]
    D = np.linalg.norm(A[:, None, :]-B[None, :, :], axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    im = ax.imshow(D.T, origin='lower', cmap='viridis_r', aspect='auto',
                   extent=[0, len(A)*co.step, 0, len(B)*co.step], vmin=0, vmax=12)
    ax.contour(np.linspace(0, len(A)*co.step, len(A)), np.linspace(0, len(B)*co.step, len(B)),
               D.T, levels=[co.D], colors=[C['clot']], linewidths=1.1)
    for z in co.zones:
        if (z['i'], z['j']) != (i, j):
            continue
        ax.add_patch(plt.Rectangle((z['s0']*co.step, z['u0']*co.step), (z['s1']-z['s0'])*co.step,
                                   (z['u1']-z['u0'])*co.step, fill=False, ec='w', lw=.9, ls='--'))
    ax.set_xlabel(f'集群 {i} 行程弧长 (mm)'); ax.set_ylabel(f'集群 {j} 行程弧长 (mm)')
    ax.set_title(f'冲突区（红线 = d_min+缓冲 {co.D:.1f} mm）', fontsize=8)
    cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label('两集群距离 (mm)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    # online execution: progress and holds measured from an episode
    sensor = DeployableSensor(env, seed=SEED)
    ctl = DeployablePursuit(env, sensor); tg = bm.PlanTargets(plan)
    bm.FALLBACK['mode'] = 'park'
    coord = TPGCoordinator(env, plan, sp, d_min_mm=2.)
    prog, holds, dist, t = [], [], [], []
    for step in range(1500):
        est = sensor.observe()
        loc = ctl.act(tg.targets(env, est.pos), est)
        hold = coord.gate(est.pos, est.active)
        loc[hold] = 0.
        prog.append(coord.progress.copy()*coord.step); holds.append(hold.copy())
        P = env.positions_mm[:3]
        dist.append(min(np.linalg.norm(P[a]-P[b]) for a in range(3) for b in range(a+1, 3)))
        t.append(float(env.elapsed_s))
        _, _, term, trunc, _ = env.step(ctl.to_world(loc, est))
        if term or trunc:
            break
    ax = axes[1]
    ax.plot(t, dist, color=C['cluster'], lw=1.2, label='最近集群间距')
    ax.axhline(2., color=C['clot'], lw=1, ls='--', label='d_min = 2 mm')
    H = np.array(holds)
    for k in range(H.shape[1]):
        on = np.flatnonzero(H[:, k])
        for a in np.split(on, np.flatnonzero(np.diff(on) > 1)+1) if len(on) else []:
            ax.axvspan(t[a[0]], t[a[-1]], color=C['tpg'], alpha=.18, lw=0)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('距离 (mm)'); ax.set_ylim(0, None)
    ax.set_title('在线执行：绿色为 TPG 令其等待的时段', fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.tight_layout(); fig.savefig(out/'fig3_tpg.png', bbox_inches='tight'); plt.close(fig)
    env.close()
    return 'fig3_tpg.png'


V2 = Path('research/validation/BENCHMARK_V2_UNION_20261005')


def _rows(pattern):
    out = []
    for f in sorted(V2.glob(pattern)):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return [r for r in out if 'error' not in r]


def _safe(rows):
    return 100*np.mean([r['cluster_safe_success'] for r in rows]) if rows else np.nan


def _boot(d, n=4000):
    d = np.asarray(d, float)
    bs = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(n)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fig4(out):
    """Main result: Safe Success by N and method, and the paired differences with CIs."""
    rl = [json.loads(l) for l in (V2/'rl_dep.jsonl').read_text().splitlines() if l.strip()]
    rl = [r for r in rl if 'error' not in r]
    series = [('本文：追踪 + TPG', C['ours'], '可部署'), ('本文：仅追踪', C['tpg'], '可部署'),
              ('传统规划（特权）', C['priv'], '特权'), ('规则路线跟随（特权）', '#cdd4db', '特权'),
              ('Codex 分层 RL / 规则', '#7a6b8f', '实测'), ('纯 RL（PPO，最佳种子）', C['rl'], '可部署')]
    vals = {}
    for n in (1, 2, 3):
        ours_tpg = _rows(f'pursuit_tpg_dep_N{n}_*.jsonl')
        vals[('本文：追踪 + TPG', n)] = _safe(ours_tpg) if ours_tpg else _safe(_rows(f'pursuit_dep_N{n}_*.jsonl'))
        vals[('本文：仅追踪', n)] = _safe(_rows(f'pursuit_dep_N{n}_*.jsonl'))
        vals[('传统规划（特权）', n)] = _safe(_rows(f'plan_route_priv_N{n}_*.jsonl'))
        vals[('规则路线跟随（特权）', n)] = _safe(_rows(f'route_follow_priv_N{n}_*.jsonl'))
        cx = _rows(f'codex_*_N{n}_*.jsonl')
        vals[('Codex 分层 RL / 规则', n)] = _safe(cx)
        best = max((np.mean([r['cluster_safe_success'] for r in rl if r['clusters'] == n and r['method'] == m])
                    for m in {r['method'] for r in rl}), default=np.nan)
        vals[('纯 RL（PPO，最佳种子）', n)] = 100*best
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]; w = .13
    for s, (name, col, info) in enumerate(series):
        x = np.arange(3)+(s-2.5)*w
        y = [vals.get((name, n), np.nan) for n in (1, 2, 3)]
        ax.bar(x, y, w, color=col, label=f'{name}', edgecolor='w', lw=.4)
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.text(xi, yi+1.2, f'{yi:.0f}', ha='center', fontsize=5.8, rotation=90)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(['N=1', 'N=2', 'N=3'])
    ax.set_ylabel('安全成功率 (%)'); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.28))
    # paired differences
    ax = axes[1]; labels, mids, los, his = [], [], [], []
    for n in (1, 2, 3):
        a = {(r['anatomy'], r['seed']): r for r in _rows(f'pursuit_tpg_dep_N{n}_*.jsonl') or _rows(f'pursuit_dep_N{n}_*.jsonl')}
        b = {(r['anatomy'], r['seed']): r for r in _rows(f'plan_route_priv_N{n}_*.jsonl')}
        k = sorted(set(a) & set(b))
        m, lo, hi = _boot([100*(a[x]['cluster_safe_success']-b[x]['cluster_safe_success']) for x in k])
        labels.append(f'N={n}  本文 − 特权规划'); mids.append(m); los.append(lo); his.append(hi)
    for n in (2, 3):
        a = {(r['anatomy'], r['seed']): r for r in _rows(f'pursuit_tpg_dep_N{n}_*.jsonl')}
        b = {(r['anatomy'], r['seed']): r for r in _rows(f'pursuit_dep_N{n}_*.jsonl')}
        k = sorted(set(a) & set(b))
        m, lo, hi = _boot([100*(a[x]['cluster_safe_success']-b[x]['cluster_safe_success']) for x in k])
        labels.append(f'N={n}  TPG 消融'); mids.append(m); los.append(lo); his.append(hi)
    y = np.arange(len(labels))
    ax.errorbar(mids, y, xerr=[np.array(mids)-np.array(los), np.array(his)-np.array(mids)],
                fmt='o', ms=4, lw=1.2, color=C['cluster'], capsize=2.5)
    ax.axvline(0, color='k', lw=.7, ls=':')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('安全成功率差值 (百分点, 95% CI)')
    fig.tight_layout(); fig.savefig(out/'fig4_results.png', bbox_inches='tight'); plt.close(fig)
    return 'fig4_results.png'


SWEEP = Path('research/validation/IMAGE_CAMERA_SWEEP_20261005/rows.jsonl')
PROBE = Path('research/validation/IMAGE_SENSING_PROBE_20261005/rows.jsonl')


def fig5(out):
    """Robustness: imaging quality, and image chain vs the Gaussian-noise position model."""
    R = [json.loads(l) for l in SWEEP.read_text().splitlines() if l.strip()]
    R = [r for r in R if 'error' not in r]
    P = [json.loads(l) for l in PROBE.read_text().splitlines() if l.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]; ax2 = ax.twinx(); ax2.spines['right'].set_visible(True)
    for n, mk, col in [(1, 'o', C['cluster']), (3, 's', C['tpg'])]:
        px, safe, div = [], [], []
        for cam in ('0.02', '0.04', '0.06'):
            X = [r for r in R if r['clusters'] == n and r['camera_setting'] == cam]
            px.append(float(cam)*1000); safe.append(_safe(X))
            div.append(100*np.mean([r['perception']['err_max_mm'] > 1. for r in X]))
        ax.plot(px, safe, mk+'-', color=col, lw=1.3, ms=4.5, label=f'N={n} 安全成功率')
        ax2.plot(px, div, mk+':', color=col, lw=1, ms=3.5, alpha=.65, label=f'N={n} 跟踪发散率')
        X = [r for r in R if r['clusters'] == n and r['camera_setting'] == 'random']
        ax.plot([20], [_safe(X)], mk, color=col, mfc='none', ms=8, mew=1.3)
    ax.set_xlabel('像素尺度 (µm/px)'); ax.set_ylabel('安全成功率 (%)'); ax.set_ylim(60, 90)
    ax2.set_ylabel('跟踪发散率 (%)'); ax2.set_ylim(0, 15)
    ax.set_xticks([20, 40, 60])
    ax.plot([], [], 'o', color='k', mfc='none', ms=8, label='逐回合随机相机（画在 20 处）')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, frameon=False, fontsize=6.2, loc='lower left')
    ax = axes[1]; labels, mids, los, his = [], [], [], []
    for n in (1, 3):
        a = {(r['anatomy'], r['seed']): r for r in P if r['clusters'] == n and r['sensing_model'] == 'image'}
        b = {(r['anatomy'], r['seed']): r for r in P if r['clusters'] == n and r['sensing_model'] == 'noise'}
        k = sorted(set(a) & set(b))
        for key, lab, sc in [('cluster_safe_success', '安全成功率 (pp)', 100.), ('particle_events', '粒子碰撞 (次/回合)', 1.)]:
            m, lo, hi = _boot([sc*(float(a[x][key])-float(b[x][key])) for x in k])
            labels.append(f'N={n}  {lab}'); mids.append(m); los.append(lo); his.append(hi)
    y = np.arange(len(labels))
    cols = [C['cluster'] if lo*hi <= 0 else C['clot'] for lo, hi in zip(los, his)]
    for yi, m, lo, hi, c in zip(y, mids, los, his, cols):
        ax.errorbar([m], [yi], xerr=[[m-lo], [hi-m]], fmt='o', ms=4, lw=1.2, color=c, capsize=2.5)
    ax.axvline(0, color='k', lw=.7, ls=':')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('图像链 − 带噪位置（95% CI；红=CI 不含 0）')
    fig.tight_layout(); fig.savefig(out/'fig5_robustness.png', bbox_inches='tight'); plt.close(fig)
    return 'fig5_robustness.png'


def _trace(fn, *a, **kw):
    """Run an episode and return (trajectories [T, n, 3], row). Positions are recorded by wrapping step."""
    import scripts.multicluster_protocol as mp
    hist = []
    orig = mp.paired_environment

    def wrapped(cfg, n, seed):
        env, man = orig(cfg, n, seed)
        step = env.step

        def rec(action):
            hist.append(env.positions_mm[:n].copy())
            return step(action)
        env.step = rec
        return env, man
    mp.paired_environment = wrapped
    try:
        import scripts.benchmark_deployable as bd
        import scripts.benchmark_multicluster as bm
        bd.paired_environment = wrapped; bm.paired_environment = wrapped
        row = fn(*a, **kw)
    finally:
        mp.paired_environment = orig
        import scripts.benchmark_deployable as bd
        import scripts.benchmark_multicluster as bm
        bd.paired_environment = orig; bm.paired_environment = orig
    return np.array(hist), row


def fig6(out):
    """Same scene, same allocation: our deployable controller vs the privileged planner."""
    import scripts.benchmark_deployable as bd
    import scripts.benchmark_multicluster as bm
    from marl.deployable_sensing import DeployableConfig
    bm.JUNCTION['model'] = 'union'
    # One representative discordant scene: of the 420 paired N=3 scenes, both succeed in 53 %, only ours in
    # 27 %, only the privileged planner in 10 %, neither in 10 %. This is a scene from the largest
    # discordant group, with the median path ratio inside it.
    seed, n = 2600700026, 3
    # exactly the configurations of the benchmarked rows, so the annotated numbers are the recorded ones
    ours, row_o = _trace(bd.run, 'pursuit_tpg_dep', n, AN6, seed, 300., 2., DeployableConfig(), 'union', 'noise')
    bm.FALLBACK['mode'] = 'help'            # bd.run leaves it on 'park' for the TPG variant
    priv, row_p = _trace(bm.run_episode, 'plan_route', n, AN6, seed, 300., 2.)
    env = env_of(n, AN6, seed)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    cols = [C['cluster'], C['tpg'], '#8a5fa8']
    for ax, (T, row, name) in zip(axes, [(ours, row_o, '本文：追踪 + TPG（可部署）'), (priv, row_p, '传统规划（特权真值）')]):
        draw_vessels(ax, env, 0, 1, lw=4.)
        for i in range(n):
            ax.plot(T[:, i, 0], T[:, i, 1], color=cols[i], lw=1.1, alpha=.9)
            ax.scatter(T[0, i, 0], T[0, i, 1], s=26, color=cols[i], edgecolor='w', lw=.5, zorder=5)
        ax.scatter(env.clot_positions_mm[:, 0], env.clot_positions_mm[:, 1], s=55, marker='X',
                   color=C['clot'], edgecolor='w', lw=.6, zorder=6)
        safe = '安全成功' if row['cluster_safe_success'] else '未达安全成功'
        t100 = f"{row['t100_s']:.0f} s" if row.get('t100_s') else '未清除'
        ax.set_title(f'{name}\n总路径 {row["path_mm"]:.0f} mm · 清除用时 {t100} · {safe}', fontsize=8)
    h = [Line2D([], [], color=cols[i], lw=1.4, label=f'集群 {i}') for i in range(n)]
    h.append(Line2D([], [], marker='X', ls='', color=C['clot'], ms=7, label='血栓'))
    axes[0].legend(handles=h, loc='lower left', frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out/'fig6_trajectories.png', bbox_inches='tight'); plt.close(fig)
    env.close()
    return 'fig6_trajectories.png'


FIGS = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6)



def fig7(out):
    """Framework and network diagram of the hierarchical pipeline with the IR-PPO residual controller."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from marl.drl_local import VEC_DIM
    fig = plt.figure(figsize=(7.4, 4.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec='#334155', fs=6.8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=ec, lw=.7))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal',
                linespacing=1.25)

    def arrow(x0, y0, x1, y1, c='#334155', ls='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=.8, color=c, ls=ls))
    # top band: hierarchy
    ax.text(1, 60, 'a  分层框架（全部为可部署信息）', fontsize=8, weight='bold')
    box(1, 49, 15, 8, '术前 CTA 地图\n中心线 · 健康半径\n血栓位置', '#e8eef5')
    box(20, 49, 15, 8, '分配 A\n测地完工时间最优\n（术前一次）', '#dbe7f3')
    box(39, 49, 15, 8, 'TPG 时序协调\n冲突区 + 优先级\n区前等待（硬约束）', '#d4ece1')
    box(58, 49, 18, 8, '纯追踪规则 u_rule\n+ IR-PPO 残差 Δu\n（本文 DRL）', '#fde2e1', bold=True)
    box(80, 49, 18, 8, '间距盾 + 死区\n→ 磁场指令\n（10 Hz）', '#eee8f5')
    for x in (16, 35, 54, 76):
        arrow(x+.3, 53, x+3.7, 53)
    box(39, 39.5, 37, 6.5, '双视角相机（俯视 x–y / 侧视 x–z）\n检测 · 三维融合 · 联合跟踪 → 位置、粒子、同伴',
        '#f3f4f6', fs=6.3)
    arrow(57, 45.5, 64, 48.7); arrow(50, 45.5, 46, 48.7)
    # bottom: network
    ax.text(1, 35.5, 'b  IR-PPO 网络（参数共享，每个集群独立执行）', fontsize=8, weight='bold')
    box(1, 22, 13, 10, '图像输入\n2×32×32\n俯视/侧视裁剪\n1.6 mm 视野', '#f1f5f9', fs=6.3)
    box(17, 22, 15, 10, 'CNN 编码器\nConv5×5/2 16\nConv3×3/2 32\nConv3×3/2 32\nFC 128 · LN', '#dbeafe', fs=6.1)
    box(1, 7, 13, 11, f'向量输入 {VEC_DIM} 维\n路线方向 · 规则指令\n速度 · 4 粒子 · 2 同伴\n地图间隙 · 分叉 · TPG', '#f1f5f9', fs=6.0)
    box(17, 9, 15, 7, 'MLP 编码器\nFC 128 · LN · GELU', '#dbeafe', fs=6.3)
    box(36, 13, 11, 13, '拼接\n256', '#e0e7ff')
    box(51, 20, 16, 9, 'Actor 主干\nFC256·LN·GELU\nFC256·GELU', '#fde2e1', fs=6.2)
    box(51, 7, 16, 9, 'Critic 主干\nFC256·LN·GELU\nFC256·GELU', '#e2f0e8', fs=6.2)
    box(71, 20, 12, 9, 'μ(Δu) ∈ R³\n零初始化\nσ 可学习', '#fde2e1', fs=6.2)
    box(71, 7, 12, 9, 'V(s)\nGAE λ=0.95\nγ=0.995', '#e2f0e8', fs=6.2)
    box(86, 17, 13, 15, 'u = clip(u_rule\n + 0.6·Δu)\n\n初始化时\nu ≡ u_rule', '#fff7d6', fs=6.2, bold=False)
    arrow(14, 27, 17, 27); arrow(14, 12.5, 17, 12.5); arrow(32, 27, 36, 22); arrow(32, 12.5, 36, 17)
    arrow(47, 21, 51, 24.5); arrow(47, 17, 51, 11.5); arrow(67, 24.5, 71, 24.5); arrow(67, 11.5, 71, 11.5)
    arrow(83, 24.5, 86, 24.5)
    ax.text(50, 2.2, 'PPO（clip 0.2，4 轮，批 2048）；奖励：清除增量 −壁接触 −粒子碰撞/接触 −间距违规 −|Δu|²；'
            '终局 +20 安全完成 / −10 集群丢失', fontsize=6.2, ha='center', color='#475569')
    fig.savefig(out/'fig7_framework.png', bbox_inches='tight', dpi=300); plt.close(fig)
    return 'fig7_framework.png'


FIGS['fig7'] = fig7


def fig9(out):
    """T-IRPPO: hierarchical pipeline + temporal Transformer residual policy (benchmark v3)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from marl.obstacle_control import K_OBS, TOKEN_DIM, WINDOW
    fig = plt.figure(figsize=(7.4, 5.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis('off')

    def box(x, y, w, h, text, fc, fs=6.6, bold=False, ec='#334155'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=ec, lw=.7))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.25)

    def arrow(x0, y0, x1, y1, c='#334155'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=.8, color=c))
    ax.text(1, 66, 'a  分层框架：全局靠术前地图，局部未知障碍靠时序 DRL', fontsize=8, weight='bold')
    box(1, 54, 15, 9, '术前 CTA 地图\n中心线 · 健康半径\n血栓位置', '#e8eef5')
    box(20, 54, 15, 9, '分配 A\n测地完工时间最优', '#dbe7f3')
    box(39, 54, 15, 9, 'TPG 时序协调\n集群间安全距离\n（硬约束）', '#d4ece1')
    box(58, 54, 19, 9, '路线追踪 + APF  u_rule\n⊕ T-IRPPO 残差 Δu\n（本文 DRL）', '#fde2e1', bold=True)
    box(81, 54, 17, 9, '间距盾 · 死区\n→ 磁场指令 10 Hz\n（增益/噪声随机化）', '#eee8f5')
    for x in (16, 35, 54, 77):
        arrow(x+.3, 58.5, x+3.7, 58.5)
    box(30, 43, 56, 7.5, '双视角数字显微镜 → 集群跟踪（位置/速度）+ 障碍检测框（静态斑块 / 漂移碎片）\n延迟 1–2 帧 · 位置噪声 · 丢帧 · 2.5 % 尺寸噪声（多层域随机化）', '#f3f4f6', fs=6.1)
    arrow(60, 50.5, 66, 53.7); arrow(50, 50.5, 46, 53.7)
    ax.text(1, 39, f'b  T-IRPPO 网络（参数共享，集群分散执行；记忆窗口 {WINDOW} 步 = 1.6 s）', fontsize=8, weight='bold')
    for k in range(4):
        box(1+k*1.2, 22-k*1.6, 15, 12, '', '#f1f5f9')
    box(5.0, 17.2, 15, 12, f't 时刻 token（{TOKEN_DIM} 维）\n路线方向 · APF 指令\n速度 · 上一动作 · 目标距离\n管腔间隙 · 分叉 · TPG\n{K_OBS} 个障碍（相对位置/尺寸/\n相对速度/表面间隙）· 2 同伴', '#f1f5f9', fs=5.7)
    ax.text(9, 31.5, 't−15 … t', fontsize=6.2, color='#475569')
    box(24, 18, 13, 10, '线性嵌入 128\nLN · GELU\n+ 时间位置编码\n（起始前用 pad token）', '#dbeafe', fs=6.0)
    box(41, 15, 17, 16, 'Transformer 编码器 ×2\n因果自注意力（时间维）\n4 头 · d=128 · FFN 256\nPre-LN', '#e0e7ff', fs=6.3, bold=True)
    box(62, 24, 14, 8.5, 'Actor 头\nMLP 256-256\nμ(Δu) 零初始化', '#fde2e1', fs=6.1)
    box(62, 12, 14, 8.5, 'Critic 头\nMLP 256-256\nV(s)', '#e2f0e8', fs=6.1)
    box(80, 18, 18, 14, 'u = clip(u_rule + Δu)\n\n初始化时 u ≡ u_rule\n（从经典方法出发，\nPPO 只学修正）', '#fff7d6', fs=6.2)
    arrow(20.2, 23, 24, 23); arrow(37, 23, 41, 23); arrow(58, 25, 62, 28); arrow(58, 21, 62, 16.5); arrow(76, 28, 80, 26)
    ax.text(50, 6.5, 'PPO：AdamW (β=0.9/0.98, wd 0.01)；学习率 3e-4→5e-5、clip 0.2→0.05、熵系数 3e-3→0 余弦退火；γ=0.995，λ=0.95，3 轮，批 2048',
            fontsize=6.0, ha='center', color='#475569')
    ax.text(50, 3.0, '奖励：清除增量 + 接近目标 − 壁接触 − 障碍近碰 − 间距违规 − |Δu|²；撞障碍 −20 并终止（同 Turbo 的安全违规）；集群丢失 −10；安全完成 +20',
            fontsize=6.0, ha='center', color='#475569')
    fig.savefig(out/'fig9_tirppo_framework.png', bbox_inches='tight', dpi=300); plt.close(fig)
    return 'fig9_tirppo_framework.png'


FIGS['fig9'] = fig9


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('research/figures/MS_20261006'))
    ap.add_argument('--only', nargs='*', choices=sorted(FIGS))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for k in (a.only or sorted(FIGS)):
        print(k, '->', FIGS[k](a.out), flush=True)


if __name__ == '__main__':
    main()
