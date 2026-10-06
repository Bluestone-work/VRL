"""Simulator GUI: a live dashboard of one episode of the deployable pipeline (cf. the simulation windows
of Medany et al. 2025, Fig. 1e, and An et al. 2026, Supplementary Movie S2).

Panels
  A  3-D vessel tree (healthy lumen), clots sized by remaining mass, clusters with their trails, debris
     particles, TPG conflict zones; clusters held by TPG are ringed
  B  top camera (x-y) and C side camera (x-z): the rendered frames the controller receives, with the
     detector's boxes (cluster / particle) and the learned residual arrow
  D  status: time, removal, per-cluster target / mode (route, hold, wait, residual) / wall clearance
  E  removal curve and particle-contact events over time
usage: sim_gui.py --anatomy mca_m1_lvo --clusters 3 --seed 2600100000 [--drl CKPT] --frames 0,300,900
       [--video out.mp4 --every 5] [--show]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, '.')
BG, FG, GRID = '#0f1720', '#d8e1ea', '#2a3644'
COL = ['#4fc3f7', '#81c784', '#ce93d8']
CLOT, PART, VESSEL, ZONE = '#ef5350', '#ffb74d', '#5b6f84', '#fff176'


def build(anatomy, n, seed, drl=None, camera='default'):
    from environments.mca_physical_env import DynamicsConfig
    from marl.deployable_sensing import DeployablePursuit
    from marl.image_sensing import CameraConfig, ImageSensor
    from marl.multicluster import MultiClusterConfig
    from marl.teacher import TEACHER_CONFIG
    from marl.tpg_coordinator import TPGCoordinator
    import scripts.benchmark_multicluster as bm
    from scripts.multicluster_protocol import paired_environment
    env, _ = paired_environment(replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy,
                                        episode_duration_s=300., junction_model='union'), n, seed)
    sensor = ImageSensor(env, CameraConfig(), seed=seed)
    plan, _ = bm.preoperative_plan(env)
    bm.FALLBACK['mode'] = 'park' if n > 1 else 'help'
    ctl = DeployablePursuit(env, sensor)
    coord = None
    if n > 1:
        _, sp = bm._station_paths(env); coord = TPGCoordinator(env, plan, sp, d_min_mm=2.)
    mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n,
                            min_spacing_mm=2. if n > 1 else 0.)
    shield = bm.Shield(mc, robot_speed_mm_s=env.config.robot_speed_mm_s, control_dt_s=env.config.control_dt_s)
    irc = None
    if drl:
        from marl.drl_local import IRController
        irc = IRController(env, sensor, ctl, drl)
    return dict(env=env, sensor=sensor, ctl=ctl, coord=coord, shield=shield, targets=bm.PlanTargets(plan),
                irc=irc, plan=plan, n=n, prev=np.zeros((n, 3)), hist=[], removal=[], events=[], t=[],
                mode=['']*n, residual=np.zeros((n, 3)), anatomy=anatomy)


def step(S):
    from scripts.benchmark_deployable import packet_from
    import scripts.benchmark_multicluster as bm
    env, n = S['env'], S['n']
    bm.FALLBACK['mode'] = 'park' if n > 1 else 'help'
    est = S['sensor'].observe(); tgt = S['targets'].targets(env, est.pos)
    rule = S['ctl'].act(tgt, est)
    hold = S['coord'].gate(est.pos, est.active) if S['coord'] is not None else np.zeros(n, bool)
    local = rule.copy()
    if S['irc'] is not None:
        local = S['irc'].act(est, rule, hold, tgt)
    F = S['ctl'].frames(est)
    S['residual'] = np.einsum('nji,nj->ni', F, local-rule)
    for i in range(n):
        S['mode'][i] = ('done' if tgt[i] < 0 else 'TPG hold' if hold[i] else
                        'wait' if not rule[i].any() and not local[i].any() else
                        'residual' if np.linalg.norm(local[i]-rule[i]) > .05 else 'route')
    local[hold] = 0.
    if n > 1:
        local = S['shield'].filtered(local, packet_from(est, F, S['prev']))
    S['prev'] = local.copy()
    _, _, term, trunc, info = env.step(S['ctl'].to_world(local, est))
    S['est'], S['tgt'], S['hold'], S['info'] = est, tgt, hold, info
    S['hist'].append(env.positions_mm[:n].copy())
    S['removal'].append(1-float(info['remaining_mass'])/float(env.initial_mass.sum()))
    S['events'].append(int(info['episode_particle_collision_events'])); S['t'].append(float(info['elapsed_s']))
    return bool(term or trunc)


def draw(S, fig=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    env, n, sensor = S['env'], S['n'], S['sensor']
    if fig is None:
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    fig.clf(); fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 4, width_ratios=[1.3, 1.3, 1, 1], height_ratios=[1, 1, .75], wspace=.12, hspace=.3,
                          left=.02, right=.985, top=.93, bottom=.06)
    fig.text(.02, .965, 'VascuSwarm Sim  ·  deployable pipeline  ·  IR-PPO residual' if S['irc'] else
             'VascuSwarm Sim  ·  deployable pipeline  ·  rule', color=FG, fontsize=12, weight='bold', va='center')
    fig.text(.985, .965, f"{S['anatomy']}   N={n}   t = {env.elapsed_s:6.1f} s / 300 s   union-of-tubes lumen",
             color=FG, fontsize=9, ha='right', va='center')
    # A: 3-D
    ax = fig.add_subplot(gs[0:2, 0:2], projection='3d'); ax.set_facecolor(BG)
    t = env.transport; pts = t.points.astype(float); rad = np.asarray(env.flow_model.healthy_radius_mm, float)
    for a, b in t.ends:
        ax.plot(pts[[a, b], 0], pts[[a, b], 1], pts[[a, b], 2], color=VESSEL, lw=3.2*(rad[a]+rad[b]), alpha=.55,
                solid_capstyle='round')
    if S['coord'] is not None:
        for z in S['coord'].zones:
            seg = S['coord'].paths[z['i']][z['s0']:z['s1']+1]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=ZONE, lw=1.4, alpha=.6)
    m = env.masses/np.maximum(env.initial_mass, 1e-9)
    alive = m > 0
    ax.scatter(*env.clot_positions_mm[alive].T, s=30+120*m[alive], c=CLOT, marker='o', edgecolors='w',
               linewidths=.5, depthshade=False)
    P = env.positions_mm; act = env.active
    ax.scatter(*P[n:][act[n:]].T, s=4, c=PART, depthshade=False, alpha=.9)
    H = np.array(S['hist']) if S['hist'] else None
    for i in range(n):
        if H is not None:
            ax.plot(H[:, i, 0], H[:, i, 1], H[:, i, 2], color=COL[i], lw=1.1, alpha=.85)
        if act[i]:
            ax.scatter(*P[i], s=70, c=COL[i], edgecolors='w', linewidths=.8, depthshade=False)
            if S.get('hold') is not None and S['hold'][i]:
                ax.scatter(*P[i], s=260, facecolors='none', edgecolors=ZONE, linewidths=1.2, depthshade=False)
    ax.set_axis_off(); ax.view_init(elev=24, azim=-62)
    lo, hi = pts.min(0), pts.max(0); c = (lo+hi)/2; span = hi-lo
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(span, .25*span.max())), zoom=1.75)
    try:
        ax.set_proj_type('persp', focal_length=.6)
    except TypeError:
        pass
    ax.text2D(.01, .97, 'A  3-D scene', transform=ax.transAxes, color=FG, fontsize=9, weight='bold')
    # B, C: camera views around the cluster with the most particles nearby
    est = S.get('est')
    focus = 0 if est is None else int(np.argmax([len(est.particles[i]) if est.active[i] else -1 for i in range(n)]))
    for k, (axes_, rax, name) in enumerate([((0, 1), [0, 1], 'B  top camera x–y'), ((0, 2), [0, 2], 'C  side camera x–z')]):
        cax = fig.add_subplot(gs[k, 2:4]); cax.set_facecolor('k')
        centre = sensor.track[focus] if sensor.track is not None else P[focus]
        prev = getattr(sensor, '_prev_truth', P.astype(float))
        img, o = sensor._render(centre[list(axes_)], rax, prev, P.astype(float))
        px = sensor.cam.pixel_mm
        ext = [o[1], o[1]+img.shape[1]*px, o[0]+img.shape[0]*px, o[0]]
        cax.imshow(img, cmap='gray', extent=ext, vmin=sensor.cam.background-.05, vmax=.6, aspect='equal')
        for q, a, fl in sensor._detect(img, o):
            big = sensor._is_cluster(a, fl); h = .13 if big else .06
            cax.add_patch(Rectangle((q[1]-h, q[0]-h), 2*h, 2*h, fill=False, lw=1., ec=COL[focus] if big else PART))
            cax.text(q[1]-h, q[0]-h-.015, 'cluster' if big else 'particle', color=COL[focus] if big else PART, fontsize=6.5)
        if np.linalg.norm(S['residual'][focus]) > .02:
            d = S['residual'][focus][list(axes_)]
            cax.annotate('', xy=(centre[axes_[1]]+1.5*d[1], centre[axes_[0]]+1.5*d[0]), xytext=(centre[axes_[1]], centre[axes_[0]]),
                         arrowprops=dict(arrowstyle='->', color='#ff4081', lw=1.4))
        HW = .6
        cax.set_xlim(centre[axes_[1]]-HW*2.7, centre[axes_[1]]+HW*2.7); cax.set_ylim(centre[axes_[0]]+HW, centre[axes_[0]]-HW)
        x0, y0 = centre[axes_[1]]-HW*2.7+.08, centre[axes_[0]]+HW-.08
        cax.plot([x0, x0+.2], [y0, y0], color='w', lw=2); cax.text(x0+.1, y0-.04, '200 µm', color='w', fontsize=6.5, ha='center')
        cax.set_xticks([]); cax.set_yticks([])
        for s in cax.spines.values():
            s.set_color(GRID)
        cax.set_title(f'{name}   (tracking cluster {focus}, {px*1000:.0f} µm/px)', color=FG, fontsize=8, loc='left')
    # D: status table
    dax = fig.add_subplot(gs[2, 0]); dax.set_axis_off(); dax.set_facecolor(BG)
    removal = S['removal'][-1] if S['removal'] else 0.
    lines = [f'removal   {100*removal:5.1f} %', f'particle events  {S["events"][-1] if S["events"] else 0}',
             f'lost clusters  {int(S["info"]["lost_robots"]) if "info" in S else 0}', '']
    lines += ['cluster  target  mode        clearance']
    for i in range(n):
        tg = '—' if est is None or S['tgt'][i] < 0 else f'clot {int(S["tgt"][i])}'
        cl = '—'
        if est is not None and est.active[i]:
            _, rr, rd = sensor.map_coordinates(est, i); cl = f'{1000*(rr-rd-env.config.robot_radius_mm):4.0f} µm'
        lines.append(f'  {i}      {tg:7s} {S["mode"][i]:10s}  {cl}')
    for k, l in enumerate(lines):
        dax.text(0, 1-k*.12, l, color=COL[k-5] if 5 <= k < 5+n else FG, fontsize=8.2, family='monospace',
                 transform=dax.transAxes, va='top')
    dax.text(0, 1.12, 'D  status', color=FG, fontsize=9, weight='bold', transform=dax.transAxes)
    # E: curves
    eax = fig.add_subplot(gs[2, 1:4]); eax.set_facecolor(BG)
    if S['t']:
        eax.plot(S['t'], 100*np.array(S['removal']), color='#4fc3f7', lw=1.5, label='clot removal (%)')
        e2 = eax.twinx(); e2.step(S['t'], S['events'], color=PART, lw=1.1, where='post', label='particle events')
        e2.tick_params(colors=FG, labelsize=7); e2.set_ylim(0, max(3, max(S['events'])+1))
        for s in e2.spines.values():
            s.set_color(GRID)
        e2.set_ylabel('particle events', color=PART, fontsize=7.5)
    eax.set_xlim(0, 300); eax.set_ylim(0, 102); eax.tick_params(colors=FG, labelsize=7)
    eax.set_xlabel('time (s)', color=FG, fontsize=7.5); eax.set_ylabel('removal (%)', color='#4fc3f7', fontsize=7.5)
    for s in eax.spines.values():
        s.set_color(GRID)
    eax.set_title('E  progress', color=FG, fontsize=9, weight='bold', loc='left')
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--anatomy', default='mca_m1_lvo'); ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2600100000); ap.add_argument('--drl')
    ap.add_argument('--frames', default='', help='comma-separated steps to save as PNG')
    ap.add_argument('--video'); ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--out', type=Path, default=Path('research/figures/GUI_20261006'))
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    if not a.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.out.mkdir(parents=True, exist_ok=True)
    S = build(a.anatomy, a.clusters, a.seed, a.drl)
    want = {int(x) for x in a.frames.split(',') if x}
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    writer = None
    if a.video:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=12, bitrate=3000); writer.setup(fig, str(a.out/a.video), dpi=110)
    k = 0
    while True:
        done = step(S); k += 1
        if k in want:
            draw(S, fig); fig.savefig(a.out/f'gui_{a.anatomy}_N{a.clusters}_step{k:04d}.png', dpi=150, facecolor=BG)
        if writer is not None and k % a.every == 0:
            draw(S, fig); writer.grab_frame()
        if a.show and k % 5 == 0:
            draw(S, fig); plt.pause(.001)
        if done or (want and not writer and not a.show and k >= max(want)):
            break
    if writer is not None:
        writer.finish()
    draw(S, fig); fig.savefig(a.out/f'gui_{a.anatomy}_N{a.clusters}_final.png', dpi=150, facecolor=BG)
    print('steps', k, 'removal', S['removal'][-1], 'events', S['events'][-1])


if __name__ == '__main__':
    main()
