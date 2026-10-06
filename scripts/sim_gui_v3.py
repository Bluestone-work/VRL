"""Simulator GUI for benchmark v3 (microscope-visible obstacles).

Panels: A 3-D scene (vessel tree, clots sized by remaining mass, clusters + trails, static obstacles in
brown, moving fragments in red with motion arrows, TPG holds ringed) | B top and C side digital-microscope
views around the focus cluster: rendered cluster blobs, obstacles drawn as dark absorbers (brightfield),
detector boxes (cyan = cluster, orange = obstacle, labelled static / moving) and the policy's correction
arrow | D status table | E removal and minimum obstacle clearance over time.
usage: sim_gui_v3.py --anatomy mca_m1_lvo --clusters 3 --seed 2600100000 [--ckpt C] --frames 200,600
       [--gif out.gif --every 4] [--show]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, '.')
BG, FG, GRID = '#0f1720', '#d8e1ea', '#2a3644'
COL = ['#4fc3f7', '#81c784', '#ce93d8']
CLOT, STATIC, DYN, VESSEL = '#ef5350', '#a1887f', '#ff7043', '#5b6f84'


def build(anatomy, n, seed, ckpt=None):
    from scripts.benchmark_obstacles import Episode
    from marl.image_sensing import CameraConfig
    ep = Episode(n, anatomy, seed, sensing='image', camera=CameraConfig())
    ep.drl = None
    if ckpt:
        from marl.obstacle_control import DRLController
        ep.drl = DRLController(ep.env, ep.sensor, ep.ctl, ckpt)
    ep.hist_pos, ep.t, ep.rem, ep.clear = [], [], [], []
    ep.mode = ['']*n; ep.corr = np.zeros((n, 3)); ep.anatomy = anatomy; ep.ckpt = ckpt
    return ep


def step(ep):
    est, tgt, rule, hold = ep.observe()
    local = ep.drl.act(est, rule, hold, tgt) if ep.drl is not None else rule
    F = ep.ctl.frames(est)
    ep.corr = np.einsum('nji,nj->ni', F, local-rule)
    for i in range(ep.n):
        ep.mode[i] = ('done' if tgt[i] < 0 else 'TPG hold' if hold[i] else 'wait' if not local[i].any() else
                      'avoid' if np.linalg.norm(local[i]-F[i]@ep.ctl.nominal[i]) > .1 else 'route')
    done, out = ep.step(est, local, hold)
    ep.est, ep.tgt, ep.hold = est, tgt, hold
    ep.hist_pos.append(ep.env.positions_mm[:ep.n].copy()); ep.t.append(float(ep.info['elapsed_s']))
    ep.rem.append(1-float(ep.info['remaining_mass'])/ep.initial)
    P, r, _ = ep.field.positions()
    g = np.linalg.norm(ep.env.positions_mm[:ep.n][:, None]-P[None], axis=-1)-ep.ctl.body-r[None] if len(r) else np.full((ep.n, 1), 9.)
    ep.clear.append(float(np.min(np.where(ep.env.active[:ep.n, None], g, 9.))))
    return done


def _camera(ep, cax, axes_, rax, focus, name):
    from matplotlib.patches import Rectangle
    sensor = ep.sensor; P = ep.env.positions_mm.astype(float)
    centre = sensor.track[focus] if sensor.track is not None else P[focus]
    img, o = sensor._render(centre[list(axes_)], rax, getattr(sensor, '_prev_truth', P), P)
    px = sensor.cam.pixel_mm; W = img.shape[0]
    U, V = np.meshgrid(o[0]+np.arange(W)*px, o[1]+np.arange(W)*px, indexing='ij')
    OP, OR, ODYN = ep.field.positions()
    for c, r in zip(OP, OR):                                   # obstacles absorb light (dark under brightfield)
        q = c[list(axes_)]
        img -= .12*np.clip(1-((U-q[0])**2+(V-q[1])**2)/r**2, 0, 1)**.5
    ext = [o[1], o[1]+W*px, o[0]+W*px, o[0]]
    cax.imshow(img, cmap='gray', extent=ext, vmin=0., vmax=.6, aspect='equal')
    for q, a, fl in sensor._detect(img, o):
        if sensor._is_cluster(a, fl):
            h = .12; cax.add_patch(Rectangle((q[1]-h, q[0]-h), 2*h, 2*h, fill=False, lw=1., ec=COL[focus]))
            cax.text(q[1]-h, q[0]-h-.02, 'cluster', color=COL[focus], fontsize=6.5)
    HW, HX = .9, .5*W*px-.02                       # displayed half-height / half-width (inside the rendered window)
    for c, r, dy in zip(OP, OR, ODYN):
        q = c[list(axes_)]
        if abs(q[0]-centre[axes_[0]]) < HW-r*.5 and abs(q[1]-centre[axes_[1]]) < HX-r*.5:
            cax.add_patch(Rectangle((q[1]-r, q[0]-r), 2*r, 2*r, fill=False, lw=1., ec=DYN if dy else STATIC))
            cax.text(q[1]-r, q[0]-r-.02, 'moving' if dy else 'static', color=DYN if dy else STATIC, fontsize=6.5)
    if np.linalg.norm(ep.corr[focus]) > .05:
        d = ep.corr[focus][list(axes_)]
        cax.annotate('', xy=(centre[axes_[1]]+.6*d[1], centre[axes_[0]]+.6*d[0]), xytext=(centre[axes_[1]], centre[axes_[0]]),
                     arrowprops=dict(arrowstyle='->', color='#ff4081', lw=1.5))
    cax.set_xlim(centre[axes_[1]]-HX, centre[axes_[1]]+HX); cax.set_ylim(centre[axes_[0]]+HW, centre[axes_[0]]-HW)
    x0, y0 = centre[axes_[1]]-HX+.1, centre[axes_[0]]+HW-.1
    cax.plot([x0, x0+.5], [y0, y0], color='w', lw=2); cax.text(x0+.25, y0-.06, '500 µm', color='w', fontsize=6.5, ha='center')
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_color(GRID)
    cax.set_title(f'{name}   (cluster {focus}, {px*1000:.0f} µm/px)', color=FG, fontsize=8, loc='left')


def draw(ep, fig):
    env, n = ep.env, ep.n
    fig.clf(); fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 4, width_ratios=[1.3, 1.3, 1, 1], height_ratios=[1, 1, .75], wspace=.12, hspace=.3,
                          left=.02, right=.985, top=.93, bottom=.06)
    who = ('T-IRPPO (Transformer residual PPO)' if ep.ckpt else 'route pursuit + APF')
    fig.text(.02, .965, f'VascuSwarm Sim  ·  obstacle benchmark v3  ·  {who}', color=FG, fontsize=12, weight='bold', va='center')
    fig.text(.985, .965, f"{ep.anatomy}   N={n}   t = {env.elapsed_s:6.1f} s / 300 s   obstacles {len(ep.field.static_r)} static + {len(ep.field.dyn)} moving",
             color=FG, fontsize=9, ha='right', va='center')
    ax = fig.add_subplot(gs[0:2, 0:2], projection='3d'); ax.set_facecolor(BG)
    t = env.transport; pts = t.points.astype(float); rad = np.asarray(env.flow_model.healthy_radius_mm, float)
    for a, b in t.ends:
        ax.plot(pts[[a, b], 0], pts[[a, b], 1], pts[[a, b], 2], color=VESSEL, lw=3.2*(rad[a]+rad[b]), alpha=.5, solid_capstyle='round')
    m = env.masses/np.maximum(env.initial_mass, 1e-9); alive = m > 0
    ax.scatter(*env.clot_positions_mm[alive].T, s=30+120*m[alive], c=CLOT, edgecolors='w', linewidths=.5, depthshade=False)
    OP, OR, ODYN = ep.field.positions()
    ax.scatter(*OP[~ODYN].T, s=250*OR[~ODYN]**2+10, c=STATIC, marker='o', edgecolors='k', linewidths=.3, depthshade=False)
    ax.scatter(*OP[ODYN].T, s=600*OR[ODYN]**2+14, c=DYN, marker='D', edgecolors='k', linewidths=.3, depthshade=False)
    H = np.array(ep.hist_pos) if ep.hist_pos else None; P = env.positions_mm
    for i in range(n):
        if H is not None:
            ax.plot(H[:, i, 0], H[:, i, 1], H[:, i, 2], color=COL[i], lw=1.1, alpha=.9)
        if env.active[i]:
            ax.scatter(*P[i], s=70, c=COL[i], edgecolors='w', linewidths=.8, depthshade=False)
            if getattr(ep, 'hold', None) is not None and ep.hold[i]:
                ax.scatter(*P[i], s=260, facecolors='none', edgecolors='#fff176', linewidths=1.2, depthshade=False)
    lo, hi = pts.min(0), pts.max(0); span = hi-lo
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(span, .25*span.max())), zoom=1.75); ax.set_axis_off(); ax.view_init(elev=24, azim=-62)
    ax.text2D(.01, .97, 'A  3-D scene   (brown: static obstacle, orange: moving fragment, red: clot)', transform=ax.transAxes, color=FG, fontsize=9, weight='bold')
    est = getattr(ep, 'est', None)
    if est is not None:
        dist = [min([np.linalg.norm(rel)-rr for rel, _, rr in est.obstacles[i]] or [9.]) if est.active[i] else 99. for i in range(n)]
        focus = int(np.argmin(dist))
    else:
        focus = 0
    for k, (axes_, rax, name) in enumerate([((0, 1), [0, 1], 'B  top microscope x–y'), ((0, 2), [0, 2], 'C  side microscope x–z')]):
        _camera(ep, fig.add_subplot(gs[k, 2:4]), axes_, rax, focus, name)
    dax = fig.add_subplot(gs[2, 0]); dax.set_axis_off()
    rem = ep.rem[-1] if ep.rem else 0.
    lines = [f'removal          {100*rem:5.1f} %', f'obstacle hits    {int(ep.field.events.sum())}',
             f'wall contact     {ep.wall_total:5.2f} robot-s', f'lost clusters    {int(ep.info["lost_robots"]) if ep.info else 0}', '',
             'cluster  target  mode       nearest obstacle']
    for i in range(n):
        tg = '—' if est is None or ep.tgt[i] < 0 else f'clot {int(ep.tgt[i])}'
        dd = '—' if est is None or not est.obstacles[i] else f'{1000*min(np.linalg.norm(rel)-ep.ctl.body-rr for rel, _, rr in est.obstacles[i]):5.0f} µm'
        lines.append(f'  {i}      {tg:7s} {ep.mode[i]:9s}  {dd}')
    for k, l in enumerate(lines):
        dax.text(0, 1-k*.11, l, color=COL[k-6] if 6 <= k < 6+n else FG, fontsize=8.2, family='monospace', transform=dax.transAxes, va='top')
    dax.text(0, 1.12, 'D  status', color=FG, fontsize=9, weight='bold', transform=dax.transAxes)
    eax = fig.add_subplot(gs[2, 1:4]); eax.set_facecolor(BG)
    if ep.t:
        eax.plot(ep.t, 100*np.array(ep.rem), color='#4fc3f7', lw=1.5)
        e2 = eax.twinx(); e2.plot(ep.t, 1000*np.clip(ep.clear, -1, 2), color=DYN, lw=1.)
        e2.axhline(0, color=DYN, lw=.6, ls=':'); e2.set_ylim(-100, 1500)
        e2.tick_params(colors=FG, labelsize=7); e2.set_ylabel('min obstacle clearance (µm)', color=DYN, fontsize=7.5)
        for s in e2.spines.values():
            s.set_color(GRID)
    eax.set_xlim(0, 300); eax.set_ylim(0, 102); eax.tick_params(colors=FG, labelsize=7)
    eax.set_xlabel('time (s)', color=FG, fontsize=7.5); eax.set_ylabel('removal (%)', color='#4fc3f7', fontsize=7.5)
    for s in eax.spines.values():
        s.set_color(GRID)
    eax.set_title('E  progress and safety margin', color=FG, fontsize=9, weight='bold', loc='left')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--anatomy', default='mca_m1_lvo'); ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2600100000); ap.add_argument('--ckpt')
    ap.add_argument('--frames', default=''); ap.add_argument('--gif'); ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--out', type=Path, default=Path('research/figures/GUI_V3_20261006')); ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    if not a.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.out.mkdir(parents=True, exist_ok=True)
    ep = build(a.anatomy, a.clusters, a.seed, a.ckpt)
    want = {int(x) for x in a.frames.split(',') if x}; tag = 'drl' if a.ckpt else 'apf'
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG); frames = []
    for k in range(1, a.max_steps+1):
        done = step(ep)
        if k in want:
            draw(ep, fig); fig.savefig(a.out/f'gui_{tag}_{a.anatomy}_N{a.clusters}_s{k:04d}.png', dpi=150, facecolor=BG)
        if a.gif and k % a.every == 0:
            draw(ep, fig); fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if a.show and k % 5 == 0:
            draw(ep, fig); plt.pause(.001)
        if done or (want and not a.gif and not a.show and k >= max(want)):
            break
    if a.gif and frames:
        from PIL import Image
        ims = [Image.fromarray(f).resize((960, 540)) for f in frames]
        ims[0].save(a.out/a.gif, save_all=True, append_images=ims[1:], duration=80, loop=0)
    print('steps', k, 'safe', ep.safe() if ep.info else None, 'obstacle hits', int(ep.field.events.sum()))


if __name__ == '__main__':
    main()
