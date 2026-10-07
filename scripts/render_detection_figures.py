"""Publication figures for the obstacle benchmark: (A) the 3-D scene and (B) what the controller perceives.

  fig_A_scene3d          vessel tree as shaded tubes, clots, static obstacles, moving fragments (with motion
                         arrows), the cluster and its trail, and the microscope field of view
  fig_B_top / fig_B_side the two biplane microscope views (top x-y, side x-z) around the cluster: rendered
                         image, true obstacles (solid) and the detector boxes the policy receives (dashed)
  fig_B_detection3d      the same detections in 3-D: true obstacles as spheres, detector output as 3-D boxes,
                         i.e. what the two 2-D views are fused into
usage: render_detection_figures.py --anatomy ica_siphon --seed 2600000001 --step 286 --out DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, '.')
from scripts.sim_gui_v3 import build, step  # noqa: E402

C_CLUSTER, C_CLOT, C_STATIC, C_DYN, C_DET = '#1f77b4', '#d62728', '#8d6e63', '#ff7f0e', '#00acc1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': .8,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03})


def tube(ax, a, b, ra, rb, n=14, **kw):
    """Shaded frustum between points a and b."""
    d = b-a; L = np.linalg.norm(d)
    if L < 1e-9:
        return
    d /= L; h = np.eye(3)[int(np.argmin(np.abs(d)))]
    e1 = np.cross(d, h); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    th = np.linspace(0, 2*np.pi, n); s = np.array([0., 1.])
    R = ra+(rb-ra)*s[:, None]
    X = a[None, None]+s[:, None, None]*L*d+R[..., None]*(np.cos(th)[None, :, None]*e1+np.sin(th)[None, :, None]*e2)
    ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], linewidth=0, antialiased=True, shade=True, **kw)


def tube_window(ax, a, b, ra, rb, c0, w, step=.4, **kw):
    """Draw only the parts of a vessel segment inside the cube |x-c0| < w (mplot3d does not clip surfaces)."""
    L = np.linalg.norm(b-a); k = max(int(np.ceil(L/step)), 1)
    for j in range(k):
        s0, s1 = j/k, (j+1)/k; p0, p1 = a+s0*(b-a), a+s1*(b-a)
        if np.all(np.abs((p0+p1)/2-c0) < w-max(ra, rb)*.5):
            tube(ax, p0, p1, ra+s0*(rb-ra), ra+s1*(rb-ra), **kw)


def sphere(ax, c, r, n=16, **kw):
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n//2+1))
    ax.plot_surface(c[0]+r*np.cos(u)*np.sin(v), c[1]+r*np.sin(u)*np.sin(v), c[2]+r*np.cos(v),
                    linewidth=0, antialiased=True, shade=True, **kw)


def box3d(ax, c, r, **kw):
    o = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*r+c
    for i in range(8):
        for j in range(i+1, 8):
            if np.sum(np.abs(o[i]-o[j]) > 1e-9) == 1:
                ax.plot(*o[[i, j]].T, **kw)


def equal3d(ax, lo, hi):
    c = (lo+hi)/2; h = (hi-lo).max()/2
    ax.set_xlim(c[0]-h, c[0]+h); ax.set_ylim(c[1]-h, c[1]+h); ax.set_zlim(c[2]-h, c[2]+h); ax.set_box_aspect((1, 1, 1))


def clean3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor('#d0d0d0'); a._axinfo['grid']['color'] = '#ececec'
    ax.tick_params(labelsize=7, pad=1)
    ax.set_xlabel('x (mm)', labelpad=-4, fontsize=8); ax.set_ylabel('y (mm)', labelpad=-4, fontsize=8); ax.set_zlabel('z (mm)', labelpad=-4, fontsize=8)


def fig_scene(ep, out, fov):
    env = ep.env; t = env.transport; pts = t.points.astype(float); rad = np.asarray(env.flow_model.healthy_radius_mm, float)
    P = env.positions_mm[0].astype(float); OP, OR, ODYN = ep.field.positions()
    fig = plt.figure(figsize=(9.0, 6.0)); ax = fig.add_axes([-.04, .08, .52, .86], projection='3d')
    for a, b in t.ends:
        tube(ax, pts[a], pts[b], rad[a], rad[b], color='#9fb3c8', alpha=.22)
    m = env.masses/np.maximum(env.initial_mass, 1e-9)
    for c, mm in zip(env.clot_positions_mm, m):
        if mm > 0:
            sphere(ax, c, .35+.35*mm, color=C_CLOT, alpha=.95)
        else:
            ax.scatter(*c, s=18, facecolors='none', edgecolors=C_CLOT, linewidths=.8, depthshade=False)
    for c, r, dy in zip(OP, OR, ODYN):
        sphere(ax, c, r, n=12, color=C_DYN if dy else C_STATIC, alpha=.95)
    if len(ep.hist_pos) > 1:
        H = np.array(ep.hist_pos)[:, 0]; ax.plot(*H.T, color=C_CLUSTER, lw=1.6)
    sphere(ax, P, float(env.config.robot_radius_mm), color=C_CLUSTER, alpha=1.)
    box3d(ax, P, fov, color='#455a64', lw=.7, ls=(0, (3, 2)))
    lo, hi = pts.min(0)-2, pts.max(0)+2
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(tuple(hi-lo), zoom=1.05)
    clean3d(ax); ax.view_init(elev=18, azim=-58)
    from matplotlib.ticker import MaxNLocator
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(MaxNLocator(3))
    zx = fig.add_axes([.42, .10, .58, .82], projection='3d')       # zoom: 6 mm around the cluster
    w = 3.
    for a, b in t.ends:
        tube_window(zx, pts[a], pts[b], rad[a], rad[b], P, w, n=28, color='#9fb3c8', alpha=.10)
    for c, r, dy in zip(OP, OR, ODYN):
        if np.all(np.abs(c-P) < w+r):
            sphere(zx, c, r, n=16, color=C_DYN if dy else C_STATIC, alpha=.95)
    for c, mm in zip(env.clot_positions_mm, m):
        if mm > 0 and np.all(np.abs(c-P) < w+1):
            sphere(zx, c, .35+.35*mm, color=C_CLOT, alpha=.9)
    if len(ep.hist_pos) > 1:
        H = np.array(ep.hist_pos)[:, 0]; H = H[np.all(np.abs(H-P) < w, 1)]; zx.plot(*H.T, color=C_CLUSTER, lw=1.2)
    sphere(zx, P, float(env.config.robot_radius_mm), color=C_CLUSTER, alpha=1.)
    box3d(zx, P, fov, color='#455a64', lw=.8, ls=(0, (3, 2)))
    equal3d(zx, P-w, P+w); clean3d(zx); zx.view_init(elev=18, azim=-58); zx.tick_params(labelsize=6)
    zx.set_title('zoom: 6 mm around the cluster', fontsize=8)
    leg = [Line2D([], [], marker='o', ls='', color=C_CLOT, label='clot (size ∝ remaining mass)'),
           Line2D([], [], marker='o', ls='', color=C_STATIC, label='static obstacle (mural residue)'),
           Line2D([], [], marker='o', ls='', color=C_DYN, label='moving fragment'),
           Line2D([], [], marker='o', ls='', color=C_CLUSTER, label='robot cluster + trail'),
           Line2D([], [], ls=(0, (3, 2)), color='#455a64', label='microscope field of view')]
    fig.legend(handles=leg, loc='lower center', ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(.5, .0))
    fig.suptitle(f'{ep.anatomy}   t = {env.elapsed_s:.1f} s   ({len(ep.field.static_r)} static, {len(ep.field.dyn)} moving obstacles)', fontsize=10, y=.99)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'fig_A_scene3d.{ext}')
    plt.close(fig)


def missed(ep, fov=3.):
    """True obstacles in the detector range with no detection within 2 radii (miss or latency)."""
    est = ep.est; OP, OR, _ = ep.field.positions(); D = [rel+est.pos[0] for rel, _, _ in est.obstacles[0]]; out = []
    for c, r in zip(OP, OR):
        if np.linalg.norm(c-est.pos[0])-r < fov and not any(np.linalg.norm(c-d) < 2*r+.05 for d in D):
            out.append((c, r))
    return out


def good_frame(ep, half=.75):
    """A detected obstacle within 0.5 mm surface gap and inside the zoomed view."""
    est = ep.est
    if not est.active[0]:
        return False
    for rel, _, r in est.obstacles[0]:
        if np.linalg.norm(rel)-ep.ctl.body-r < .5 and np.all(np.abs(rel[:2]) < half-r*.3) and abs(rel[2]) < half-r*.3:
            return True
    return False


def fig_view(ep, out, axes_, name, tag, half=.75):
    sensor = ep.sensor; P = ep.env.positions_mm.astype(float); est = ep.est
    centre = sensor.track[0] if sensor.track is not None else P[0]
    img, o = sensor._render(centre[list(axes_)], list(axes_), getattr(sensor, '_prev_truth', P), P)
    px = sensor.cam.pixel_mm; W = img.shape[0]
    U, V = np.meshgrid(o[0]+np.arange(W)*px, o[1]+np.arange(W)*px, indexing='ij')
    OP, OR, ODYN = ep.field.positions()
    for c, r in zip(OP, OR):
        q = c[list(axes_)]; img = img-.12*np.clip(1-((U-q[0])**2+(V-q[1])**2)/r**2, 0, 1)**.5
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img.T, cmap='gray', origin='lower', extent=[o[0], o[0]+W*px, o[1], o[1]+W*px], vmin=0, vmax=.6, interpolation='nearest')
    c0 = centre[list(axes_)]
    for q, a, fl in sensor._detect(img, o):
        if sensor._is_cluster(a, fl):
            h = .11; ax.add_patch(Rectangle((q[0]-h, q[1]-h), 2*h, 2*h, fill=False, lw=1.2, ec=C_CLUSTER))
    for c, r, dy in zip(OP, OR, ODYN):
        q = c[list(axes_)]
        if np.all(np.abs(q-c0) < half+r):
            ax.add_patch(Rectangle((q[0]-r, q[1]-r), 2*r, 2*r, fill=False, lw=1.2, ec=C_DYN if dy else C_STATIC))
            ax.text(q[0]-r+.015, q[1]+r-.015, 'moving' if dy else 'static', color=C_DYN if dy else C_STATIC, fontsize=7,
                    va='top', clip_on=True, bbox=dict(fc='black', ec='none', alpha=.45, pad=.6))
    for rel, _, r in est.obstacles[0]:
        q = (rel+est.pos[0])[list(axes_)]
        if np.all(np.abs(q-c0) < half+r):
            ax.add_patch(Rectangle((q[0]-r, q[1]-r), 2*r, 2*r, fill=False, lw=1.4, ls=(0, (3, 2)), ec=C_DET))
            gap = np.linalg.norm(rel)-ep.ctl.body-r
            ax.text(q[0]-r+.015, q[1]-r+.015, f'{1000*gap:.0f} µm', color=C_DET, fontsize=7, va='bottom', clip_on=True,
                    bbox=dict(fc='black', ec='none', alpha=.45, pad=.6))
    for c, r in missed(ep):
        q = c[list(axes_)]
        if np.all(np.abs(q-c0) < half+r):
            ax.text(q[0], q[1], '×\nmissed', color='#ff5252', fontsize=7, ha='center', va='center')
    q = est.pos[0][list(axes_)]; ax.plot(*q, '+', color='#ffeb3b', ms=9, mew=1.4)
    ax.set_xlim(c0[0]-half, c0[0]+half); ax.set_ylim(c0[1]-half, c0[1]+half)
    x0, y0 = c0[0]-half+.08, c0[1]-half+.08
    ax.plot([x0, x0+.5], [y0, y0], color='w', lw=2.5, solid_capstyle='butt'); ax.text(x0+.25, y0+.04, '500 µm', color='w', fontsize=8, ha='center')
    lab = 'xyz'; ax.set_xlabel(f'{lab[axes_[0]]} (mm)'); ax.set_ylabel(f'{lab[axes_[1]]} (mm)')
    ax.set_title(f'{name}  ({px*1000:.0f} µm/px)', fontsize=9)
    leg = [Line2D([], [], color=C_CLUSTER, lw=1.2, label='cluster detection'),
           Line2D([], [], color='#ffeb3b', marker='+', ls='', ms=8, label='fused 3-D cluster estimate'),
           Line2D([], [], color=C_STATIC, lw=1.2, label='true static obstacle'),
           Line2D([], [], color=C_DYN, lw=1.2, label='true moving fragment'),
           Line2D([], [], color=C_DET, lw=1.4, ls=(0, (3, 2)), label='detector box (policy input), surface gap')]
    ax.legend(handles=leg, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=7, frameon=False)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'fig_B_{tag}.{ext}')
    plt.close(fig)


def fig_det3d(ep, out, half=1.0):
    est = ep.est; P = ep.env.positions_mm[0].astype(float); OP, OR, ODYN = ep.field.positions()
    fig = plt.figure(figsize=(5.4, 5.6)); ax = fig.add_subplot(projection='3d')
    t = ep.env.transport; pts = t.points.astype(float); rad = np.asarray(ep.env.flow_model.healthy_radius_mm, float)
    for a, b in t.ends:
        tube_window(ax, pts[a], pts[b], rad[a], rad[b], P, half+.6, step=.2, n=32, color='#9fb3c8', alpha=.08)
    for c, r, dy in zip(OP, OR, ODYN):
        if np.all(np.abs(c-P) < half+r):
            sphere(ax, c, r, color=C_DYN if dy else C_STATIC, alpha=.85)
    for rel, _, r in est.obstacles[0]:
        c = rel+est.pos[0]
        if np.all(np.abs(c-P) < half+r):
            box3d(ax, c, r, color=C_DET, lw=1.1, ls=(0, (3, 2)))
    sphere(ax, P, float(ep.env.config.robot_radius_mm), color=C_CLUSTER, alpha=1.)
    ax.scatter(*est.pos[0], marker='+', s=80, c='#f9a825', linewidths=1.6, depthshade=False)
    for c, r in missed(ep):
        if np.all(np.abs(c-P) < half+r):
            ax.scatter(*c, marker='x', s=60, c='#ff5252', linewidths=1.5, depthshade=False)
    if len(ep.hist_pos) > 1:
        H = np.array(ep.hist_pos)[-80:, 0]; ax.plot(*H.T, color=C_CLUSTER, lw=1.4)
    equal3d(ax, P-half, P+half); clean3d(ax); ax.view_init(elev=20, azim=-55)
    leg = [Line2D([], [], marker='o', ls='', color=C_CLUSTER, label='cluster (true)'),
           Line2D([], [], marker='+', ls='', color='#f9a825', ms=9, label='cluster estimate (biplane)'),
           Line2D([], [], marker='o', ls='', color=C_STATIC, label='static obstacle (true)'),
           Line2D([], [], marker='o', ls='', color=C_DYN, label='moving fragment (true)'),
           Line2D([], [], color=C_DET, ls=(0, (3, 2)), label='3-D detector box (policy input)'),
           Line2D([], [], marker='x', ls='', color='#ff5252', label='true obstacle not detected this frame')]
    ax.legend(handles=leg, loc='upper center', fontsize=7, frameon=False, bbox_to_anchor=(.5, -.02), ncol=2)
    ax.set_title('Perceived 3-D obstacle field around the cluster (2 mm window)', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'fig_B_detection3d.{ext}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anatomy', default='ica_siphon'); p.add_argument('--seed', type=int, default=2600000001)
    p.add_argument('--step', type=int, default=0, help='fixed step; 0 = first frame after 60 with a detected obstacle within 0.5 mm'); p.add_argument('--out', type=Path, default=Path('research/figures/DETECTION_FIGS_20261007'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    ep = build(a.anatomy, 1, a.seed)
    for k in range(1, 3001):
        if step(ep) or (a.step and k >= a.step) or (not a.step and k > 60 and good_frame(ep)):
            break
    print('frame', k, 't', round(ep.env.elapsed_s, 1), 'detections', len(ep.est.obstacles[0]), 'missed', len(missed(ep)))
    fig_scene(ep, a.out, fov=.75)
    fig_view(ep, a.out, (0, 1), 'Top microscope view (x–y)', 'top')
    fig_view(ep, a.out, (0, 2), 'Side microscope view (x–z)', 'side')
    fig_det3d(ep, a.out)
    print('saved', sorted(x.name for x in a.out.iterdir()))


if __name__ == '__main__':
    main()
