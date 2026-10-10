"""Publication renderer for the lysis benchmark (PyBullet EGL, visualisation only; physics never calls this).

One scene (anatomy, N, seed, flow, latency, variation) is rolled out once per method with exactly the benchmark
pipeline; the recorded truth trajectories are then rendered with PyBullet's hardware EGL renderer (transparent
vessel lumen built from smooth tube meshes, clots scaled by remaining mass, clusters) and annotated with matplotlib
(trajectories projected with the same camera, scale bar, time, removal, wall contact, legend).
Outputs: <out>/<tag>_grid.png|pdf (methods x time frames), <out>/<tag>_<method>.mp4, <out>/<tag>_curves.png|pdf,
<out>/<tag>_rollouts.json (the recorded metrics, for traceability).
usage: render_lysis_paper.py --out DIR --anatomy mca_m1_lvo --clusters 3 --seed 2600000003 --flow 0.05 --latency 3
       --methods adaptive_settle,settle,sched:research/runs/.../policy.pt --times 20,60,120,end
"""
from __future__ import annotations

import argparse
import json
import os
import pkgutil
from pathlib import Path

import numpy as np

LABEL = {'nav_off': 'nav_tf_v3 learner-off', 'sel_learned': 'Ours (prediction-assisted)', 'sel_learned_fb': 'Ours (+ fallback)',
         'sel_learned_inv': 'Ours (belief inversion)', 'sel_learned_fb_inv': 'Ours (belief inversion + fallback)',
         'sel_online': 'Online estimator + selector', 'switch_settle': 'SwitchSettle', 'fixed_settle': 'Fixed Settle',
         'adaptive_settle': 'Adaptive Settle', 'settle': 'Fixed Settle', 'no_settle': 'Pursuit + wall guard',
         'switch': 'Frame-age switch rule', 'sched': 'Ours (Scheduled-Settle RL)', 'nav': 'Route-frame PPO',
         'pac_nmpc': 'PAC-NMPC', 'stpg': 'STPG'}
CLUSTER_RGB = [(0.12, 0.47, 0.71), (0.17, 0.63, 0.17), (1.0, 0.5, 0.05)]


def controller(name, ep):
    from scripts.benchmark_lysis import AdaptiveSettleGuard, SettleGuard, SwitchSettle, WallGuard
    kind, _, ckpt = name.partition(':')
    if kind == 'adaptive_settle':
        return AdaptiveSettleGuard(ep)
    if kind == 'settle':
        return SettleGuard(ep)
    if kind == 'no_settle':
        return WallGuard(ep)
    if kind == 'switch':
        return SwitchSettle(ep, .15)
    if kind == 'sched':
        from scripts.train_sched_settle import SchedController
        return SchedController(ep, ckpt)
    if kind == 'nav':
        from scripts.train_lysis_nav import NavController
        return NavController(ep, ckpt)
    if kind == 'pac_nmpc':
        from marl.lysis_baselines import PacNMPC
        return PacNMPC(ep)
    from scripts.eval_exp0090 import controller as c90           # EXP0090 arms: nav_off=, sel_learned[_fb][_inv], ...
    return c90(name, ep, OPTS)


OPTS = {}


def rollout(name, a):
    from marl.deployable_sensing import DeployableConfig
    from scripts.benchmark_lysis import LysisEpisode
    kind = name.partition(':')[0].partition('=')[0]
    ep = LysisEpisode(a.clusters, a.anatomy, a.seed, sense_cfg=DeployableConfig(latency_steps=a.latency),
                      variation=a.variation if a.variation > 0 else None, flow_inlet_mm_s=a.flow,
                      tpg=kind != 'pac_nmpc')
    ctl = controller(name, ep); env = ep.env; n = ep.n
    rec = dict(t=[], pos=[], mass=[], wall=[], removal=[])
    while True:
        est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
        done, out = ep.step(est, ctl(ep, est, tgt, rule, hold), hold)
        rec['t'].append(float(env.elapsed_s)); rec['pos'].append(env.positions_mm[:n].astype(float).tolist())
        rec['mass'].append(env.masses.astype(float).tolist()); rec['wall'].append(float(np.sum(out['wall'])))
        rec['removal'].append(1-float(env.masses.sum())/ep.initial)
        if done:
            break
    row = ep.row(kind)
    geo = dict(points=env.transport.points.astype(float).tolist(), ends=np.asarray(env.transport.ends).tolist(),
               radius=np.asarray(env.flow_model.healthy_radius_mm, float).tolist(),
               clots=env.clot_positions_mm.astype(float).tolist(), clot_mass0=env.initial_mass.astype(float).tolist(),
               body=float(env.config.robot_radius_mm))
    ep.close()
    keep = ('task_success', 'strict_success', 'removal', 't50_s', 't90_s', 't100_s', 'wall_contact_s', 'removal_auc',
            'spacing_violation_pair_s')
    return rec, geo, {k: row[k] for k in keep}


# ---------------------------------------------------------------------------------------------------- scene
def tube_mesh(p0, p1, r0, r1, sides=24, rings=2):
    d = p1-p0; L = float(np.linalg.norm(d)); t = d/max(L, 1e-9)
    h = np.array([1., 0, 0]) if abs(t[0]) < .9 else np.array([0, 1., 0])
    u = np.cross(t, h); u /= np.linalg.norm(u); v = np.cross(t, u)
    ang = np.linspace(0, 2*np.pi, sides, endpoint=False)
    V, I = [], []
    for k in range(rings):
        f = k/(rings-1); c = p0+f*d; r = r0+f*(r1-r0)
        for g in ang:
            V.append((c+r*(np.cos(g)*u+np.sin(g)*v)).tolist())
    for k in range(rings-1):
        for j in range(sides):
            a, b = k*sides+j, k*sides+(j+1) % sides; c, e = a+sides, b+sides
            I += [a, c, b, b, c, e, a, b, c, b, e, c]           # both windings: visible from inside and outside
    return V, I


class Scene:
    def __init__(self, geo, width, height, vessel_alpha=.22):
        import pybullet as p
        self.p = p; self.W, self.H = width, height
        self.c = p.connect(p.DIRECT)
        # Hardware EGL is fast, but on headless machines its plugin can abort
        # the interpreter.  Allow the detached automation to select the robust
        # TinyRenderer path with LYSIS_DISABLE_EGL=1.
        egl = None if os.environ.get('LYSIS_DISABLE_EGL', '').lower() in ('1', 'true', 'yes') else pkgutil.get_loader('eglRenderer')
        self.hw = False
        if egl is not None:
            self.hw = p.loadPlugin(egl.get_filename(), '_eglRendererPlugin', physicsClientId=self.c) >= 0
        P = np.asarray(geo['points']); R = np.asarray(geo['radius'])
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix='lysis_mesh_'); obj = Path(self.tmp)/'vessel.obj'
        lines, off = [], 0
        for a_, b_ in geo['ends']:                               # one OBJ with every tube segment
            V, I = tube_mesh(P[a_], P[b_], R[a_], R[b_])
            lines += [f'v {x:.5f} {y:.5f} {z:.5f}' for x, y, z in V]
            lines += [f'f {I[j]+1+off} {I[j+1]+1+off} {I[j+2]+1+off}' for j in range(0, len(I), 3)]
            off += len(V)
        obj.write_text('\n'.join(lines)+'\n')
        s = p.createVisualShape(p.GEOM_MESH, fileName=str(obj), rgbaColor=[.86, .42, .42, vessel_alpha],
                                specularColor=[.2, .2, .2], physicsClientId=self.c)
        p.createMultiBody(0, -1, s, [0, 0, 0], physicsClientId=self.c)
        deg = np.bincount(np.asarray(geo['ends']).ravel(), minlength=len(P))
        for k in np.flatnonzero(deg >= 2):                      # smooth joints between consecutive segments
            s = p.createVisualShape(p.GEOM_SPHERE, radius=float(R[k]), rgbaColor=[.86, .42, .42, vessel_alpha*.6],
                                    physicsClientId=self.c)
            p.createMultiBody(0, -1, s, P[k].tolist(), physicsClientId=self.c)
        self.clots, self.r_clot = [], []
        for q in geo['clots']:
            k = int(np.argmin(np.linalg.norm(P-np.asarray(q), axis=1))); r = .8*float(R[k])
            self.r_clot.append(r)
            s = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[.45, .02, .05, 1.], physicsClientId=self.c)
            self.clots.append((s, p.createMultiBody(0, -1, s, list(q), physicsClientId=self.c), np.asarray(q)))
        self.geo = geo; self.bodies = []
        self.vis_r = max(4*geo['body'], .012*float(np.ptp(P, axis=0).max()))

    def clusters(self, n):
        p = self.p
        for i in range(n):
            col = list(CLUSTER_RGB[i % 3])+[1.]
            s = p.createVisualShape(p.GEOM_SPHERE, radius=self.vis_r, rgbaColor=col, physicsClientId=self.c)
            self.bodies.append(p.createMultiBody(0, -1, s, [0, 0, 0], physicsClientId=self.c))

    def camera(self, focus_pts, yaw=None, pitch=-35., fov=35.):
        p = self.p; F = np.asarray(focus_pts); c = (F.min(0)+F.max(0))/2; ext = float(np.ptp(F, axis=0).max())
        if yaw is None:
            X = F-F.mean(0); w, V = np.linalg.eigh(X.T@X); n = V[:, 0]   # view along the least-spread axis
            yaw = float(np.degrees(np.arctan2(n[1], n[0])))+90.
        self.view = p.computeViewMatrixFromYawPitchRoll(c.tolist(), .62*ext/np.tan(np.radians(fov/2))+ext*.05,
                                                        yaw, pitch, 0, 2)
        self.proj = p.computeProjectionMatrixFOV(fov, self.W/self.H, .01, 50*ext+10)
        self.M = np.asarray(self.proj).reshape(4, 4).T@np.asarray(self.view).reshape(4, 4).T

    def project(self, X):
        X = np.asarray(X, float).reshape(-1, 3); h = np.c_[X, np.ones(len(X))]@self.M.T
        ndc = h[:, :3]/h[:, 3:4]
        return np.c_[(ndc[:, 0]+1)/2*self.W, (1-ndc[:, 1])/2*self.H]

    def render(self, pos, mass):
        p = self.p
        for b, x in zip(self.bodies, pos):
            p.resetBasePositionAndOrientation(b, list(x), [0, 0, 0, 1], physicsClientId=self.c)
        for (s, b, q), m, m0, r in zip(self.clots, mass, self.geo['clot_mass0'], self.r_clot):
            f = (max(m, 0.)/max(m0, 1e-12))**(1/3)
            p.resetBasePositionAndOrientation(b, (q if f > .02 else q+1e4).tolist(), [0, 0, 0, 1], physicsClientId=self.c)
            p.changeVisualShape(b, -1, rgbaColor=[.45+.4*(1-f), .02, .05, 1.], physicsClientId=self.c)
        img = p.getCameraImage(self.W, self.H, self.view, self.proj, shadow=0, lightDirection=[.4, -.6, 1.],
                               renderer=p.ER_BULLET_HARDWARE_OPENGL if self.hw else p.ER_TINY_RENDERER,
                               physicsClientId=self.c)
        rgb = np.asarray(img[2], np.uint8).reshape(self.H, self.W, 4)[..., :3].copy()
        seg = np.asarray(img[4]).reshape(self.H, self.W)
        rgb[seg < 0] = 255                                       # white background
        return rgb

    def close(self):
        self.p.disconnect(self.c)


# ---------------------------------------------------------------------------------------------------- figures
def annotate(ax, scene, rec, k, title=None, scale_mm=5.):
    import matplotlib.patheffects as pe
    P = np.asarray(rec['pos'][:k+1])                            # [T, n, 3]
    for i in range(P.shape[1]):
        uv = scene.project(P[:, i])
        ax.plot(uv[:, 0], uv[:, 1], '-', color=CLUSTER_RGB[i % 3], lw=1.2, alpha=.9)
    c = np.asarray(scene.geo['points']).mean(0)
    a, b = scene.project([c, c+np.array([scale_mm, 0, 0])])
    L = float(np.linalg.norm(b-a)); x0, y0 = scene.W*.05, scene.H*.93
    ax.plot([x0, x0+L], [y0, y0], 'k-', lw=3); ax.text(x0+L/2, y0-scene.H*.02, f'{scale_mm:g} mm', ha='center', va='bottom', fontsize=8)
    txt = f"t = {rec['t'][k]:.0f} s   removal {100*rec['removal'][k]:.0f} %   wall {sum(rec['wall'][:k+1]):.1f} robot-s"
    ax.text(scene.W*.02, scene.H*.04, txt, fontsize=8, va='top',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def frame_index(rec, t):
    T = np.asarray(rec['t'])
    return len(T)-1 if t == 'end' else int(np.clip(np.searchsorted(T, float(t)), 0, len(T)-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--anatomy', default='mca_m1_lvo')
    ap.add_argument('--clusters', type=int, default=3); ap.add_argument('--seed', type=int, default=2600000003)
    ap.add_argument('--flow', type=float, default=.05); ap.add_argument('--latency', type=int, default=3)
    ap.add_argument('--variation', type=float, default=0.)
    ap.add_argument('--methods', default='adaptive_settle,settle'); ap.add_argument('--times', default='20,60,120,end')
    ap.add_argument('--width', type=int, default=960); ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--yaw', type=float); ap.add_argument('--pitch', type=float, default=-35.)
    ap.add_argument('--video', action='store_true'); ap.add_argument('--tag')
    ap.add_argument('--predictors'); ap.add_argument('--fallback-std', type=float); ap.add_argument('--weights', default='{}')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    OPTS.update(predictors=a.predictors, fallback_std=a.fallback_std, weights=json.loads(a.weights))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    methods = a.methods.split(','); tag = a.tag or f'{a.anatomy}_N{a.clusters}_s{a.seed}_f{a.flow:g}_L{a.latency}_v{a.variation:g}'
    runs = {}
    for m in methods:
        rec, geo, row = rollout(m, a); runs[m] = (rec, row); print(m, row, flush=True)
    focus = np.vstack([np.asarray(geo['clots'])]+[np.asarray(r[0]['pos']).reshape(-1, 3) for r in runs.values()])
    scene = Scene(geo, a.width, a.height); scene.clusters(a.clusters); scene.camera(focus, a.yaw, a.pitch)
    times = a.times.split(',')
    fig, axes = plt.subplots(len(methods), len(times), figsize=(3.2*len(times), 2.5*len(methods)+.4), squeeze=False)
    for r_, m in enumerate(methods):
        rec, row = runs[m]; kind = m.partition(':')[0].partition('=')[0]
        for c_, t in enumerate(times):
            k = frame_index(rec, t); ax = axes[r_, c_]
            ax.imshow(scene.render(rec['pos'][k], rec['mass'][k]))
            annotate(ax, scene, rec, k, title=(LABEL.get(kind, kind) if c_ == 0 else None))
        status = 'Strict success' if row['strict_success'] else ('cleared' if row['task_success'] else 'not cleared')
        axes[r_, -1].text(1.02, .5, status, transform=axes[r_, -1].transAxes, rotation=90, va='center', fontsize=9)
    fig.suptitle(f'{a.anatomy}, N = {a.clusters}, inlet {a.flow:g} mm/s, image latency {a.latency} steps'
                 + (f', dynamics s = {a.variation:g}' if a.variation else ''), fontsize=11)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CLUSTER_RGB[i % 3], marker='o', lw=1.2, label=f'cluster {i+1} (trajectory)')
               for i in range(a.clusters)]
    handles += [Line2D([], [], color=(.45, .02, .05), marker='o', ls='', ms=8, label='thrombus (size ~ remaining mass)'),
                Line2D([], [], color=(.86, .42, .42), lw=6, alpha=.4, label='vessel lumen (pre-operative CTA)')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, 1)); [fig.savefig(a.out/f'{tag}_grid.{e}', dpi=300) for e in ('png', 'pdf')]; plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for m in methods:
        rec, _ = runs[m]; kind = m.partition(':')[0].partition('=')[0]
        ax[0].plot(rec['t'], 100*np.asarray(rec['removal']), label=LABEL.get(kind, kind))
        ax[1].plot(rec['t'], np.cumsum(rec['wall']), label=LABEL.get(kind, kind))
    ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('thrombus removed (%)'); ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('cumulative wall contact (robot-s)'); ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); [fig.savefig(a.out/f'{tag}_curves.{e}', dpi=300) for e in ('png', 'pdf')]; plt.close(fig)
    if a.video:
        import imageio.v2 as imageio
        for m in methods:
            rec, _ = runs[m]; kind = m.partition(':')[0].partition('=')[0]
            w = imageio.get_writer(a.out/f'{tag}_{kind}.mp4', fps=20, codec='libx264', quality=8)
            for k in range(0, len(rec['t']), 3):
                fig, ax = plt.subplots(figsize=(a.width/150, a.height/150), dpi=150)
                ax.imshow(scene.render(rec['pos'][k], rec['mass'][k])); annotate(ax, scene, rec, k, title=LABEL.get(kind, kind))
                fig.tight_layout(pad=.2); fig.canvas.draw()
                w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
            w.close()
    (a.out/f'{tag}_rollouts.json').write_text(json.dumps({m: dict(metrics=r[1], t=r[0]['t'][-1]) for m, r in runs.items()},
                                                         indent=2, default=float))
    scene.close(); print('saved', a.out)


if __name__ == '__main__':
    main()
