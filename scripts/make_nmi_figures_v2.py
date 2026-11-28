"""NMI figure set v2 (2026-11-28), laid out after ~/桌面/VRL_NMI_图片布置.pptx.

Every data panel is computed from a result file under research/; the file list is
written to DATA_SOURCES.md next to the figures.  Panels whose experiment does not
exist yet are drawn as labelled placeholders, never with invented numbers.

usage: make_nmi_figures_v2.py --out ~/桌面/VRL_NMI_figures_20261128 [--only fig3]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / 'research'

# Nature house style: Arial/Helvetica (Liberation Sans is metric-identical), 5-7 pt,
# 180 mm double column, bold lowercase panel letters.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 6.8,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'lines.linewidth': 1.,
    'axes.edgecolor': '#3c4250', 'xtick.color': '#3c4250', 'ytick.color': '#3c4250',
    'axes.labelcolor': '#1d2129', 'text.color': '#1d2129',
    'savefig.dpi': 450, 'figure.dpi': 110, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'legend.frameon': False, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold', 'axes.titlepad': 3,
})
MM = 1 / 25.4
W2 = 180 * MM          # double column
# Categorical palette validated with the dataviz validator (light surface, all pairs):
# CVD dE >= 9.2, normal-vision dE >= 16.3.  Rule-based pursuit is the neutral reference.
M = {'IPPO': '#2a78d6', 'MAPPO-V2': '#4a3aa7', 'STPG': '#1baf7a', 'PAC-NMPC': '#eb6834',
     'Pursuit': '#8b929c'}
MLAB = {'IPPO': 'IPPO (ours)', 'MAPPO-V2': 'MAPPO', 'STPG': 'Switchable TPG',
        'PAC-NMPC': 'PAC-NMPC', 'Pursuit': 'Pure pursuit'}
MMARK = {'IPPO': 'o', 'MAPPO-V2': 's', 'STPG': '^', 'PAC-NMPC': 'D', 'Pursuit': 'v'}
CL = ['#2a78d6', '#eb6834', '#1baf7a']          # cluster 1-3 (same validated hues)
INK, INK2, INK3 = '#1d2129', '#4b5361', '#7a8290'
VESSEL, TARGET, GRID = '#cdd2da', '#c25e00', '#e9ebef'
SAFE, GLOBAL, LEARN = '#c25e00', '#4b5361', '#1f5fae'
SOURCES: dict[str, set] = {}


def src(fig, path):
    SOURCES.setdefault(fig, set()).add(str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path))
    return path


def jload(fig, path):
    return json.loads(Path(src(fig, path)).read_text())


def label(ax, s, x=-0.02, y=1.0, fig=None):
    t = (fig or ax.figure).transFigure if fig is not None else ax.transAxes
    ax.text(x, y, s, transform=t, fontsize=8, fontweight='bold', va='bottom', ha='left', color='#000')


def flabel(fig, s, x, y):
    fig.text(x, y, s, fontsize=8, fontweight='bold', va='top', ha='left', color='#000')


def ftitle(fig, s, x, y, **kw):
    fig.text(x, y, s, fontsize=6.8, va='top', ha='left', color=INK, **kw)


def clean(ax, grid='y'):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.45); ax.set_axisbelow(True)


def box(ax, x, y, w, h, title, sub=None, ec=GLOBAL, fc='#ffffff', fs=6.2, lw=.6, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012',
                                fc=fc, ec=ec, lw=lw, zorder=3, transform=ax.transAxes))
    ax.text(x + w / 2, y + h * (.64 if sub else .5), title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=4, transform=ax.transAxes)
    if sub:
        ax.text(x + w / 2, y + h * .27, sub, ha='center', va='center', fontsize=fs - 1.1,
                color=INK2, zorder=4, transform=ax.transAxes)


def arrow(ax, p0, p1, color=INK2, lw=.7, ls='-', rad=0., ms=6, trans=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1,
                                 zorder=5, transform=trans or ax.transAxes))


def placeholder(ax, text, sub=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('#b9bfc9'); s.set_linestyle((0, (3, 2))); s.set_linewidth(.6)
    ax.set_facecolor('#f6f7f9')
    ax.text(.5, .56 if sub else .5, text, ha='center', va='center', transform=ax.transAxes,
            fontsize=6.2, color=INK2, fontweight='bold')
    if sub:
        ax.text(.5, .38, sub, ha='center', va='center', transform=ax.transAxes, fontsize=5.4, color=INK3)


def save(fig, out, name, caption):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=.02)
    fig.savefig(out / f'{name}.png', bbox_inches='tight', pad_inches=.02, dpi=300)
    (out / f'{name}_caption.txt').write_text(caption.strip() + '\n')
    plt.close(fig)


# ------------------------------------------------------------------ geometry
def tree(ax, env, a0=0, a1=1, lw=3.2, color=VESSEL, alpha=1., z=0):
    t = env.transport
    pts = t.points.astype(float)
    rad = np.asarray(env.flow_model.healthy_radius_mm, float)
    for a, b in t.ends:
        ax.plot(pts[[a, b], a0], pts[[a, b], a1], color=color, solid_capstyle='round',
                lw=lw * .5 * (rad[a] + rad[b]), alpha=alpha, zorder=z)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def lysis_episode(n, anatomy, seed, **kw):
    from scripts.benchmark_lysis import LysisEpisode
    return LysisEpisode(n, anatomy, seed, horizon=300., sensing='image', **kw)


def row_env(row):
    """Rebuild the exact simulator scene of a navigation_dataset_v1 manifest row."""
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    p = CurriculumVascularEnv(seed=int(row['scene_seed']), clusters=3, curriculum=False,
                              reward_mode='navigation', zero_latency=True, direct_action=True,
                              dataset_rows=[row], horizon=300.)
    p.train_anatomies = [row['anatomy']]
    p.reset(options=dict(dataset_row=row))
    return p


def scalebar(ax, length=10, label_=None, loc=(.06, .06)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0); y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color=INK, lw=1.1, solid_capstyle='butt', zorder=9)
    ax.text(x + length / 2, y + .025 * (y1 - y0), label_ or f'{length:g} mm', ha='center', va='bottom',
            fontsize=5.4, color=INK, zorder=9)


# ------------------------------------------------------- paired frozen evaluation
PN = R / 'paired_navigation_corrected'
NEW = R / 'NMI_FIG_BASELINES_20261128'
SEEDS = (41, 42, 43)
SPLITS = {'train': 'train180', 'holdout': 'holdout100'}


def eval_files():
    """method -> split -> list of episode files (one per training seed for learned methods)."""
    f = {'IPPO': {s: [PN / f'mappo_v2_s{k}/ippo_{v}.json' for k in SEEDS] for s, v in SPLITS.items()},
         'MAPPO-V2': {s: [PN / f'mappo_v2_s{k}/paired_{v}.json' for k in SEEDS] for s, v in SPLITS.items()},
         'Pursuit': {s: [R / f'COMPARISON_20261125_REPAIRED/pursuit_{v}.json'] for s, v in SPLITS.items()},
         'STPG': {s: [NEW / f'stpg_{v}.json'] for s, v in SPLITS.items()},
         'PAC-NMPC': {s: [NEW / f'pac_{v}.json'] for s, v in SPLITS.items()}}
    return {m: d for m, d in f.items() if all(p.exists() for ps in d.values() for p in ps)}


def load_eval(fig):
    """Per method and split: array [seed, scene] of outcome dicts keyed by scenario_hash order."""
    files = eval_files()
    out, order = {}, {}
    for m, d in files.items():
        out[m] = {}
        for s, paths in d.items():
            runs = []
            for p in paths:
                eps = jload(fig, p)['episodes']
                runs.append({e['scenario_hash']: e for e in eps})
            if s not in order:
                order[s] = sorted(runs[0])
            for r in runs:
                if sorted(r) != order[s]:
                    raise ValueError(f'{m} {s}: scene set differs from the reference pairing')
            out[m][s] = runs
    return out, order


def metric(e, key):
    if key == 'success':
        return float(bool(e['navigation_success']))
    if key == 'safe':
        return float(bool(e['strict_success']))
    if key == 'wall':
        return float(float(e['wall_contact_s']) > 0)
    if key == 'wall_s':
        return float(e['wall_contact_s'])
    if key == 'lost':
        return float(bool(e['lost']))
    raise KeyError(key)


def scene_values(runs, hashes, key):
    """seed-mean per scene (learned methods have three seeds, classical ones one run)."""
    return np.asarray([np.mean([metric(r[h], key) for r in runs]) for h in hashes], float)


def boot_ci(v, b=4000, seed=0, stat=np.mean):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = stat(v[rng.integers(0, len(v), (b, len(v)))], axis=1)
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_wtl(a, b):
    d = np.asarray(a) - np.asarray(b)
    return int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())


def methods_present(ev):
    return [m for m in ('IPPO', 'MAPPO-V2', 'Pursuit', 'STPG', 'PAC-NMPC') if m in ev]


def legend_methods(ax, ms, loc='upper left', ncol=1, bbox=None, **kw):
    h = [Line2D([], [], marker=MMARK[m], ls='none', mfc=M[m], mec='white', mew=.4, ms=4.2, label=MLAB[m])
         for m in ms]
    return ax.legend(handles=h, loc=loc, ncol=ncol, bbox_to_anchor=bbox, handletextpad=.2,
                     columnspacing=.8, borderaxespad=.2, **kw)


# ===================================================================== Fig. 1
def three_target_trace(fig, n, seed=2944200004, tpg=True):
    tag = 'geo_tpg' if tpg else 'geo_off'
    return jload(fig, NEW / f'{tag}_n{n}_s{seed}_trace.json')


def fig1(out):
    fig = plt.figure(figsize=(W2, 150 * MM))
    # ---- a: multifocal task on a real vessel tree
    from marl.deployable_sensing import DeployableConfig
    ep = lysis_episode(3, 'hcc_portal_multifocal', 1910000000, tpg=True,
                       sense_cfg=DeployableConfig(latency_steps=0), flow_inlet_mm_s=.05,
                       dynamic_obstacle_count=8)
    env = ep.env
    ax = fig.add_axes([.0, .58, .40, .33])
    tree(ax, env, 0, 1, lw=3.6)
    cl = np.asarray(env.clot_positions_mm, float)
    rp = np.asarray(env.positions_mm, float)
    for k, c in enumerate(cl):
        ax.add_patch(Circle((c[0], c[1]), 1.6, fc=TARGET, ec='white', lw=.6, zorder=6))
        ax.text(c[0] + 2.2, c[1] + 1.2, f'L{k+1}', fontsize=5.8, color=TARGET, fontweight='bold', zorder=7)
    for i, p in enumerate(rp[:3]):
        ax.plot(p[0], p[1], 's', ms=4.4, mfc=CL[i], mec='white', mew=.6, zorder=8)
    w = 14.
    c0 = rp[0]
    ax.add_patch(Rectangle((c0[0] - w / 2, c0[1] - w / 2), w, w, fill=False, ec=LEARN, lw=.7,
                           ls=(0, (2.5, 1.5)), zorder=7))
    ax.text(c0[0], c0[1] + w / 2 + 1.2, 'local imaging window', fontsize=5.2, color=LEARN, ha='center', va='bottom')
    ax.autoscale_view()
    scalebar(ax, 20, loc=(.70, .04))
    ax.text(.0, 1.0, 'Preoperative CTA map: vessel graph + lesion sites', transform=ax.transAxes,
            fontsize=6.2, color=INK2, va='bottom')
    hs = [Line2D([], [], marker='o', ls='none', mfc=TARGET, mec='white', ms=5, label='Target lesion'),
          *[Line2D([], [], marker='s', ls='none', mfc=CL[i], mec='white', ms=4.4, label=f'Cluster {i+1}')
            for i in range(3)]]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(-.01, -.02), handletextpad=.2, labelspacing=.25)
    flabel(fig, 'a', .0, .985)
    ftitle(fig, 'Multifocal targeted delivery with magnetic microrobot clusters', .025, .985)

    # ---- b: provenance, hierarchical swarm robotics -> vascular clusters
    axb = fig.add_axes([.43, .55, .57, .41]); axb.axis('off')
    flabel(fig, 'b', .43, .985)
    ftitle(fig, 'Transferring the hierarchical swarm recipe into the vasculature', .455, .985)
    tiers = [('Task allocation', 'targets → robots (Hungarian / auction)',
              'Lesion allocation on the vessel graph', 'min–max geodesic tour, CTA-derived'),
             ('Multi-agent path finding', 'conflict-free routes (CBS / TPG)',
              'Geodesic routes + temporal plan graph', 'precedence at shared segments'),
             ('Decentralized MARL', 'shared policy, local sensing (IPPO)',
              'Shared IPPO local actor', 'image-history feedback, 13 candidates'),
             ('Collision avoidance', 'agent–agent only',
              'Safety layer', 'wall SDF, spacing shield, command clip')]
    y = .86
    axb.text(.20, .975, 'Aerial / ground swarms', ha='center', fontsize=6.2, fontweight='bold', color=INK2,
             transform=axb.transAxes)
    axb.text(.74, .975, 'Vascular clusters (this work)', ha='center', fontsize=6.2, fontweight='bold',
             color=LEARN, transform=axb.transAxes)
    for t0, s0, t1, s1 in tiers:
        box(axb, .00, y - .135, .40, .155, t0, s0, ec='#9aa1ad', fc='#f7f8fa', fs=5.9)
        box(axb, .54, y - .135, .40, .155, t1, s1, ec=LEARN, fc='#f3f7fd', fs=5.9)
        arrow(axb, (.405, y - .058), (.535, y - .058), color='#9aa1ad')
        y -= .19
    axb.text(.47, .06, 'what changes in vivo: lumen-constrained motion · one external field drives all clusters · '
             'flow-dominated, patient-specific dynamics · delayed image feedback · wall contact is harm',
             ha='center', va='center', fontsize=5.2, color=INK3, transform=axb.transAxes, wrap=True)

    # ---- c / d / e: control loop
    top = .47
    flabel(fig, 'c', .0, top + .01); ftitle(fig, 'Global layer: graph planning (before navigation)', .025, top + .01)
    flabel(fig, 'd', .37, top + .01); ftitle(fig, 'Local layer: shared IPPO policy (every 0.1 s)', .395, top + .01)
    flabel(fig, 'e', .70, top + .01); ftitle(fig, 'Safe execution and feedback', .725, top + .01)
    # c: real routes on the MCA tree, with the TPG wait location
    ep2 = lysis_episode(3, 'mca_m1_lvo', 2944200004, shield=True, tpg=False, navigation=True)
    tr = three_target_trace('fig1', 3)
    P = np.asarray([s['positions_mm'] for s in tr['trace']], float)
    hold = np.asarray([s['hold'] for s in tr['trace']], bool)
    axc = fig.add_axes([.0, .19, .33, .26])
    tree(axc, ep2.env, 0, 1, lw=3.0)
    for i in range(3):
        axc.plot(P[:, i, 0], P[:, i, 1], color=CL[i], lw=1.1, zorder=4)
        axc.plot(P[0, i, 0], P[0, i, 1], 's', ms=3.8, mfc=CL[i], mec='white', mew=.5, zorder=6)
    tg = np.asarray(ep2.env.clot_positions_mm, float)[:3]
    axc.plot(tg[:, 0], tg[:, 1], 'o', ms=4.6, mfc=TARGET, mec='white', mew=.5, zorder=7)
    if hold.any():
        k, i = np.argwhere(hold)[0]
        axc.add_patch(Circle(P[k, i, :2], 3.2, fill=False, ec=INK, lw=.6, ls=(0, (2, 1.2)), zorder=8))
        axc.text(P[k, i, 0], P[k, i, 1] - 4.5, 'TPG hold before\nshared segment', fontsize=5, color=INK, zorder=8,
                 ha='center', va='top')
    axc.autoscale_view()
    axl = fig.add_axes([.0, .02, .33, .16]); axl.axis('off')
    steps = [('1  Vessel graph G = (V, E)', 'CTA centrelines, edge weight = length'),
             ('2  Allocation', 'min max tour length over clusters (makespan)'),
             ('3  Routes + temporal plan graph', 'spatial separation + precedence → d ≥ d_min')]
    for j, (t, s) in enumerate(steps):
        axl.text(.02, .90 - j * .33, t, fontsize=5.9, fontweight='bold', color=INK, transform=axl.transAxes, va='top')
        axl.text(.06, .90 - j * .33 - .14, s, fontsize=5.3, color=INK2, transform=axl.transAxes, va='top')
    # d: IPPO
    axd = fig.add_axes([.37, .02, .31, .43]); axd.axis('off')
    obs = ['motion history (24 steps)', 'route / map context', 'local image 4×32×32',
           'obstacle tracks (≤8)', 'neighbour features', 'online flow estimate']
    for j, o in enumerate(obs):
        axd.add_patch(FancyBboxPatch((.0, .86 - j * .085), .40, .065, boxstyle='round,pad=0,rounding_size=0.01',
                                     fc='#f3f7fd', ec=LEARN, lw=.45, transform=axd.transAxes))
        axd.text(.20, .892 - j * .085, o, fontsize=5.2, ha='center', va='center', color=INK, transform=axd.transAxes)
        arrow(axd, (.405, .892 - j * .085), (.50, .66), color='#9fb6d6', lw=.45, ms=4)
    axd.text(.20, .955, 'observation $o_i$ (deployable only)', fontsize=5.5, ha='center', color=INK2, transform=axd.transAxes)
    box(axd, .50, .56, .48, .20, 'Shared actor $\\pi_\\theta(a\\,|\\,o_i)$', 'branch encoders → 128-d trunk\nscores 13 candidate commands',
        ec=LEARN, fc='#eaf2fc', fs=6.0)
    box(axd, .50, .30, .48, .15, 'Masked categorical choice', 'invalid candidates removed', ec=LEARN, fs=5.8)
    arrow(axd, (.74, .555), (.74, .455), color=LEARN)
    box(axd, .02, .06, .40, .15, 'Critic $V_\\phi(o_i)$', 'local, training only', ec='#9aa1ad', fc='#f7f8fa', fs=5.8)
    arrow(axd, (.42, .135), (.60, .295), color='#9aa1ad', ls=(0, (2, 1.5)))
    axd.text(.50, .17, 'PPO update with\nshared parameters θ\n(one policy, N agents)', fontsize=5.1, color=INK3,
             transform=axd.transAxes, va='center')
    # e: safety and plant
    axe = fig.add_axes([.70, .02, .30, .43]); axe.axis('off')
    seq = [('TPG gate', 'release or hold at conflict zone'),
           ('Safety layer', 'wall SDF projection · spacing shield'),
           ('Command limit', '|u| ≤ 1 mm/s, 0.1 s period'),
           ('Magnetic actuation + flow', 'patient-specific gain, lag, pulsatility'),
           ('Imaging', 'pose estimate → next observation')]
    for j, (t, s) in enumerate(seq):
        yy = .80 - j * .175
        box(axe, .06, yy, .88, .13, t, s, ec=SAFE if j < 3 else GLOBAL, fc='#fff8f2' if j < 3 else '#f7f8fa', fs=5.9)
        if j:
            arrow(axe, (.50, yy + .175), (.50, yy + .132), color=INK2)
    arrow(axe, (.335, .30), (.372, .30), color=INK2, trans=fig.transFigure)
    fig.text(.353, .305, 'route,\ntarget', fontsize=4.8, color=INK3, ha='center', va='bottom')
    arrow(axe, (.676, .181), (.717, .385), color=LEARN, trans=fig.transFigure)
    fig.text(.70, .27, '$u_t$', fontsize=6.5, color=LEARN)
    arrow(axe, (.717, .078), (.66, .078), color=INK3, ls=(0, (2.5, 1.5)), trans=fig.transFigure)
    fig.text(.688, .083, 'next $o_i$', fontsize=5.2, color=INK3, ha='center', va='bottom')
    hold_s = float(tr['episode']['waiting_robot_s'])
    save(fig, out, 'fig1_framework', f"""
Fig. 1 | Hierarchical control of magnetic microrobot clusters for multifocal targeted delivery.
a, Task: three clusters (squares) deliver to lesion sites L1-L4 (orange) in a portal-hepatic vessel tree
(simulator geometry; width scaled by lumen radius). The vessel graph and lesion sites are known from
preoperative CTA; flow, actuation response and moving obstacles are not. Each cluster is observed only
through a local imaging window (dashed). b, The control stack follows the hierarchy used for aerial and
ground swarms (allocation, multi-agent path finding, decentralized learned control, collision avoidance)
and replaces each tier with its vascular counterpart. c, Global layer, computed once before navigation:
geodesic distances on the vessel graph, a min-max (makespan) allocation of lesions to clusters, and a
temporal plan graph (TPG) that fixes the order in which clusters pass shared segments. Traces are the
executed paths of the three clusters in an MCA scene (seed 2944200004); the dashed circle marks where
the TPG held one cluster ({hold_s:.1f} robot·s in total). d, Local layer: one actor with shared parameters
is run independently by every cluster on its own deployable observation and selects one of 13 candidate
velocity commands; the critic is local and used only during training (IPPO). e, The TPG gate and a
safety layer (wall signed-distance projection, inter-cluster spacing shield, command limit) act on every
command before actuation; the next image closes the loop.
""")


# ===================================================================== Fig. 2
K13 = {41: 'EXP0141_IPPO_SEMANTIC_REDUCED_S41_20261016', 42: 'EXP0142_IPPO_SEMANTIC_REDUCED_S42_20261016',
       43: 'EXP0143_IPPO_SEMANTIC_REDUCED_S43_20261016'}
MAPPO_LONG = {42: 'mappo_v2_long_500000_s42_serial', 43: 'mappo_v2_long_500000_s43'}


def rolling(fig, run_dir, window=15):
    p = src(fig, R / 'runs' / run_dir / 'episodes.jsonl')
    rows = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    steps = np.cumsum([float(r.get('agent_steps') or 0) for r in rows])
    ok = np.asarray([bool(r.get('navigation_success')) for r in rows], float)
    y = np.convolve(ok, np.ones(window) / window, mode='valid')
    return steps[window - 1:], y, len(rows)


def progress_curve(fig, run_dir):
    import csv
    p = src(fig, PN / run_dir / 'progress.csv')
    xs, ys = [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get('rollout/success_rate') not in (None, ''):
                xs.append(float(r['time/total_timesteps'])); ys.append(float(r['rollout/success_rate']))
    return np.asarray(xs), np.asarray(ys)


def fig2(out):
    fig = plt.figure(figsize=(W2, 118 * MM))
    flabel(fig, 'a', 0, .995); ftitle(fig, 'Policy network and training signal (IPPO, parameters shared by all clusters)', .025, .995)
    ax = fig.add_axes([0, .52, 1, .44]); ax.axis('off')
    br = [('Motion history', 'GRU 11→32'), ('Route / map context', 'MLP 32→32'), ('Local image', 'CNN 4×32×32→32'),
          ('Obstacle tracks', 'set pool 25→32'), ('Neighbours', 'MLP 8→32'), ('Candidate set $U_k$', '13×3→32')]
    for j, (t, s) in enumerate(br):
        y = .84 - j * .13
        box(ax, .00, y, .15, .105, t, s, ec=LEARN, fc='#f3f7fd', fs=5.6)
        arrow(ax, (.152, y + .052), (.215, .47), color='#9fb6d6', lw=.45, ms=4)
    box(ax, .00, .02, .15, .1, 'Online flow estimate', 'least squares, zero-init adapter', ec='#9aa1ad', fc='#f7f8fa', fs=5.4)
    arrow(ax, (.152, .07), (.40, .375), color='#9aa1ad', lw=.45, ms=4)
    box(ax, .215, .38, .13, .18, 'concat 192', 'per-cluster state', ec=LEARN, fs=5.8)
    box(ax, .37, .38, .14, .18, 'Trunk 128', 'LayerNorm · GELU', ec=LEARN, fs=5.8)
    arrow(ax, (.346, .47), (.369, .47), color=LEARN)
    box(ax, .55, .58, .17, .2, 'Actor head', '2×128 → 13 logits\nvalidity mask', ec=LEARN, fc='#eaf2fc', fs=5.9)
    box(ax, .55, .16, .17, .2, 'Value head $V(o_i)$', 'separate 128 branch', ec='#9aa1ad', fc='#f7f8fa', fs=5.9)
    arrow(ax, (.512, .50), (.548, .66), color=LEARN); arrow(ax, (.512, .44), (.548, .27), color='#9aa1ad')
    box(ax, .76, .58, .11, .2, 'Choice $u_t$', 'sample (train)\nargmax (deploy)', ec=LEARN, fs=5.8)
    arrow(ax, (.722, .68), (.758, .68), color=LEARN)
    box(ax, .895, .58, .105, .2, 'Safety layer', '→ actuation', ec=SAFE, fc='#fff8f2', fs=5.8)
    arrow(ax, (.872, .68), (.893, .68), color=SAFE)
    box(ax, .76, .16, .24, .2, 'PPO update (training only)', 'GAE · clipped ratio · reward $r_i$', ec='#9aa1ad', fc='#f7f8fa', fs=5.8)
    arrow(ax, (.722, .26), (.758, .26), color='#9aa1ad', ls=(0, (2, 1.5)))
    arrow(ax, (.83, .37), (.66, .575), color='#9aa1ad', ls=(0, (2, 1.5)), rad=.2)
    ax.text(.88, .055, '$r_i$ = 1.5Δd + 8g − 2W − 8L − 2C − 5E − 0.5Q − 0.02‖Δu‖² − c_t (+20 shared bonus)',
            fontsize=5, color=INK3, ha='center', transform=ax.transAxes)
    ax.text(.64, .93, 'blue: deployed inference path   grey dashed: training only', fontsize=5.2, color=INK3,
            ha='center', transform=ax.transAxes)

    # ---- b: learning
    axb = fig.add_axes([.055, .07, .27, .36])
    flabel(fig, 'b', 0, .47); ftitle(fig, 'Training-time success', .025, .47)
    curves = []
    for k, (seed, run) in enumerate(K13.items()):
        x, y, n = rolling('fig2', run)
        axb.plot(x / 1e5, y * 100, lw=.6, color=M['IPPO'], alpha=.35)
        curves.append((x, y))
    g = np.linspace(max(c[0][0] for c in curves), min(c[0][-1] for c in curves), 200)
    st = np.vstack([np.interp(g, x, y) for x, y in curves]) * 100
    axb.plot(g / 1e5, st.mean(0), color=M['IPPO'], lw=1.4, label='IPPO, 3 seeds')
    for seed, run in MAPPO_LONG.items():
        try:
            x, y = progress_curve('fig2', run)
        except FileNotFoundError:
            continue
        axb.plot(x / 1e5, y * 100, color=M['MAPPO-V2'], lw=.8, alpha=.8,
                 label='MAPPO, 500k run (in progress)' if seed == 42 else None)
    axb.set_xlabel('Agent steps (×10⁵)'); axb.set_ylabel('Navigation success (%)')
    axb.set_ylim(0, 100); clean(axb)
    axb.legend(loc='lower right', handlelength=1.2)

    # ---- c: frozen paired evaluation, all methods
    ev, order = load_eval('fig2')
    ms = methods_present(ev)
    flabel(fig, 'c', .36, .47); ftitle(fig, 'Frozen paired evaluation, N = 3 (same 280 scenes for every method)', .385, .47)
    txt = []
    for j, (sp, nm) in enumerate((('train', 'Training anatomies (180 scenes)'), ('holdout', 'Held-out anatomies (100 scenes)'))):
        a = fig.add_axes([.43 + j * .19, .07, .15, .36])
        ref = scene_values(ev['IPPO'][sp], order[sp], 'success')
        for i, m in enumerate(ms):
            for key, yoff, filled in (('success', 0, True), ('safe', 0, False)):
                v = scene_values(ev[m][sp], order[sp], key)
                mu, lo, hi = boot_ci(v)
                x = i + (-.14 if key == 'success' else .14)
                a.plot([x, x], [lo * 100, hi * 100], color=M[m], lw=.8)
                a.plot(x, mu * 100, MMARK[m], ms=4, mfc=M[m] if filled else 'white', mec=M[m], mew=.7)
            if m != 'IPPO':
                w, t, l = paired_wtl(ref, scene_values(ev[m][sp], order[sp], 'success'))
                txt.append(f'{sp} IPPO vs {MLAB[m]}: {w}/{t}/{l}')
        a.set_xticks(range(len(ms))); a.set_xticklabels([MLAB[m].replace(' (ours)', '') for m in ms], rotation=45,
                                                         ha='right', fontsize=5.4)
        a.set_ylim(40, 100); a.set_title(nm, fontsize=5.8, color=INK2); clean(a)
        if j == 0:
            a.set_ylabel('Success (%)')
            a.legend(handles=[Line2D([], [], marker='o', ls='none', mfc=INK2, mec=INK2, ms=3.6, label='navigation'),
                              Line2D([], [], marker='o', ls='none', mfc='white', mec=INK2, ms=3.6, label='safe (strict)')],
                     loc='lower left', handletextpad=.1)
    a3 = fig.add_axes([.83, .07, .17, .36])
    placeholder(a3, 'Flow-estimate ablation', 'IPPO with / without online\nflow branch, 3 seeds\n(not yet run)')
    flabel(fig, 'd', .815, .47); ftitle(fig, 'Module ablation', .84, .47)
    save(fig, out, 'fig2_network_learning', f"""
Fig. 2 | Local policy and its evaluation. a, Network executed by every cluster on its own observation.
Six branch encoders (32-d each) feed a 128-d LayerNorm-GELU trunk; an online flow estimate (least-squares
identification of commanded gain and drift from the command and image history) enters through a
zero-initialised adapter. The actor scores 13 candidate velocity commands and masks invalid ones; the value
head and the PPO update are used only during training. b, Rolling training-time navigation success (15-episode
window) for three IPPO seeds (thin lines, mean thick) under the difficulty curriculum, and the ongoing 500,000-step
MAPPO runs (centralized critic). Training-time values are not comparable with frozen evaluation. c, Frozen
evaluation on the navigation_dataset_v1 manifest, N = 3: every method runs the identical 180 training-anatomy and
100 held-out-anatomy scenes (matched scenario hashes) with the same global layer and safety layer. Filled markers,
navigation success; open markers, strict (safe) success; bars, 95% bootstrap interval over scenes. Learned
methods are the mean of three training seeds per scene. Paired win/tie/loss of IPPO against each method
(navigation success): {'; '.join(txt)}. d, Placeholder: the flow-branch ablation has not been run.
""")


# ===================================================================== Fig. 3
SHORT = {'mca_m1_lvo': 'MCA M1', 'pulmonary_saddle': 'Pulmonary saddle', 'coronary_lm_bifurcation': 'Coronary LM',
         'iliac_may_thurner': 'Iliac (May–Thurner)', 'popliteal_calf_dvt': 'Popliteal DVT', 'ica_siphon': 'ICA siphon',
         'ica_terminus_t': 'ICA terminus', 'carotid_bifurcation': 'Carotid bif.', 'renal_artery': 'Renal artery',
         'coronary_rca': 'Coronary RCA', 'basilar_vertebral': 'Basilar–vertebral', 'cerebral_venous_sinus': 'Cerebral sinus',
         'sma_embolism': 'SMA', 'femoropopliteal_pad': 'Femoro-popliteal'}
TRAIN_COL, HOLD_COL = '#2a78d6', '#c25e00'


def nav_split():
    d = json.loads(src('fig3', ROOT / 'configs/evaluation_splits.json').read_text())['navigation_dataset_v1']
    return list(d['train_anatomies']), list(d['holdout_test_anatomies'])


def plane(env):
    pts = env.transport.points.astype(float)
    ext = np.ptp(pts, axis=0)
    a0, a1 = np.argsort(ext)[::-1][:2]
    return int(min(a0, a1)), int(max(a0, a1))


METH_KEY = {'pursuit': 'Pursuit', 'IPPO': 'IPPO', 'MAPPO': 'MAPPO-V2', 'MAPPO-V2': 'MAPPO-V2'}
REPLAYS = [(3400000000, 'All succeed (MCA, weak flow)'), (3405052004, 'IPPO completes, rules stall at wall'),
           (3413006012, 'IPPO stalls, rules complete')]


def reached_mask(tr, tol):
    D = np.asarray([s['distance_to_target_mm'] for s in tr], float)
    return D, np.maximum.accumulate(D <= tol, axis=0)


def fig3(out):
    train, hold = nav_split()
    names = train + hold
    tags = [f'T{i+1}' for i in range(len(train))] + [f'U{i+1}' for i in range(len(hold))]
    fig = plt.figure(figsize=(W2, 182 * MM))
    flabel(fig, 'a', 0, .995)
    ftitle(fig, 'Anatomical diversity: 9 training territories (T) and 5 whole-class held-out territories (U)', .025, .997)
    from marl.deployable_sensing import DeployableConfig
    for k, (nm, tag) in enumerate(zip(names, tags)):
        col = k % 7; r = k // 7
        a = fig.add_axes([.005 + col * .142, .815 - r * .135, .13, .115])
        ep = lysis_episode(3, nm, 1910000000, tpg=True, sense_cfg=DeployableConfig(latency_steps=0),
                           flow_inlet_mm_s=.05, dynamic_obstacle_count=8)
        a0, a1 = plane(ep.env)
        tree(a, ep.env, a0, a1, lw=1.5)
        c = np.asarray(ep.env.clot_positions_mm, float)
        a.plot(c[:, a0], c[:, a1], 'o', ms=2.2, mfc=TARGET, mec='white', mew=.3, zorder=5)
        accent = TRAIN_COL if tag[0] == 'T' else HOLD_COL
        a.set_title(f'{tag}  {SHORT.get(nm, nm)}', fontsize=5.5, color=accent, pad=1)
        a.plot([.05, .95], [-.04, -.04], transform=a.transAxes, color=accent, lw=1.2, clip_on=False)
    fig.text(.025, .975, 'Scene factors crossed in every territory: 3 flow regimes × 3 target distributions × '
             '4 obstacle densities × 2 layouts = 72 conditions; N = 1, 2, 3', fontsize=5.6, color=INK3, va='top')

    # ---- b: per-anatomy outcomes
    ev, order = load_eval('fig3')
    ms = methods_present(ev)
    flabel(fig, 'b', 0, .655)
    ftitle(fig, 'Per-territory outcome on the frozen manifest (N = 3; 20 scenes per territory)', .025, .655)
    rows_ = [('success', 'Completion (%)'), ('safe', 'Safe success (%)'), ('wall', 'Wall contact (%)')]
    by = {}
    for sp in ('train', 'holdout'):
        ref = next(iter(ev[ms[0]][sp]))
        for h in order[sp]:
            by.setdefault(ref[h]['anatomy'], (sp, []))[1].append(h)
    for r, (key, yl) in enumerate(rows_):
        a = fig.add_axes([.065, .535 - r * .098, .93, .078])
        for xi, nm in enumerate(names):
            sp, hs = by[nm]
            for i, m in enumerate(ms):
                v = scene_values(ev[m][sp], hs, key) * 100
                a.plot(xi + (i - (len(ms) - 1) / 2) * .13, v.mean(), MMARK[m], ms=3.3, mfc=M[m], mec='white',
                       mew=.3, zorder=3)
        a.axvspan(len(train) - .5, len(names) - .5, color='#fbf3ea', zorder=0, lw=0)
        a.set_xlim(-.6, len(names) - .4); a.set_ylabel(yl, fontsize=5.6)
        a.set_xticks(range(len(names))); a.set_xticklabels(tags if r == 2 else [], fontsize=5.6)
        for t, tl in zip(tags, a.get_xticklabels()):
            tl.set_color(TRAIN_COL if t[0] == 'T' else HOLD_COL)
        a.set_ylim(-5, 105); a.set_yticks([0, 50, 100]); clean(a)
        if r == 0:
            legend_methods(a, ms, loc='lower left', ncol=len(ms), bbox=(0, 1.0))

    # ---- c: representative episodes
    flabel(fig, 'c', 0, .275)
    ftitle(fig, 'Representative episodes: executed IPPO paths and the farthest cluster’s distance to its target', .025, .275)
    notes = []
    for j, (seed, title) in enumerate(REPLAYS):
        d = jload('fig3', R / f'replay_remaining_corrected/scene_{seed}.json')
        recs = {METH_KEY.get(r['method'], r['method']): r for r in d['records']}
        phys = row_env(d['row'])
        env = phys.ep.env
        a0, a1 = plane(env)
        at = fig.add_axes([.005 + j * .335, .075, .14, .15])
        tree(at, env, a0, a1, lw=1.8)
        tol = float(getattr(phys, 'nav_goal_tolerance_mm', .35))
        tr = recs['IPPO']['trace']
        P = np.asarray([s['position_mm_world'] for s in tr], float)
        tg = np.asarray(env.clot_positions_mm, float)[np.asarray(recs['IPPO']['navigation_targets'], int)]
        for i in range(P.shape[1]):
            at.plot(P[:, i, a0], P[:, i, a1], color=CL[i], lw=.8, zorder=4)
            at.plot(P[0, i, a0], P[0, i, a1], 's', ms=2.6, mfc=CL[i], mec='white', mew=.3, zorder=6)
        at.plot(tg[:, a0], tg[:, a1], 'o', ms=3, mfc=TARGET, mec='white', mew=.3, zorder=7)
        at.set_title(title, fontsize=5.6, color=INK, loc='left', pad=1)
        phys.close()
        ad = fig.add_axes([.165 + j * .335, .075, .155, .13])
        for m in ('Pursuit', 'MAPPO-V2', 'IPPO'):
            if m not in recs:
                continue
            t = np.asarray([s['elapsed_s'] for s in recs[m]['trace']], float)
            D, reached = reached_mask(recs[m]['trace'], tol)
            far = np.where(reached, 0., D).max(1)
            ad.plot(t, far, color=M[m], lw=1.1 if m == 'IPPO' else .8, ls='-' if m == 'IPPO' else (0, (3, 1.2)),
                    label=MLAB[m].replace(' (ours)', ''))
            ok = bool(recs[m]['episode_metrics'].get('navigation_success'))
            ad.plot(t[-1], far[-1], '*' if ok else 'X', ms=5 if ok else 4, color=M[m], mec='white', mew=.3, zorder=5)
            notes.append(f'scene {seed} {m}: success={ok}, wall contact '
                         f'{float(recs[m]["episode_metrics"].get("wall_contact_s", 0)):.1f} robot·s')
        ad.set_xlabel('Time (s)', fontsize=5.6); clean(ad)
        if j == 0:
            ad.set_ylabel('Remaining distance (mm)', fontsize=5.6)
            ad.legend(loc='upper right', handlelength=1.4, fontsize=5.2)
    save(fig, out, 'fig3_anatomy_generalization', f"""
Fig. 3 | Navigation across vascular territories. a, The 14 registered territories (centreline trees,
width scaled by lumen radius, orange dots registered target sites; projection onto the two widest axes).
The global and local layers are trained and tuned on T1-T9 only; U1-U5 are held out as whole classes. Within
each territory three flow regimes, three target distributions, four dynamic-obstacle densities and two
layouts are crossed (72 conditions). b, Per-territory task completion (all assigned targets reached),
safe success (completion without lost clusters or contact after arrival) and percentage of scenes with any wall
contact, N = 3, 20 frozen-manifest scenes per territory, identical scenes for every method; learned methods are
averaged over three training seeds. Shaded columns: held-out territories. c, Three replayed scenes with the frozen
seed-41 policies. Left, executed IPPO paths (squares, start; orange, targets). Right, the distance of the
farthest unfinished cluster to its target for IPPO, MAPPO and pure pursuit (star, success; cross, failure at the
deadline or loss). The scenes were selected from early diagnostic replays and illustrate the three outcome types;
the matrix in b, not these examples, measures how often each occurs. {'; '.join(notes)}.
""")


# ===================================================================== Fig. 4
def pooled(ev, order, m, key, sel):
    v = []
    for sp in ('train', 'holdout'):
        ref = ev[m][sp][0]
        hs = [h for h in order[sp] if sel(ref[h])]
        v.extend(scene_values(ev[m][sp], hs, key))
    return np.asarray(v)


def fig4(out):
    from marl.physio_variation import waveform
    fig = plt.figure(figsize=(W2, 128 * MM))
    flabel(fig, 'a', 0, .995); ftitle(fig, 'Three flow regimes on identical scenes (inlet centreline speed)', .025, .995)
    t = np.linspace(0, 4, 800)
    modes = [('Weak steady', .025, 0.), ('Strong steady', .10, 0.), ('Pulsatile', .05, .6)]
    for j, (nm, q0, A) in enumerate(modes):
        a = fig.add_axes([.055 + j * .115, .62, .1, .30])
        for hr, al in ((50, .35), (75, 1.), (120, .35)):
            q = q0 * np.maximum(1 + A * waveform(2 * np.pi * hr / 60 * t), .05) if A else np.full_like(t, q0)
            a.plot(t, q, color=INK if A else INK2, lw=.9 if al == 1 else .6, alpha=al)
            if not A:
                break
        a.set_ylim(0, .14); a.set_title(f'{nm}\n{q0:g} mm/s' + (' nominal' if A else ''), fontsize=5.6, color=INK2)
        a.set_xlabel('Time (s)', fontsize=5.6); clean(a)
        if j == 0:
            a.set_ylabel('Inlet speed (mm/s)', fontsize=5.6)
        else:
            a.set_yticklabels([])
    fig.text(.0, .50, 'Pulsatile: q(t) = q̄·[1 + A·w(2πf t)], A ≤ 0.6,\nheart rate 50–120 bpm (thin: 50 and 120).\n'
             'All regimes add patient-level variation in mean flow,\nactuation gain, direction bias and lag.',
             fontsize=5.1, color=INK3, va='top')

    # ---- b: synchronised keyframes and time series
    d = jload('fig4', R / 'replay_remaining_corrected/scene_3405052004.json')
    recs = {METH_KEY.get(r['method'], r['method']): r for r in d['records']}
    pt = recs['Pursuit']['trace']; it = recs['IPPO']['trace']
    wall_p = np.asarray([s['wall_contact_s_step'] for s in pt], float).sum(0)
    ci = int(np.argmax(wall_p))
    def series(tr, key):
        return np.asarray([np.asarray(s[key], float)[ci] for s in tr])
    ti = np.asarray([s['elapsed_s'] for s in it]); tp = np.asarray([s['elapsed_s'] for s in pt])
    clr_i = series(it, 'estimated_body_clearance_mm_before'); clr_p = series(pt, 'estimated_body_clearance_mm_before')
    spd_i = np.linalg.norm(np.asarray([np.asarray(s['final_sent_command_world'], float)[ci] for s in it]), axis=1)
    spd_p = np.linalg.norm(np.asarray([np.asarray(s['final_sent_command_world'], float)[ci] for s in pt]), axis=1)
    phys = row_env(d['row']); env = phys.ep.env
    reach_i = reached_mask(it, float(getattr(phys, 'nav_goal_tolerance_mm', .35)))[1][:, ci]
    live = ~reach_i
    n_ = min(len(ti), len(tp))
    div = np.where(np.abs(spd_i[:n_] - spd_p[:n_]) > 1e-3)[0]
    k0 = int(div[0]) if len(div) else 0
    wp = np.asarray([np.asarray(s['wall_contact_s_step'], float)[ci] for s in pt])
    kw = np.where((wp > 0) & (tp > ti[k0]))[0]
    t0 = ti[k0]; t1 = tp[kw[0]] if len(kw) else ti[k0] + 5
    t2 = ti[np.argmax(reach_i)] if reach_i.any() else ti[-1]
    k1 = int(np.searchsorted(ti, t1))
    flabel(fig, 'b', .40, .995)
    ftitle(fig, f'Wall-proximity manoeuvre, scene 3405052004, cluster {ci+1}: keyframes and measured traces share one clock', .425, .995)
    a0, a1 = plane(env)
    Pi = np.asarray([np.asarray(s['position_mm_world'], float)[ci] for s in it])
    Pp = np.asarray([np.asarray(s['position_mm_world'], float)[ci] for s in pt])
    for j, (tk, nm) in enumerate(((t0, 't₀ IPPO slows'), (t1, 't₁ pursuit hits wall'), (t2, 't₂ IPPO arrives'))):
        a = fig.add_axes([.42 + j * .195, .72, .17, .23])
        tree(a, env, a0, a1, lw=6)
        ki = int(np.searchsorted(ti, tk)); kp = int(np.searchsorted(tp, tk))
        a.plot(Pp[:kp + 1, a0], Pp[:kp + 1, a1], color=M['Pursuit'], lw=.8, ls=(0, (3, 1.2)))
        a.plot(Pi[:ki + 1, a0], Pi[:ki + 1, a1], color=M['IPPO'], lw=1.0)
        a.plot(Pp[min(kp, len(Pp) - 1), a0], Pp[min(kp, len(Pp) - 1), a1], 'v', ms=4, color=M['Pursuit'], mec='white', mew=.3, zorder=6)
        a.plot(Pi[min(ki, len(Pi) - 1), a0], Pi[min(ki, len(Pi) - 1), a1], 'o', ms=4, color=M['IPPO'], mec='white', mew=.3, zorder=7)
        c = .5 * (Pp[min(kp, len(Pp) - 1)] + Pi[min(ki, len(Pi) - 1)])
        a.set_xlim(c[a0] - 10, c[a0] + 10); a.set_ylim(c[a1] - 10, c[a1] + 10); a.set_aspect('equal', adjustable='box')
        a.set_title(f'{nm}  ({tk:.1f} s)', fontsize=5.6, color=INK2, pad=1)
        if j == 0:
            scalebar(a, 5, loc=(.08, .06))
    phys.close()
    lo, hi = max(t0 - 8, 0), t2 + 10
    for r, (yi, yp, yl) in enumerate(((clr_i, clr_p, 'Est. wall clearance (mm)'), (spd_i, spd_p, 'Command speed (mm/s)'))):
        a = fig.add_axes([.44 + r * .30, .46, .235, .17])
        si = (ti >= lo) & (ti <= hi); sp_ = (tp >= lo) & (tp <= hi)
        a.plot(tp[sp_], yp[sp_], color=M['Pursuit'], lw=.8, ls=(0, (3, 1.2)), label='Pure pursuit')
        a.plot(ti[si], yi[si], color=M['IPPO'], lw=1.0, label='IPPO')
        for tk in (t0, t1, t2):
            a.axvline(tk, color=INK3, lw=.5, ls=(0, (1.5, 1.5)))
        a.set_xlabel('Time (s)', fontsize=5.6); a.set_ylabel(yl, fontsize=5.6); clean(a)
        if r == 0:
            a.axhline(0, color=SAFE, lw=.5); a.legend(loc='upper right', fontsize=5.2)

    # ---- c, d: condition matrix
    ev, order = load_eval('fig4')
    ms = methods_present(ev)
    groups = [('Weak', lambda e: e['flow_mode'] == 'weak'), ('Strong', lambda e: e['flow_mode'] == 'strong'),
              ('Pulsatile', lambda e: e['flow_mode'] == 'pulsatile'),
              ('Low', lambda e: e['obstacle_density'] == 'low'), ('Medium', lambda e: e['obstacle_density'] == 'medium'),
              ('High', lambda e: e['obstacle_density'] == 'high'), ('Stress', lambda e: e['obstacle_density'] == 'stress')]
    flabel(fig, 'c', 0, .37); ftitle(fig, 'Robustness by flow regime and obstacle density (280 paired scenes)', .025, .37)
    flabel(fig, 'd', .52, .37); ftitle(fig, 'Safety cost by condition', .545, .37)
    lines = []
    panels = [(fig.add_axes([.055, .08, .44, .25]), 'success', 'Task completion (%)'),
              (fig.add_axes([.585, .08, .17, .25]), 'wall', 'Scenes with wall contact (%)'),
              (fig.add_axes([.83, .08, .17, .25]), 'lost', 'Scenes with a lost cluster (%)')]
    for a, key, yl in panels:
        for g, (gn, sel) in enumerate(groups):
            for i, m in enumerate(ms):
                v = pooled(ev, order, m, key, sel)
                mu, l, h = boot_ci(v)
                x = g + (i - (len(ms) - 1) / 2) * .14
                a.plot([x, x], [l * 100, h * 100], color=M[m], lw=.6)
                a.plot(x, mu * 100, MMARK[m], ms=3.2, mfc=M[m], mec='white', mew=.3, zorder=3)
                if key == 'success':
                    lines.append(f'{gn} {m} {mu*100:.1f}% (n={len(v)})')
        a.axvline(2.5, color='#c9ced6', lw=.5)
        a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups], fontsize=5.4,
                                                             rotation=0 if key == 'success' else 60)
        a.set_ylabel(yl, fontsize=5.6); clean(a)
        if key == 'success':
            a.set_ylim(0, 105)
            a.text(1, -.17, 'flow regime', transform=a.get_xaxis_transform(), ha='center', fontsize=5.4, color=INK3)
            a.text(4.5, -.17, 'obstacle density', transform=a.get_xaxis_transform(), ha='center', fontsize=5.4, color=INK3)
            legend_methods(a, ms, loc='lower left', ncol=3)
    save(fig, out, 'fig4_flow_robustness', f"""
Fig. 4 | Blood-flow disturbance and dynamic avoidance. a, The three flow regimes paired on identical scenes:
weak steady (0.025 mm/s), strong steady (0.10 mm/s) and pulsatile (0.05 mm/s nominal; two-harmonic arterial
waveform, heart rate 50-120 bpm). Every regime also randomises patient-level mean flow, actuation gain, direction
bias and response lag (Methods), none of which is observable from the preoperative image. b, Scene 3405052004
(pulsatile flow), cluster {ci+1}, frozen seed-41 policies: keyframes when IPPO first deviates from pursuit by slowing (t0), the first
pursuit wall contact (t1) and IPPO arrival (t2), and the measured wall clearance and executed command speed on the same time axis
(dotted lines mark t0-t2). In this scene pure pursuit accumulates {wall_p[ci]:.1f} robot·s of wall contact on this
cluster. c, Task completion by flow regime and obstacle density, pooled over 180 training-anatomy and 100
held-out-anatomy scenes; markers, mean (learned methods: mean of three seeds per scene); bars, 95% bootstrap
interval over scenes. d, Percentage of scenes with any wall contact and with a lost cluster under the same
conditions. Values: {'; '.join(lines)}.
""")


# ===================================================================== Fig. 5
NB = R / 'navigation_budget_20261127'


def rows_of(fig, path, n=None):
    rs = jload(fig, path)['rows']
    return [r for r in rs if n is None or int(r['clusters']) == n]


def fig5(out, seed=2944200004):
    fig = plt.figure(figsize=(W2, 112 * MM))
    flabel(fig, 'a', 0, .995)
    ftitle(fig, f'Same scene, targets and 300 s deadline with N = 1, 2, 3 clusters (MCA, seed {seed})', .025, .995)
    ep = lysis_episode(3, 'mca_m1_lvo', seed, shield=True, tpg=False, navigation=True)
    tg = np.asarray(ep.env.clot_positions_mm, float)[:3]
    for j, n in enumerate((1, 2, 3)):
        tr = three_target_trace('fig5', n, seed)
        P = np.asarray([s['positions_mm'] for s in tr['trace']], float)
        a = fig.add_axes([.0 + j * .19, .57, .18, .38])
        tree(a, ep.env, 0, 1, lw=2.6)
        for i in range(n):
            a.plot(P[:, i, 0], P[:, i, 1], color=CL[i], lw=1.0, zorder=4)
            a.plot(P[0, i, 0], P[0, i, 1], 's', ms=3.6, mfc=CL[i], mec='white', mew=.4, zorder=6)
        for k, c in enumerate(tg):
            a.plot(c[0], c[1], 'o', ms=4.4, mfc=TARGET, mec='white', mew=.4, zorder=7)
            a.text(c[0] + 1.5, c[1] + 1.5, f'L{k+1}', fontsize=5.2, color=TARGET, zorder=8)
        e = tr['episode']
        last = e['deadline_scored_last_arrival_s']
        a.set_title(f'N = {n}   last arrival {float(last):.1f} s', fontsize=5.8, color=INK, pad=1)
        if j == 0:
            a.autoscale_view(); scalebar(a, 10, loc=(.62, .05))

    # ---- b: TPG timeline for N = 3
    tr = three_target_trace('fig5', 3, seed)
    e = tr['episode']
    T = np.asarray([s['time_s'] for s in tr['trace']], float)
    H = np.asarray([s['hold'] for s in tr['trace']], bool)
    Wl = np.asarray([s['wall_s'] for s in tr['trace']], float)
    arr = [np.nanmax([e['first_arrival_s'][j] if e['first_arrival_s'][j] is not None else np.nan for j in tour])
           for tour in e['assignment']]
    flabel(fig, 'b', .58, .995); ftitle(fig, 'Temporal plan graph: release / hold per cluster (N = 3)', .605, .995)
    a = fig.add_axes([.62, .66, .37, .28])
    dt = float(np.median(np.diff(T)))
    for i in range(3):
        for k in range(len(T)):
            if T[k] >= arr[i]:
                break
            a.add_patch(Rectangle((T[k], i - .32), dt, .64, color='#b8bec8' if H[k, i] else CL[i], lw=0))
        a.plot(arr[i], i, '*', ms=6, color=TARGET, mec='white', mew=.3, zorder=5)
        if np.diff(Wl[:, i], prepend=0).max() > 0:
            wk = np.where(np.diff(Wl[:, i], prepend=Wl[0, i]) > 0)[0]
            a.plot(T[wk], np.full(len(wk), i + .42), '|', ms=3, color=SAFE, mew=.6)
    a.set_yticks(range(3)); a.set_yticklabels([f'Cluster {i+1}' for i in range(3)], fontsize=5.6)
    a.set_xlim(0, max(arr) * 1.05); a.set_ylim(-.6, 2.7); a.invert_yaxis()
    a.set_xlabel('Time (s)', fontsize=5.6); clean(a, 'x')
    a.legend(handles=[Rectangle((0, 0), 1, 1, color=CL[0], label='moving'),
                      Rectangle((0, 0), 1, 1, color='#b8bec8', label='TPG hold'),
                      Line2D([], [], marker='*', ls='none', color=TARGET, ms=6, label='arrival'),
                      Line2D([], [], marker='|', ls='none', color=SAFE, ms=4, label='wall contact')],
             loc='upper center', bbox_to_anchor=(.5, -.32), ncol=4, fontsize=5.2, handlelength=1)

    # ---- c: N scaling and allocation
    flabel(fig, 'c', 0, .50); ftitle(fig, 'Benefit of parallel clusters (6 paired MCA scenes)', .025, .48)
    tp = rows_of('fig5', NB / 'geodesic_tpg.json')
    by = {}
    for r in tp:
        by.setdefault(int(r['seed']), {})[int(r['clusters'])] = float(r['deadline_scored_last_arrival_s'])
    a = fig.add_axes([.055, .08, .17, .34])
    for s, d in by.items():
        a.plot([1, 2, 3], [d[1], d[2], d[3]], color='#c3c8d0', lw=.6, marker='o', ms=2.4, mfc='white', mec='#9aa1ad')
    mean = [np.mean([d[n] for d in by.values()]) for n in (1, 2, 3)]
    a.plot([1, 2, 3], mean, color=M['IPPO'], lw=1.3, marker='o', ms=4, mec='white')
    cov = {n: sum(bool(r['all_three_reached']) for r in tp if int(r['clusters']) == n) for n in (1, 2, 3)}
    syn = {n: sum(bool(r['synchronized_success']) for r in tp if int(r['clusters']) == n) for n in (1, 2, 3)}
    for n in (1, 2, 3):
        a.text(n, 1.02, f'{cov[n]}/6', transform=a.get_xaxis_transform(), ha='center', fontsize=5.2, color=INK2)
    a.set_xticks([1, 2, 3]); a.set_xlim(.6, 3.4); a.set_xlabel('Clusters N', fontsize=5.6)
    a.set_ylabel('Last arrival (s, fail = 300)', fontsize=5.6); clean(a)
    import csv
    al = list(csv.DictReader(open(src('fig5', NB / 'allocations.csv'))))
    a2 = fig.add_axes([.29, .08, .17, .34])
    for n in (1, 2, 3):
        rr = [r for r in al if int(r['clusters']) == n]
        g = np.asarray([float(r['geodesic_scored_s']) for r in rr]); l = np.asarray([float(r['linear_sum_scored_s']) for r in rr])
        for gi, li in zip(g, l):
            a2.plot([n - .15, n + .15], [li, gi], color='#c3c8d0', lw=.6)
        a2.plot(np.full(len(l), n - .15), l, 'v', ms=3.2, mfc=M['Pursuit'], mec='white', mew=.3)
        a2.plot(np.full(len(g), n + .15), g, 'o', ms=3.2, mfc=M['IPPO'], mec='white', mew=.3)
    a2.set_xticks([1, 2, 3]); a2.set_xlabel('Clusters N', fontsize=5.6); a2.set_ylabel('Last arrival (s)', fontsize=5.6)
    a2.legend(handles=[Line2D([], [], marker='v', ls='none', mfc=M['Pursuit'], mec='white', ms=4, label='Hungarian (sum)'),
                       Line2D([], [], marker='o', ls='none', mfc=M['IPPO'], mec='white', ms=4, label='Min–max geodesic')],
              loc='upper right', fontsize=5.2); clean(a2)

    # ---- d: coordination ablation
    flabel(fig, 'd', .52, .50); ftitle(fig, 'Coordination layer at N = 3: none, fixed TPG, CCBS search', .545, .48)
    arms = [('No TPG', NB / 'geodesic_off.json'), ('TPG', NB / 'geodesic_tpg.json'), ('CCBS', NB / 'geodesic_ccbs.json')]
    data = {nm: {int(r['seed']): r for r in rows_of('fig5', p, 3)} for nm, p in arms}
    fx = {'No TPG': rows_of('fig5', R / 'COMPARISON_20261125_REPAIRED/fixed_three_tpg_off.json', 3),
          'TPG': rows_of('fig5', R / 'COMPARISON_20261125_REPAIRED/fixed_three_tpg_on.json', 3)}
    metrics = [('deadline_scored_last_arrival_s', 'Last arrival (s)'), ('waiting_robot_s', 'TPG wait (robot·s)'),
               ('spacing_violation_pair_s', 'Spacing violation (pair·s)')]
    summ = []
    for k, (key, yl) in enumerate(metrics):
        a = fig.add_axes([.585 + k * .16, .08, .10 if k < 2 else .12, .34])
        xs = []
        for i, (nm, _) in enumerate(arms):
            v = np.asarray([float(r[key]) for r in data[nm].values()])
            a.plot(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, 'o', ms=2.8, mfc='white', mec='#9aa1ad', mew=.5)
            a.plot([i - .22, i + .22], [v.mean()] * 2, color=INK, lw=1.1)
            summ.append(f'{nm} {yl} mean {v.mean():.2f}')
            xs.append(nm)
        if key == 'spacing_violation_pair_s':
            for i, nm in enumerate(('No TPG', 'TPG')):
                v = np.asarray([float(r[key]) for r in fx[nm]])
                a.plot(np.full(len(v), 3 + i) + np.linspace(-.12, .12, len(v)), v, 'v', ms=2.8, mfc='white', mec=SAFE, mew=.5)
                a.plot([3 + i - .22, 3 + i + .22], [v.mean()] * 2, color=SAFE, lw=1.1)
                summ.append(f'fixed assignment {nm} spacing mean {v.mean():.2f}')
            xs += ['No TPG*', 'TPG*']
        a.set_xticks(range(len(xs))); a.set_xticklabels(xs, rotation=50, ha='right', fontsize=5.3)
        a.set_ylabel(yl, fontsize=5.6); clean(a)
    save(fig, out, 'fig5_cluster_scaling', f"""
Fig. 5 | Cluster number and spatio-temporal coordination. a, One MCA scene (seed {seed}) with the same three
lesions, start pool and 300 s deadline, served by N = 1, 2 or 3 clusters (min-max geodesic allocation, temporal
plan graph, pure-pursuit local control so that only the global layer changes). b, Release/hold state of each cluster
in the N = 3 run: coloured, moving; grey, held by the TPG at a shared segment; star, arrival; orange ticks, steps
with wall contact. c, Left: deadline-scored last-arrival time in six paired scenes (grey, per scene; blue, mean;
numbers, scenes in which all three lesions were reached: N=1 {cov[1]}/6, N=2 {cov[2]}/6, N=3 {cov[3]}/6; all three within
30 s: {syn[1]}/6, {syn[2]}/6, {syn[3]}/6). Mean last arrival {mean[0]:.1f}, {mean[1]:.1f} and {mean[2]:.1f} s. Right: the same
scenes with the min-max geodesic allocation versus a sequential Hungarian (minimum-sum) assignment. d, N = 3 with no
temporal coordination, the fixed TPG, and continuous-time conflict-based search (CCBS, adapted to fixed routes);
circles, geodesic allocation; triangles (*), fixed assignment, for which spacing violations occur. {'; '.join(summ)}.
Six scenes from one anatomy: intervals describe layout-to-layout variation only.
""")


# ===================================================================== Fig. 6
def fig6(out):
    """In-vitro figure: layout only.  No simulation stands in for an experiment."""
    fig = plt.figure(figsize=(W2, 120 * MM))
    flabel(fig, 'a', 0, .995); ftitle(fig, 'Magnetic actuation platform and perfused vessel phantom', .025, .995)
    a = fig.add_axes([.0, .56, .40, .38]); placeholder(a, 'Platform photograph', 'pump · coil array / permanent magnet · camera\nlabel flow direction, channel diameter, scale bar')
    a = fig.add_axes([.42, .56, .18, .38]); placeholder(a, 'Phantom close-up', 'transparent channel, field of view')
    flabel(fig, 'b', .62, .995); ftitle(fig, 'Continuous navigation in real flow', .645, .995)
    for j, t in enumerate(('t₀ start', 't₁ bifurcation', 't₂ arrival')):
        a = fig.add_axes([.62 + j * .128, .56, .12, .38]); placeholder(a, t, 'micrograph\nscale bar')
    flabel(fig, 'c', 0, .48); ftitle(fig, 'Multi-cluster delivery in vitro, N = 1, 2, 3', .025, .48)
    for j in range(3):
        a = fig.add_axes([.0 + j * .125, .06, .115, .38]); placeholder(a, f'N = {j+1}', 'overlay of tracks\n+ lesion marks')
    flabel(fig, 'd', .39, .48); ftitle(fig, 'Single cluster under three flow regimes', .415, .48)
    for j, t in enumerate(('weak steady', 'strong steady', 'pulsatile')):
        a = fig.add_axes([.39 + j * .125, .06, .115, .38]); placeholder(a, t, 'micrograph track')
    flabel(fig, 'e', .78, .48); ftitle(fig, 'Sim-to-real', .805, .48)
    a = fig.add_axes([.80, .10, .19, .32]); placeholder(a, 'Simulation vs in vitro', 'success, arrival time\nreport n and 95% CI')
    save(fig, out, 'fig6_in_vitro_layout', """
Fig. 6 (layout only) | In-vitro validation. All panels are placeholders: these data must come from the
magnetic actuation platform and cannot be substituted with simulation. a, Platform and phantom. b, Keyframes
of one continuous navigation in real flow. c, N = 1, 2, 3 clusters on the same phantom and targets. d, One
cluster under three flow regimes. e, Paired comparison of simulation and in-vitro outcome with n and intervals.
""")


FIGS = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    names = [n for n in a.only.split(',') if n] or list(FIGS)
    for n in names:
        FIGS[n](a.out)
        print('wrote', n, flush=True)
    lines = ['# Data sources (generated by scripts/make_nmi_figures_v2.py)', '']
    for k in sorted(SOURCES):
        lines.append(f'## {k}'); lines += [f'- `{p}`' for p in sorted(SOURCES[k])]; lines.append('')
    (a.out / f'DATA_SOURCES_{"_".join(names)}.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
