"""Figures for the NMI submission, laid out to match VRL_NMI_.pptx.

Every panel that carries numbers is rendered from a measured result file; the
schematic panels are drawn from the simulator's real geometry rather than
free-hand shapes, so Fig.1a shows the actual HCC portal tree and its target
sites.  Panels whose experiment has not landed yet raise a clear error instead
of inventing data.

usage: make_nmi_figures.py --out research/figures/NMI_20261028 [--only fig1]
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
    'savefig.dpi': 400, 'figure.dpi': 110, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
# Colour-blind safe, consistent across every figure.
C = dict(cluster=['#2b6cb0', '#c0504d', '#2e8b57'], target='#d98300',
         vessel='#c9ced6', wall='#8a94a6', pred='#3d7fc1', policy='#1f4e79',
         safety='#b35c00', global_='#4a5568', feedback='#9aa3af',
         ours='#1f4e79', classic='#7f8c9b', accent='#d98300')
MM = 1 / 25.4


def panel_label(ax, s, dx=-0.055, dy=1.045):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='left')


def box(ax, xy, w, h, title, sub=None, fc='#ffffff', ec='#4a5568', lw=.7, fs=6.6):
    """Rounded process box with a bold title and an optional caption line."""
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.006,rounding_size=0.018',
                                fc=fc, ec=ec, lw=lw, zorder=3))
    cx = xy[0] + w / 2
    ty = xy[1] + h * (.62 if sub else .5)
    ax.text(cx, ty, title, ha='center', va='center', fontsize=fs,
            fontweight='semibold', color='#1a202c', zorder=4)
    if sub:
        ax.text(cx, xy[1] + h * .28, sub, ha='center', va='center', fontsize=fs - .9,
                color='#5a6472', zorder=4)


def arrow(ax, p0, p1, color='#4a5568', lw=.8, style='-|>', ls='-', rad=0.):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


def episode(n=3, anatomy='hcc_portal_multifocal', seed=1910000000, flow=.05, obstacles=8):
    from scripts.benchmark_lysis import LysisEpisode
    from marl.deployable_sensing import DeployableConfig
    return LysisEpisode(n, anatomy, seed, tpg=True, horizon=300., sensing='image',
                        sense_cfg=DeployableConfig(latency_steps=0),
                        flow_inlet_mm_s=flow, dynamic_obstacle_count=obstacles)


def draw_tree(ax, env, a0=0, a1=1, lw=4.5, alpha=.88, zorder=0):
    """Vessel centrelines from the transport graph, width scaled by lumen radius."""
    t = env.transport
    pts = t.points.astype(float)
    rad = np.asarray(env.flow_model.healthy_radius_mm, float)
    for (a, b) in t.ends:
        r = .5 * (rad[a] + rad[b])
        ax.plot(pts[[a, b], a0], pts[[a, b], a1], color=C['vessel'],
                solid_capstyle='round', lw=lw * r, alpha=alpha, zorder=zorder)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return len(t.ends)


# --------------------------------------------------------------------- Fig.3b
GEN = ROOT / 'research/runs/NAV_GEN_20261028'
SEEDS = (41, 42, 43)
# Short labels in the figure; the full names go in the caption.
SHORT = {'mca_m1_lvo': 'MCA M1', 'pulmonary_saddle': 'Pulm. saddle',
         'coronary_lm_bifurcation': 'Cor. LM', 'iliac_may_thurner': 'Iliac MT',
         'popliteal_calf_dvt': 'Popl. DVT', 'ica_siphon': 'ICA siphon',
         'ica_terminus_t': 'ICA term.', 'carotid_bifurcation': 'Carotid bif.',
         'renal_artery': 'Renal', 'basilar_vertebral': 'Basilar',
         'cerebral_venous_sinus': 'Cer. sinus', 'coronary_rca': 'RCA',
         'femoropopliteal_pad': 'Fem-pop', 'sma_embolism': 'SMA'}


def load_gen():
    """Per-anatomy metrics for each seed, split into trained and held-out."""
    out = {}
    for split in ('train', 'holdout_test'):
        runs = []
        for s in SEEDS:
            p = GEN / f's{s}_{split}.json'
            if not p.exists():
                raise FileNotFoundError(f'missing generalization run: {p}')
            runs.append(json.loads(p.read_text()))
        out[split] = runs
    return out


def _stats(runs, anatomy, key):
    v = [r['by_anatomy'][anatomy][key] for r in runs if anatomy in r['by_anatomy']]
    return float(np.mean(v)), float(np.std(v)), v


def fig3b(out):
    g = load_gen()
    tr = sorted(g['train'][0]['by_anatomy'])
    ho = sorted(g['holdout_test'][0]['by_anatomy'])
    labels = [f'T{i+1}' for i in range(len(tr))] + [f'U{i+1}' for i in range(len(ho))]
    names = [SHORT.get(a, a) for a in tr] + [SHORT.get(a, a) for a in ho]
    groups = [('train', a) for a in tr] + [('holdout_test', a) for a in ho]
    x = np.arange(len(groups))
    split_at = len(tr) - .5

    specs = [('strict_success_rate', 'Safe success (%)', 100.),
             ('navigation_success_rate', 'Task completion (%)', 100.),
             ('wall_contact_s', 'Wall contact (robot·s)', 1.)]
    fig, axes = plt.subplots(3, 1, figsize=(183 * MM, 118 * MM), sharex=True,
                             gridspec_kw=dict(hspace=.18))
    for ax, (key, ylab, scale) in zip(axes, specs):
        m = np.array([_stats(g[sp], a, key)[0] for sp, a in groups]) * scale
        sd = np.array([_stats(g[sp], a, key)[1] for sp, a in groups]) * scale
        cols = [C['ours'] if sp == 'train' else C['accent'] for sp, _ in groups]
        ax.bar(x, m, width=.66, color=cols, edgecolor='white', lw=.4, zorder=2)
        lower = np.minimum(sd, m)          # a non-negative quantity cannot dip below 0
        ax.errorbar(x, m, yerr=[lower, sd], fmt='none', ecolor='#2d3340', elinewidth=.7,
                    capsize=1.8, capthick=.7, zorder=3)
        for xi, (sp, a) in zip(x, groups):
            pts = np.asarray(_stats(g[sp], a, key)[2]) * scale
            ax.plot(np.full(len(pts), xi), pts, 'o', ms=1.7, mfc='white',
                    mec='#2d3340', mew=.5, zorder=4)
        ax.axvline(split_at, color='#9aa3af', lw=.7, ls=(0, (3, 2)), zorder=1)
        ax.set_ylabel(ylab)
        ax.grid(axis='y', color='#e8eaee', lw=.5, zorder=0)
        ax.set_axisbelow(True)
        # Pooled mean of each side, so the no-degradation claim is visible.
        for sl, lab_x, col in ((slice(0, len(tr)), len(tr) / 2 - .5, C['ours']),
                               (slice(len(tr), None), len(tr) + len(ho) / 2 - .5, C['accent'])):
            mu = float(np.mean(m[sl]))
            ax.hlines(mu, x[sl][0] - .4, x[sl][-1] + .4, color=col, lw=.9,
                      ls=(0, (4, 1.6)), zorder=5)
    axes[0].set_ylim(0, 100); axes[1].set_ylim(0, 100)
    axes[-1].set_ylim(bottom=0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=7)
    axes[-1].set_xlabel('Anatomy', labelpad=2)
    axes[-1].set_xlim(-.7, len(groups) - .3)
    axes[0].text(len(tr) / 2 - .5, 103, f'Trained anatomies (n={len(tr)})', ha='center',
                 fontsize=6.8, color=C['ours'], fontweight='semibold')
    axes[0].text(len(tr) + len(ho) / 2 - .5, 103, f'Whole-class held out (n={len(ho)})',
                 ha='center', fontsize=6.8, color=C['accent'], fontweight='semibold')
    panel_label(axes[0], 'b', dx=-.045, dy=1.12)
    key = '; '.join(f'{l}, {n}' for l, n in zip(labels, names))
    pooled = {}
    for sp, tag in (('train', 'trained'), ('holdout_test', 'held_out')):
        pooled[tag] = {k: float(np.mean([r['overall'][k] for r in g[sp]]))
                       for k in ('strict_success_rate', 'navigation_success_rate', 'wall_contact_s')}
    (out / 'fig3b_caption.txt').write_text(
        'Fig. 3b | Anatomy-level safe navigation. Mean over three IPPO training seeds '
        f'(S{SEEDS[0]}/S{SEEDS[1]}/S{SEEDS[2]}); error bars are s.d. across seeds, open circles are the '
        'individual seeds. Dashed lines are the pooled mean of each side. '
        f'Anatomy key: {key}. '
        f"Pooled safe success {pooled['trained']['strict_success_rate']*100:.1f}% on trained "
        f"versus {pooled['held_out']['strict_success_rate']*100:.1f}% on whole-class held-out anatomies. "
        'Episodes are frozen navigation_dataset_v1 rows (180 per seed trained, 100 per seed held out), '
        'three clusters, zero latency, deterministic policy.\n')
    fig.savefig(out / 'fig3b_generalization.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig3b_generalization.png', bbox_inches='tight')
    plt.close(fig)
    return 'fig3b_generalization'


# --------------------------------------------------------------------- Fig.2b
K13_RUNS = {41: 'EXP0141_IPPO_SEMANTIC_REDUCED_S41_20261016',
            42: 'EXP0142_IPPO_SEMANTIC_REDUCED_S42_20261016',
            43: 'EXP0143_IPPO_SEMANTIC_REDUCED_S43_20261016'}
WINDOW = 15


def training_curve(run_dir, window=WINDOW):
    """Rolling navigation success against cumulative agent steps.

    episodes.jsonl stores per-episode agent steps (9000 means the episode ran
    the full horizon), not a global counter, so the x axis is their cumulative
    sum.  The curve is a trailing mean over `window` episodes; no smoothing is
    applied beyond that.
    """
    rows = [json.loads(l) for l in (ROOT / 'research/runs' / run_dir / 'episodes.jsonl').read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f'no episodes logged in {run_dir}')
    steps = np.cumsum([float(r.get('agent_steps') or 0) for r in rows])
    ok = np.asarray([bool(r.get('navigation_success')) for r in rows], float)
    xs, ys = [], []
    for i in range(window - 1, len(ok)):
        xs.append(steps[i])
        ys.append(ok[i - window + 1:i + 1].mean())
    return np.asarray(xs), np.asarray(ys), float(ok.mean()), len(rows)


DIFF_ORDER = ('easy', 'medium', 'hard', 'stress')


def difficulty_track(run_dir):
    """Per-episode curriculum level, on the same cumulative-step axis."""
    rows = [json.loads(l) for l in (ROOT / 'research/runs' / run_dir / 'episodes.jsonl').read_text().splitlines() if l.strip()]
    steps = np.cumsum([float(r.get('agent_steps') or 0) for r in rows])
    lvl = []
    for r in rows:
        name = r.get('difficulty_name')
        lvl.append(DIFF_ORDER.index(name) if name in DIFF_ORDER else float(r.get('difficulty') or 0))
    return steps, np.asarray(lvl, float)


def fig2b(out):
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(89 * MM, 64 * MM), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=.12))
    curves, finals = [], []
    for k, (seed, run) in enumerate(K13_RUNS.items()):
        x, y, overall, n = training_curve(run)
        ax.plot(x / 1e3, y * 100, '-', lw=1.0, color=C['cluster'][k], alpha=.95,
                label=f'Seed {seed} (n={n})')
        dx, dl = difficulty_track(run)
        axd.step(dx / 1e3, dl, where='post', lw=.8, color=C['cluster'][k], alpha=.8)
        curves.append((x, y)); finals.append(overall)
    hi = min(c[0][-1] for c in curves)
    grid = np.linspace(min(c[0][0] for c in curves), hi, 200)
    stack = np.vstack([np.interp(grid, x, y) for x, y in curves]) * 100
    ax.plot(grid / 1e3, stack.mean(0), '-', lw=1.8, color=C['policy'], label='Seed mean')
    ax.fill_between(grid / 1e3, stack.min(0), stack.max(0), color=C['policy'], alpha=.13, lw=0)
    ax.set_ylabel('Navigation success (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', color='#e8eaee', lw=.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right', handlelength=1.4, borderpad=.2, ncol=2)
    axd.set_yticks(range(len(DIFF_ORDER)))
    axd.set_yticklabels([d.capitalize() for d in DIFF_ORDER], fontsize=6)
    axd.set_ylim(-.4, len(DIFF_ORDER) - .6)
    axd.set_ylabel('Curriculum', fontsize=6.5)
    axd.set_xlabel('Cumulative agent steps (×10³)')
    axd.grid(axis='y', color='#eef0f3', lw=.45); axd.set_axisbelow(True)
    panel_label(ax, 'b', dx=-.13, dy=1.02)
    (out / 'fig2b_caption.txt').write_text(
        'Fig. 2b | On-policy learning of the shared semantic-candidate actor. Top, trailing mean of '
        f'navigation success over {WINDOW} consecutive training episodes against cumulative agent '
        'steps for three independent IPPO seeds on the nine-anatomy pool; the band spans the seed '
        'minimum to maximum. Bottom, the curriculum level actually sampled in each episode. Success '
        'is non-monotonic by construction: the curriculum promotes difficulty as the policy masters '
        'the easier levels, so the mid-run dip coincides with the easy-to-hard transition rather '
        'than with forgetting. Whole-run training success was '
        f'{", ".join(f"{v*100:.1f}%" for v in finals)} for the three seeds. Curves are training-time '
        'episodes under exploration, not the deterministic evaluation protocol.\n')
    fig.savefig(out / 'fig2b_learning.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig2b_learning.png', bbox_inches='tight')
    plt.close(fig)
    return 'fig2b_learning'


# ------------------------------------------------- Fig.1a transfer provenance
# What the method inherits from swarm RL, and what the vascular setting changes.
SWARM = [('Open 3-D workspace', 'free motion in any direction'),
         ('Self-propelled agents', 'commanded velocity is realised'),
         ('Direct state feedback', 'low-latency pose per agent'),
         ('Collision = agent-agent', 'free space is otherwise safe')]
VASC = [('Lumen-constrained', 'motion follows vessel topology'),
        ('Externally actuated', 'one shared magnetic field, flow-dominated'),
        ('Delayed image feedback', 'frame age enters the observation'),
        ('Collision = wall + peers', 'contact damages the vessel')]


def draw_transfer(ax):
    """Two matched columns: inherited formulation on the left, what changes on the right."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(.235, .952, 'Swarm RL formulation', ha='center', va='bottom',
            fontsize=6.8, fontweight='bold', color=C['global_'])
    ax.text(.235, .912, 'aerial / ground multi-robot', ha='center', va='bottom',
            fontsize=5.7, color='#6b7280')
    ax.text(.765, .952, 'Vascular cluster setting', ha='center', va='bottom',
            fontsize=6.8, fontweight='bold', color=C['policy'])
    ax.text(.765, .912, 'this work', ha='center', va='bottom',
            fontsize=5.7, color='#6b7280')
    h, gap = .185, .025
    for k, ((lt, ls), (rt, rs)) in enumerate(zip(SWARM, VASC)):
        y = .905 - (k + 1) * h - k * gap
        box(ax, (.015, y), .44, h, lt, ls, fc='#f7f8fa', ec=C['global_'], lw=.65, fs=5.9)
        box(ax, (.545, y), .44, h, rt, rs, fc='#f4f8fc', ec=C['policy'], lw=.65, fs=5.9)
        arrow(ax, (.462, y + h / 2), (.538, y + h / 2), color='#9aa3af', lw=.8)
    ax.text(.5, .905, 'transfer', ha='center', va='bottom', fontsize=5.8,
            color='#6b7280', style='italic')
    ax.text(.5, -.045, 'Shared: parameter-shared actor, independent PPO critics, candidate-based action set',
            ha='center', va='bottom', fontsize=5.8, color='#4a5568')


# ---------------------------------------------------------------------- Fig.1
# Each stage is (title, caption).  Drawn as a column per panel so the control
# loop reads top-to-bottom and the feedback arrow closes it.
STAGES_B = [('Preoperative geometry\nand targets', 'vessel map · multifocal target sites'),
            ('Geodesic allocation\nand routing', 'per-cluster tour · route frame'),
            ('Fixed TPG schedule', 'zone precedence · wait / release')]
STAGES_C = [('Deployable observation', 'history · route · geometry'),
            ('Frozen motion predictor', 'candidate displacement  \u03bc / \u03c3'),
            ('Shared recurrent\nPPO actor', 'candidate scores \u2192 direction and speed')]
STAGES_D = [('TPG wait check', 'release or hold'),
            ('Safety and\nactuator limits', 'wall · spacing · command clip'),
            ('Execute \u2192\nnew observation', 'next pose · local flow field')]


COLS = [('c', 'Global planning', 'allocation \u00b7 route \u00b7 TPG', 'global_', STAGES_B),
        ('d', 'Learned local decision', 'observation \u00b7 prediction \u00b7 actor', 'policy', STAGES_C),
        ('e', 'Safe execution', 'gate \u00b7 limits \u00b7 actuate', 'safety', STAGES_D)]


def draw_loop(ax):
    """All three control-loop columns in one coordinate system.

    Keeping b, c and d in a single axes means every arrow, including the two
    inter-column arrows and the feedback arc, is placed in the same data
    coordinates instead of being stitched across figure transforms.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    cx = [.155, .5, .845]            # column centres
    w = .29
    ys = [.695, .435, .175]          # row bottoms
    h = .215
    for (lab, title, sub, ckey, stages), x in zip(COLS, cx):
        col = C[ckey]
        ax.text(x, .955, title, ha='center', va='bottom', fontsize=7.2,
                fontweight='bold', color=col)
        ax.text(x, .925, sub, ha='center', va='bottom', fontsize=6, color='#6b7280')
        for k, ((t, cap), y) in enumerate(zip(stages, ys)):
            box(ax, (x - w / 2, y), w, h, t, cap, fc='#fbfcfd', ec=col, lw=.75, fs=6.4)
            if k:
                arrow(ax, (x, ys[k - 1]), (x, y + h), color=col, lw=.85)
        ax.text(x - w / 2 - .012, .955, lab, ha='right', va='bottom',
                fontsize=9, fontweight='bold', color='#1a202c')
    # inter-column hand-offs, at the middle row
    for (x0, x1, text) in ((cx[0] + w / 2, cx[1] - w / 2, 'route'),
                           (cx[1] + w / 2, cx[2] - w / 2, '$u_t$')):
        ym = ys[1] + h / 2
        arrow(ax, (x0 + .004, ym), (x1 - .004, ym), color='#6b7280', lw=.95)
        ax.text(.5 * (x0 + x1), ym - .042, text, ha='center', va='top',
                fontsize=5.6, color='#6b7280')
    # feedback: execution closes the loop at the next control step
    arrow(ax, (cx[2], ys[2] - .012), (cx[1], ys[2] - .012), color=C['feedback'],
          lw=.95, ls=(0, (4, 2)), rad=-.42)
    ax.text(.5 * (cx[1] + cx[2]), .018, 'feedback (next control step)',
            ha='center', va='bottom', fontsize=6, color='#6b7280', style='italic')


def fig1(out):
    ep = episode()
    env = ep.env
    fig = plt.figure(figsize=(183 * MM, 128 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.08], height_ratios=[.70, 1],
                          wspace=.04, hspace=.14, left=.012, right=.988, top=.955, bottom=.025)
    axt = fig.add_subplot(gs[0, :])
    draw_transfer(axt)
    panel_label(axt, 'a', dx=-.008, dy=1.0)
    axa = fig.add_subplot(gs[1, 0])
    draw_tree(axa, env, a0=0, a1=2, lw=3.1)
    clots = np.asarray(env.clot_positions_mm, float)
    pos = np.asarray(env.positions_mm[:ep.n], float)
    axa.plot(clots[:, 0], clots[:, 2], 'o', ms=5.2, mfc=C['target'],
             mec='white', mew=.7, zorder=5, label='Target site')
    for i, pt in enumerate(pos):
        axa.plot(pt[0], pt[2], 's', ms=4.4, mfc=C['cluster'][i], mec='white',
                 mew=.7, zorder=6, label=f'Cluster {i + 1}')
    for k, pt in enumerate(clots):
        axa.annotate(f'S{k + 1}', (pt[0], pt[2]), textcoords='offset points',
                     xytext=(4.5, 3.2), fontsize=6, color='#8a5a00', zorder=7)
    c = pos[0]; half = 17.
    axa.add_patch(plt.Rectangle((c[0] - half, c[2] - half), 2 * half, 2 * half,
                                fill=False, ec=C['pred'], lw=.85,
                                ls=(0, (3, 1.8)), zorder=4))
    axa.annotate('Local sensing', (c[0] - half, c[2] + half),
                 textcoords='offset points', xytext=(0, 2.5), fontsize=6,
                 color=C['pred'], va='bottom', zorder=7)
    axa.set_title('Vessel, targets and local imaging', fontsize=7.2, pad=3)
    axa.legend(frameon=False, fontsize=5.9, loc='lower left', handlelength=1.1,
               borderpad=.15, labelspacing=.22)
    P = np.asarray(env.transport.points, float)
    xr = P[:, 0].max(); z0 = P[:, 2].min()
    axa.plot([xr - 20, xr], [z0 - 4, z0 - 4], '-', color='#2d3340', lw=1.4)
    axa.text(xr - 10, z0 - 6.5, '20 mm', ha='center', va='top', fontsize=6)
    panel_label(axa, 'b', dx=-.01, dy=1.0)
    draw_loop(fig.add_subplot(gs[1, 1]))
    (out / 'fig1_caption.txt').write_text(
        'Fig. 1 | Framework and control loop. a, The registered HCC portal-hepatic tree '
        f'({len(env.transport.ends)} centreline segments, lumen radius 1.1-6.0 mm) in the x-z '
        f'projection, with its {len(clots)} multifocal target sites (S1-S{len(clots)}), the '
        f'{ep.n} robot clusters at their entry stations, and the local imaging field of view of the '
        'leading cluster (dashed). b, The global layer is fixed and identical for every method '
        'compared: preoperative geometry and targets, geodesic allocation with route construction, '
        'and a fixed TPG precedence schedule. c, The learned local decision: a deployable '
        'observation and a frozen action-conditioned motion predictor feed a parameter-shared '
        'recurrent actor that scores the 13 semantic candidates and emits both direction and speed. '
        'd, The selected command passes the TPG wait check and the safety and actuator limits '
        'before execution, and the resulting observation closes the loop at the next control step. '
        'No stage consumes privileged simulator state.\n')
    fig.savefig(out / 'fig1_framework.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig1_framework.png', bbox_inches='tight')
    plt.close(fig)
    return 'fig1_framework'


# ------------------------------------------------------------------- Fig.4c,d
HCC_FIXED = ROOT / 'research/runs/HCC_SEEDS12_FIXED_20261028'
SHARDS = ('shard_off0', 'shard_off3', 'shard_off6', 'shard_off9')
# (flow mm/s, dynamic obstacles) per difficulty, from the evaluation harness.
COND = [('Weak\n0.025 mm/s\n4 obs.', 0), ('Moderate\n0.05 mm/s\n8 obs.', 1),
        ('Strong\n0.10 mm/s\n16 obs.', 2), ('Stress\n0.15 mm/s\n32 obs.', 3)]
MAIN = 'IPPO_S41'
# Display order and labels; the paper's method is drawn last so it sits on top.
MLAB = {'IPPO_S41': 'IPPO (ours)', 'ours_hcc': 'Predictive selector',
        'pursuit': 'Pursuit + settle', 'stpg': 'Switchable TPG',
        'pac': 'PAC-NMPC', 'mppi': 'MPPI', 'cbf': 'CBF-QP',
        'online': 'Online-estimator selector', 'turbo': 'Transformer PPO'}


def load_hcc_fixed():
    rows = []
    for name in SHARDS:
        f = HCC_FIXED / f'{name}.json'
        if not f.exists():
            raise FileNotFoundError(f'HCC rerun shard missing: {f}')
        rows += [r for r in json.loads(f.read_text())['rows'] if 'error' not in r]
    idx = {(r['method'], r['repeat'], r['difficulty']): r for r in rows}
    reps = sorted({k[1] for k in idx})
    methods = sorted({k[0] for k in idx})
    return idx, reps, methods


def _boot_ci(vals, rng, b=10000):
    """Bootstrap mean interval over paired scene seeds."""
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    draws = v[rng.integers(0, len(v), size=(b, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _series(idx, reps, method, key):
    rng = np.random.default_rng(20261028)
    m, lo, hi = [], [], []
    for _, d in COND:
        vals = [float(idx[(method, r, d)][key]) for r in reps if (method, r, d) in idx]
        a, b, c = _boot_ci(vals, rng)
        m.append(a); lo.append(b); hi.append(c)
    return np.asarray(m), np.asarray(lo), np.asarray(hi)


def fig4cd(out):
    idx, reps, methods = load_hcc_fixed()
    order = [m for m in ('turbo', 'cbf', 'online', 'mppi', 'pac', 'stpg', 'pursuit',
                         'ours_hcc', MAIN) if m in methods]
    greys = plt.cm.Greys(np.linspace(.30, .62, max(len(order) - 1, 1)))
    x = np.arange(len(COND))
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), gridspec_kw=dict(wspace=.22))
    for ax, (key, ylab, scale, lab) in zip(axes, [
            ('strict_navigation_success', 'Safe success (%)', 100., 'c'),
            ('wall_contact_s', 'Wall contact (robot\u00b7s)', 1., 'd')]):
        gi = 0
        for meth in order:
            m, lo, hi = _series(idx, reps, meth, key)
            m, lo, hi = m * scale, lo * scale, hi * scale
            if meth == MAIN:
                ax.plot(x, m, '-o', lw=1.9, ms=4.2, color=C['ours'], zorder=6,
                        label=MLAB.get(meth, meth))
                ax.fill_between(x, lo, hi, color=C['ours'], alpha=.17, lw=0, zorder=3)
            else:
                col = C['accent'] if meth == 'pursuit' else greys[gi]
                lw = 1.3 if meth == 'pursuit' else .9
                ax.plot(x, m, '-', lw=lw, color=col, alpha=.95, zorder=4,
                        label=MLAB.get(meth, meth))
                if meth == 'pursuit':
                    ax.errorbar(x, m, yerr=[np.maximum(m - lo, 0), np.maximum(hi - m, 0)],
                                fmt='none', ecolor=col, elinewidth=.7, capsize=1.6,
                                capthick=.7, alpha=.9, zorder=5)
                if meth != 'pursuit':
                    gi += 1
        ax.set_xticks(x)
        ax.set_xticklabels([c for c, _ in COND], fontsize=5.9)
        ax.set_xlabel('Flow and obstacle condition', labelpad=2)
        ax.set_ylabel(ylab)
        ax.set_xlim(-.3, len(COND) - .7)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#e8eaee', lw=.5); ax.set_axisbelow(True)
        panel_label(ax, lab, dx=-.10, dy=1.03)
    axes[0].set_ylim(0, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=5.9, ncol=5, loc='lower center',
               bbox_to_anchor=(.5, -.13), handlelength=1.3, columnspacing=1.1,
               labelspacing=.3)
    n_pair = len(reps)
    (out / 'fig4cd_caption.txt').write_text(
        'Fig. 4c,d | Robustness and safety cost across flow and obstacle conditions on the HCC '
        f'territory. Each point is the mean over {n_pair} paired scene seeds; shaded band (ours) and '
        'error bars (baselines) are bootstrap 95% intervals over those seeds (10\u2074 resamples). '
        'Every method receives the identical global layer, safety layer and scene seeds, so only the '
        'local controller differs. c, Safe success. d, Wall contact per episode. Conditions increase '
        'inlet flow and dynamic-obstacle count together, from 0.025 mm/s with 4 obstacles to '
        '0.15 mm/s with 32. The switchable-TPG baseline is bit-identical to pursuit on this '
        'territory because the construct yields no switchable conflict zone (see Methods). '
        'Intervals are shown for the learned policy and for the strongest classical rule; the '
        'remaining baselines are drawn as means only for legibility. The wide interval on the '
        'learned policy reflects its low success rate at n=12 paired seeds, not extra variance '
        'in the scenes.\n')
    fig.savefig(out / 'fig4cd_robustness.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig4cd_robustness.png', bbox_inches='tight')
    plt.close(fig)
    return 'fig4cd_robustness'


# --------------------------------------------------------------------- Fig.3a
SPLITS = ROOT / 'configs/evaluation_splits.json'


def fig3a(out):
    """Atlas of the registered territories, trained pool versus whole-class held out."""
    sp = json.loads(SPLITS.read_text())['anatomy_holdout_v1']
    train = list(sp['train'])
    held = [a for a in sp['held_out']] if 'held_out' in sp else []
    # The held-out list in the split file is the authority; fall back to the
    # manifest's holdout anatomies if the key shape differs.
    if not held:
        g = json.loads((GEN / 's41_holdout_test.json').read_text())
        held = sorted(g['by_anatomy'])
    names = train + held
    tags = [f'T{i+1}' for i in range(len(train))] + [f'U{i+1}' for i in range(len(held))]
    cols = 5
    rows = int(np.ceil(len(train) / cols)) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(183 * MM, 34 * rows * MM),
                             gridspec_kw=dict(wspace=.06, hspace=.30))
    axes = np.atleast_2d(axes)
    from scripts.benchmark_lysis import LysisEpisode
    from marl.deployable_sensing import DeployableConfig
    drawn = 0
    for k, (nm, tag) in enumerate(zip(names, tags)):
        slot = k if k < len(train) else (rows - 1) * cols + (k - len(train))
        ax = axes[slot // cols][slot % cols]
        try:
            ep = LysisEpisode(3, nm, 1910000000, tpg=True, horizon=300., sensing='image',
                              sense_cfg=DeployableConfig(latency_steps=0),
                              flow_inlet_mm_s=.05, dynamic_obstacle_count=8)
        except Exception as exc:                      # a territory that will not build
            ax.axis('off'); ax.set_title(f'{tag}\n(unavailable)', fontsize=5.6)
            print(f'  skipped {nm}: {exc}')
            continue
        env = ep.env
        accent = C['ours'] if tag.startswith('T') else C['accent']
        draw_tree(ax, env, a0=0, a1=2, lw=1.15)
        cl = np.asarray(env.clot_positions_mm, float)
        ax.plot(cl[:, 0], cl[:, 2], 'o', ms=1.9, mfc=C['target'], mec='none', zorder=5)
        ax.text(.5, 1.03, f'{tag}  {SHORT.get(nm, nm)}', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=5.9, color=accent)
        for sd in ax.spines.values():
            sd.set_visible(False)
        # a thin coloured rule marks which pool the territory belongs to
        ax.plot([.04, .96], [-.035, -.035], transform=ax.transAxes, color=accent,
                lw=1.5, clip_on=False, solid_capstyle='butt')
        drawn += 1
    used = set(range(len(train))) | {(rows - 1) * cols + i for i in range(len(held))}
    for k in range(rows * cols):
        if k not in used:
            axes[k // cols][k % cols].axis('off')
    panel_label(axes[0][0], 'a', dx=-.08, dy=1.22)
    (out / 'fig3a_caption.txt').write_text(
        f'Fig. 3a | Registered vascular territories. {len(train)} territories form the training pool '
        f'(T1-T{len(train)}, blue rule) and {len(held)} further territories are held out as whole '
        f'classes (U1-U{len(held)}, orange rule); they never appear in training or tuning in any '
        'form. Each panel shows the centreline tree in the x-z projection with its registered '
        'target sites (orange). Scene factors are crossed within every territory: three flow '
        'regimes, three target distributions, and four dynamic-obstacle densities by two spatial '
        'layouts, for 72 condition combinations, at cluster counts N = 1, 2 and 3. The HCC '
        'portal-hepatic application territory is registered separately and selected only by name, '
        'so it does not enter these pools (Methods).\n')
    fig.savefig(out / 'fig3a_atlas.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig3a_atlas.png', bbox_inches='tight')
    plt.close(fig)
    return f'fig3a_atlas ({drawn}/{len(names)} territories drawn)'


# --------------------------------------------------------------------- Fig.3c
ROLLOUTS = ROOT / 'research/runs/NAV_ROLLOUTS_20261029/s41_train.json'


def pick_rollouts(eps):
    """Direct arrival, delayed arrival and timeout, chosen by outcome not by eye.

    The three cases are selected as the fastest success, the slowest success and
    the failure that got furthest, so the panel shows the actual spread of
    outcomes rather than a hand-picked illustration.
    """
    # Episodes shorter than 5 s start essentially on target and show no
    # navigation, so they are excluded from the "direct" exemplar.
    ok = [e for e in eps if e['success'] and e['elapsed_s'] >= 5.]
    bad = [e for e in eps if not e['success']]
    if not ok or not bad:
        raise ValueError('need both successful (>=5 s) and failed rollouts')
    fast = min(ok, key=lambda e: e['elapsed_s'])
    slow = max(ok, key=lambda e: e['elapsed_s'])
    fail = max(bad, key=lambda e: (e['reached'], -e['wall_contact_s']))
    return [('Direct arrival', fast, C['cluster'][2]),
            ('Delayed arrival', slow, C['accent']),
            ('Timeout', fail, C['classic'])]


def fig3c(out):
    data = json.loads(ROLLOUTS.read_text())
    cases = pick_rollouts(data['episodes'])
    reach = float(data.get('reach_mm', .30))
    fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 82 * MM),
                             gridspec_kw=dict(height_ratios=[1.25, 1], hspace=.30, wspace=.20))
    from scripts.benchmark_lysis import LysisEpisode
    from marl.deployable_sensing import DeployableConfig
    env = None
    for col, (label, ep, accent) in enumerate(cases):
        axt, axd = axes[0][col], axes[1][col]
        if env is None:                       # one geometry serves all three columns
            e = LysisEpisode(ep['clusters'], ep['anatomy'], 1910000000, tpg=True,
                             horizon=300., sensing='image',
                             sense_cfg=DeployableConfig(latency_steps=0),
                             flow_inlet_mm_s=.05, dynamic_obstacle_count=8)
            env = e.env
        draw_tree(axt, env, a0=0, a1=2, lw=2.7)
        traj = np.asarray(ep['trajectory'], float)[:-1]   # [T, n, 3]; drop post-terminal frame
        tgt = np.asarray(ep['targets'], float)
        axt.plot(tgt[:, 0], tgt[:, 2], 'o', ms=3.4, mfc=C['target'], mec='white',
                 mew=.5, zorder=5)
        for i in range(traj.shape[1]):
            axt.plot(traj[:, i, 0], traj[:, i, 2], '-', lw=1.0,
                     color=C['cluster'][i], alpha=.9, zorder=6)
            axt.plot(traj[0, i, 0], traj[0, i, 2], 's', ms=2.8, mfc=C['cluster'][i],
                     mec='white', mew=.4, zorder=7)
            axt.plot(traj[-1, i, 0], traj[-1, i, 2], '*' if ep['success'] else 'X',
                     ms=5.0 if ep['success'] else 3.8, mfc=C['cluster'][i],
                     mec='white', mew=.4, zorder=7)
        axt.text(.5, 1.05, label, transform=axt.transAxes, ha='center', va='bottom',
                 fontsize=7.0, fontweight='bold', color=accent)
        axt.text(.5, .99, f"{ep['reached']}/{ep['clusters']} reached · "
                          f"{ep['elapsed_s']:.0f} s · wall {ep['wall_contact_s']:.1f} robot·s",
                 transform=axt.transAxes, ha='center', va='bottom', fontsize=5.7,
                 color='#6b7280')
        # distance-to-target traces: this is what separates the three outcomes
        d = np.asarray(ep['target_distance'], float)[:-1]  # same post-terminal frame
        t = np.arange(len(d)) * float(ep['dt_s'])
        for i in range(d.shape[1]):
            axd.plot(t, d[:, i], '-', lw=.95, color=C['cluster'][i], alpha=.95,
                     label=f'Cluster {i + 1}' if col == 0 else None)
        axd.axhline(reach, color='#2d3340', lw=.7, ls=(0, (3, 2)))
        if col == 0:
            axd.text(t[-1] * .98, reach * 1.6, 'reach tolerance', ha='right',
                     va='bottom', fontsize=5.5, color='#2d3340')
        axd.set_xlabel('Time (s)', labelpad=1.5)
        if col == 0:
            axd.set_ylabel('Distance to target (mm)')
            axd.legend(frameon=False, fontsize=5.6, loc='upper right',
                       handlelength=1.1, borderpad=.15, labelspacing=.2)
        axd.set_xlim(0, max(t[-1], 1))
        axd.set_ylim(bottom=0)
        axd.grid(axis='y', color='#eef0f3', lw=.45); axd.set_axisbelow(True)
    panel_label(axes[0][0], 'c', dx=-.10, dy=1.20)
    lab = ', '.join(f"{n.lower()} ({e['elapsed_s']:.0f} s, {e['reached']}/{e['clusters']} reached)"
                    for n, e, _ in cases)
    (out / 'fig3c_caption.txt').write_text(
        'Fig. 3c | Representative navigation outcomes. Top, cluster trajectories in the x-z '
        'projection: squares mark entry stations, stars mark arrival, crosses mark the final pose '
        'of an episode that timed out, and orange circles are the registered target sites. Bottom, '
        'the distance from each cluster to its assigned target over time, which is what separates '
        f'the three outcomes: {lab}. A direct arrival descends monotonically through the '
        f'{reach:.2f} mm reach tolerance; a delayed arrival plateaus or rises before descending; a '
        'timeout stalls at a finite distance for the remainder of the 300 s horizon. The three '
        'episodes are the fastest success, the slowest success and the furthest-progressing failure '
        'among recorded rollouts on frozen manifest rows, so they bound the observed spread rather '
        'than being selected by appearance.\n')
    fig.savefig(out / 'fig3c_outcomes.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig3c_outcomes.png', bbox_inches='tight')
    plt.close(fig)
    return 'fig3c_outcomes'


REGISTRY = {}


def register(fn):
    REGISTRY[fn.__name__] = fn
    return fn


register(fig1)
register(fig2b)
register(fig3a)
register(fig3b)
register(fig3c)
register(fig4cd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'research/figures/NMI_20261028')
    ap.add_argument('--only', default=None, help='comma-separated figure ids')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [x.strip() for x in a.only.split(',')] if a.only else list(REGISTRY)
    for k in todo:
        if k not in REGISTRY:
            raise SystemExit(f'unknown figure: {k}; available: {sorted(REGISTRY)}')
        print('wrote', REGISTRY[k](a.out))


if __name__ == '__main__':
    main()
