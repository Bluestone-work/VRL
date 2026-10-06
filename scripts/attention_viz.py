"""Temporal attention of the T-IRPPO policy, projected onto space (cf. Turbo, Supplementary Eq. S24-S25).

At step t the last token queries the memory window [t-15, t]; the head-averaged weight of each past step is
spread over the obstacles detected at that step (Gaussian footprint of the obstacle radius), giving a map
of which past obstacle observations drove the current correction. Also reports the attention-by-lag
profile averaged over all steps where an obstacle was within 0.5 mm, against steps without.
usage: attention_viz.py --ckpt C --anatomy mca_m1_lvo --clusters 1 --seed 2600100000 --out DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '.')
from scripts.make_manuscript_figures import C  # noqa: E402


def attention(net, seq, mask):
    """Per layer: [B, heads, window] weights of the last query token."""
    x = net.embed(seq); x = torch.where(mask[..., None], net.pad.expand_as(x), x)+net.pos
    out = []
    for layer in net.core.layers:
        h = layer.norm1(x)
        a, w = layer.self_attn(h, h, h, attn_mask=net.causal, need_weights=True, average_attn_weights=False)
        x = x+a; x = x+layer._ff_block(layer.norm2(x))
        out.append(w[:, :, -1, :])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--anatomy', default='mca_m1_lvo')
    ap.add_argument('--clusters', type=int, default=1); ap.add_argument('--seed', type=int, default=2600100000)
    ap.add_argument('--out', type=Path, default=Path('research/figures/V3_20261006'))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from marl.obstacle_control import DRLController
    from scripts.benchmark_obstacles import Episode
    ep = Episode(a.clusters, a.anatomy, a.seed)
    drl = DRLController(ep.env, ep.sensor, ep.ctl, a.ckpt); net = drl.net
    log = []          # (t, cluster pos, est obstacles world list, attn per layer [heads, W], correction norm)
    obs_hist = []
    while True:
        est, tgt, rule, hold = ep.observe()
        local = drl.act(est, rule, hold, tgt)
        seq, mask = torch.as_tensor(drl.hist.seq), torch.as_tensor(drl.hist.mask)
        with torch.no_grad():
            W = [w[0].numpy() for w in attention(net, seq[:1], mask[:1])]
        obs_hist.append([(est.pos[0]+rel, r) for rel, _, r in est.obstacles[0]])
        gap = min([np.linalg.norm(rel)-ep.ctl.body-r for rel, _, r in est.obstacles[0]] or [9.])
        log.append(dict(t=ep.env.elapsed_s, pos=est.pos[0].copy(), W=W, gap=gap,
                        corr=float(np.linalg.norm(local[0]-rule[0])), obs=list(obs_hist[-16:])))
        done, _ = ep.step(est, local, hold)
        if done:
            break
    print('safe', ep.safe(), 'obstacle hits', int(ep.field.events.sum()), 'steps', len(log))
    near = [l for l in log if l['gap'] < .5 and len(l['obs']) == 16]
    far = [l for l in log if l['gap'] > 1.5 and len(l['obs']) == 16]
    lag = np.arange(-15, 1)*.1
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for k, (L, lab, col) in enumerate(((near, '障碍 < 0.5 mm', C['clot']), (far, '障碍 > 1.5 mm', C['priv']))):
        if L:
            for li in range(len(L[0]['W'])):
                prof = np.mean([l['W'][li].mean(0) for l in L], 0)
                axes[0].plot(lag, prof, color=col, lw=1.2, ls='-' if li == len(L[0]['W'])-1 else ':',
                             label=f'{lab} · 第 {li+1} 层 (n={len(L)})')
    axes[0].set_xlabel('记忆中的时间偏移 (s)'); axes[0].set_ylabel('注意力权重'); axes[0].legend(frameon=False, fontsize=5.6)
    axes[0].set_title('按时间偏移的注意力', fontsize=8)
    # strongest-correction step near an obstacle: spatial projection (top view x-y)
    if near:
        l = max(near, key=lambda x: x['corr'])
        w = l['W'][-1].mean(0)
        P = np.array([x['pos'] for x in log]); idx = log.index(l)
        win = P[max(idx-15, 0):idx+1]
        c = l['pos']; span = 1.4
        gx, gy = np.meshgrid(np.linspace(c[0]-span, c[0]+span, 160), np.linspace(c[1]-span, c[1]+span, 160))
        H = np.zeros_like(gx)
        for tau, obs in enumerate(l['obs']):
            for o, r in obs:
                H += w[tau]*np.exp(-((gx-o[0])**2+(gy-o[1])**2)/(2*max(r, .05)**2))
        ax = axes[1]
        ax.imshow(H, extent=[gx.min(), gx.max(), gy.min(), gy.max()], origin='lower', cmap='magma')
        ax.plot(win[:, 0], win[:, 1], color='#4fc3f7', lw=1.2); ax.scatter(*c[:2], s=25, color='#4fc3f7', edgecolor='w', zorder=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f't = {l["t"]:.1f} s：注意力投影到障碍', fontsize=8)
        ax = axes[2]
        im = ax.imshow(l['W'][-1], aspect='auto', cmap='viridis', extent=[-1.5, 0, len(l['W'][-1])-.5, -.5])
        ax.set_xlabel('时间偏移 (s)'); ax.set_ylabel('注意力头'); ax.set_title('最后一层各头注意力', fontsize=8)
        fig.colorbar(im, ax=ax, pad=.02)
    fig.tight_layout(); fig.savefig(a.out/'v3_attention.png', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
