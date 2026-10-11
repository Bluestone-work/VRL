"""Create reproducible diagnostics for Scheduled-Settle training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in a.log.read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit('empty training log')
    t = np.array([r['minutes'] for r in rows]); it = np.array([r['it'] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    curves = [('strict', 'Strict success'), ('success', 'Task success'), ('removal', 'Removal fraction'), ('t90', 'T90 (s)')]
    for axis, (key, label) in zip(ax.flat, curves):
        y = np.array([r.get(key, np.nan) for r in rows], float)
        axis.plot(t, y, color='#1769aa', lw=1.3)
        axis.set_xlabel('training minutes'); axis.set_ylabel(label); axis.grid(alpha=.25)
    fig.suptitle('FlowResidual training diagnostics')
    fig.savefig(a.out/'training_curves.png', dpi=220); fig.savefig(a.out/'training_curves.pdf'); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    names = ['speed residual', 'drift coefficient']
    for j, name in enumerate(names):
        ax[j].plot(t, [r.get('a_mean', [np.nan, np.nan])[j] for r in rows], color='#b23a48')
        ax[j].axhline(0, color='k', lw=.6); ax[j].set_title(name); ax[j].set_xlabel('minutes'); ax[j].grid(alpha=.25)
    ax[2].plot(t, [r.get('strict_lat1', np.nan) for r in rows], label='latency 1')
    ax[2].plot(t, [r.get('strict_lat3', np.nan) for r in rows], label='latency 3')
    ax[2].set_ylim(0, 1); ax[2].set_title('latency split'); ax[2].legend(frameon=False); ax[2].grid(alpha=.25)
    fig.savefig(a.out/'action_diagnostics.png', dpi=220); fig.savefig(a.out/'action_diagnostics.pdf'); plt.close(fig)

    (a.out/'latest.json').write_text(json.dumps(rows[-1], indent=2))


if __name__ == '__main__':
    main()
