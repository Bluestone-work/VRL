"""EXP0091 registered gate (research/validation/EXP0091_20261011/PROTOCOL.md); prints GATE PASS / GATE FAIL."""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

import sys
V0 = 'research/validation/EXP0090_20261011/'
V1 = sys.argv[1] if len(sys.argv) > 1 else 'research/validation/EXP0091_20261011/'
OURS_METHOD = sys.argv[2] if len(sys.argv) > 2 else 'sel_learned_fb'


def arm(r):
    m, a = r['method'], r.get('method_arg') or ''
    if m == OURS_METHOD:
        return 'ours_'+('A' if 'EXP0090' in a else 'B' if 'predictor_B' in a else 'C')
    if m == 'nav':
        return 'nav_tf_v3'
    return m


def load(files, keep=None):
    by = defaultdict(lambda: defaultdict(list))
    for f in files:
        for l in open(f):
            r = json.loads(l); a = arm(r)
            if keep and a not in keep:
                continue
            by[(r['cell'], r['anatomy'], r['clusters'], r['seed'])][a].append(r)
    return by


def boot(d, B=4000):
    d = np.asarray(d, float); rng = np.random.default_rng(0); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    base = {'switch_settle', 'fixed_settle', 'stpg', 'pac_nmpc', 'nav_off', 'nav_tf_v3'}
    hard = load([V0+'hard/episodes.jsonl'], base)
    for k, d in load([V1+'dev_hard_ours/episodes.jsonl', V1+'dev_hard_online/episodes.jsonl']).items():
        hard[k].update(d)
    v5 = load([V0+'v5/episodes.jsonl'], base)
    for k, d in load([V1+'dev_v5_ours/episodes.jsonl', V1+'dev_v5_online/episodes.jsonl', V1+'dev_v5_paper/episodes.jsonl']).items():
        v5[k].update(d)
    ours = ['ours_A', 'ours_B', 'ours_C']
    sv = lambda d, a: np.mean([r['strict_success'] for r in d[a]])
    ok = True; lines = []

    def compare(by, title, cells):
        nonlocal ok
        ks = [k for k, d in by.items() if k[0] in cells or (cells == 'hard' and not k[0].startswith('v5'))]
        ks = [k for k in ks if all(a in by[k] for a in ours+list(base)+['sel_online'])]
        o = np.mean([np.mean([sv(by[k], a) for a in ours]) for k in ks])
        seeds = [np.mean([sv(by[k], a) for k in ks]) for a in ours]
        b = {a: np.mean([sv(by[k], a) for k in ks]) for a in base}
        f = np.mean([sv(by[k], 'sel_online') for k in ks])
        best = max(b, key=b.get); passed = o >= b[best]-1e-12
        ok &= passed
        lines.append(f'{title}: n={len(ks)} ours={100*o:.1f} (seeds {", ".join(f"{100*x:.1f}" for x in seeds)}) '
                     f'best baseline {best}={100*b[best]:.1f} F={100*f:.1f} -> {"ok" if passed else "FAIL"}; '
                     + ' '.join(f'{a}={100*v:.1f}' for a, v in sorted(b.items())))
        return ks

    kh = compare(hard, 'hard matrix ALL', 'hard')
    kv = []
    for s in ('v5_s0', 'v5_s1', 'v5_s1.5'):
        kv += compare(v5, f'V5 {s} (mean over N)', {s})
    d = [np.mean([sv(hard[k], a) for a in ours])-sv(hard[k], 'sel_online') for k in kh]
    d += [np.mean([sv(v5[k], a) for a in ours])-sv(v5[k], 'sel_online') for k in kv]
    m, lo, hi = boot(d); c2 = lo > 0; ok &= c2
    lines.append(f'Ours - F (hard + V5 pooled, n={len(d)}): {100*m:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}] -> {"ok" if c2 else "FAIL"}')
    print('\n'.join(lines)); print('GATE PASS' if ok else 'GATE FAIL')


if __name__ == '__main__':
    main()
