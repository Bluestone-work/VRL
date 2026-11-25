"""Compare training arms on the frozen manifest: success and loss rate side by side.

The N=3 collapse coincides with the loss rate jumping from 4.4% to 16.7%, so an
arm that only improves success while still losing clusters has not addressed the
mechanism. Both are reported, with the per-seed values, and differences against a
named reference arm are paired over manifest rows.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 42, 43)


def arm_runs(spec):
    """spec = 'label=dir/pattern' where pattern contains {s}."""
    label, _, pat = spec.partition('=')
    files = [ROOT / pat.format(s=s) for s in SEEDS]
    found = [f for f in files if f.exists()]
    if not found:
        return label, None, []
    runs = [json.loads(f.read_text()) for f in found]
    return label, runs, [f.name for f in found]


def summarize(runs):
    o = [r['overall'] for r in runs]
    g = lambda k: (float(np.mean([x[k] for x in o])), float(np.std([x[k] for x in o])),
                   [round(float(x[k]) * 100, 1) for x in o])
    return dict(success=g('navigation_success_rate'), strict=g('strict_success_rate'),
                lost=g('lost_rate'), collision=g('collision_episode_rate'),
                wall=(float(np.mean([x['wall_contact_s'] for x in o])),
                      float(np.std([x['wall_contact_s'] for x in o])), []),
                episodes=sum(x['episodes'] for x in o))


def paired(ref_runs, arm_runs_, key='navigation_success'):
    """Paired difference over scenario hashes present in both arms."""
    def table(runs):
        d = defaultdict(list)
        for r in runs:
            for e in r['episodes']:
                d[e.get('scenario_hash')].append(bool(e[key]))
        return {h: float(np.mean(v)) for h, v in d.items()}
    a, b = table(ref_runs), table(arm_runs_)
    keys = sorted(set(a) & set(b))
    if len(keys) < 10:
        return None
    da = np.array([b[k] - a[k] for k in keys])
    rng = np.random.default_rng(20261030)
    boot = da[rng.integers(0, len(da), size=(10000, len(da)))].mean(1)
    wins = int(sum(1 for k in keys if b[k] > a[k]))
    losses = int(sum(1 for k in keys if b[k] < a[k]))
    n = wins + losses
    kk = min(wins, losses)
    pv = min(1., 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n) if n else 1.
    return dict(n=len(keys), mean=float(da.mean()) * 100,
                lo=float(np.percentile(boot, 2.5)) * 100,
                hi=float(np.percentile(boot, 97.5)) * 100,
                wins=wins, losses=losses, p=pv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True,
                    help="label=path/with/{s}.json, repeatable; first is the reference")
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        label, runs, names = arm_runs(spec)
        if runs is None:
            print(f'{label:<22} 缺(无评估结果)')
            continue
        arms.append((label, runs, summarize(runs)))
    if not arms:
        raise SystemExit('no arm had results')
    print(f'{"arm":<22}{"success":>22}{"lost":>20}{"wall":>12}{"n":>7}')
    print('-' * 83)
    for label, _, s in arms:
        su, lo = s['success'], s['lost']
        print(f'{label:<22}{su[0]*100:>7.1f}±{su[1]*100:<4.1f} {str(su[2]):<9}'
              f'{lo[0]*100:>7.1f}±{lo[1]*100:<4.1f}{s["wall"][0]:>11.2f}{s["episodes"]:>7}')
    ref_label, ref_runs, _ = arms[0]
    print(f'\n相对 {ref_label} 的配对差值(同 scenario_hash):')
    for label, runs, _ in arms[1:]:
        r = paired(ref_runs, runs)
        if r is None:
            print(f'  {label:<20} 配对场景不足')
            continue
        print(f'  {label:<20}{r["mean"]:+6.1f} pp  CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]  '
              f'胜/负 {r["wins"]}/{r["losses"]}  p={r["p"]:.4f}  n={r["n"]}')


if __name__ == '__main__':
    main()
