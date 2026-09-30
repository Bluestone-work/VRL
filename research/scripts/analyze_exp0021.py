"""EXP_0021 preregistered analysis: aggregate the fixed-semantics evaluation.

Reads eval_*/episodes.jsonl under the EXP_0021 run directory and produces:
  - per-arm per-checkpoint success mean ± sample SD (3 seeds × 140 episodes)
  - paired per-seed repair−base difference at each checkpoint
  - tail metrics: failed-episode dead time (stall_steps at episode end,
    fraction of failures with stall ≥ K) and eval stall-cut counts
  - per-scenario success table (pooled seeds, best checkpoint)
  - training-side tail-state density (stall_cut share of training episodes)

Writes aggregate.json next to the eval dirs and prints a compact report.
The quarantined first-pass evals (raw-action-semantics bug) are excluded by
construction: they live in eval_invalid_action_semantics_20260928/.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           'research/runs/EXP_0021_TAIL_REPAIR_20260927a')
CKPTS = (1003520, 2007040, 3000032)
SEEDS = (42, 43, 44)
K = 50


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def main():
    evals = defaultdict(dict)  # (arm, ckpt) -> {seed: [episodes]}
    for d in sorted(RUN.glob('eval_*_s*')):
        parts = d.name.split('_')
        # Skip quarantined/other prefixes (e.g. eval_invalid_action_semantics_…)
        if parts[1] not in ('base', 'repair'):
            continue
        arm, seed, ts = parts[1], int(parts[2][1:]), int(parts[3])
        eps = [json.loads(l) for l in (d / 'episodes.jsonl').open()]
        evals[(arm, ts)][seed] = eps

    aggregate = {'run': str(RUN), 'checkpoints': CKPTS, 'arms': {}, 'paired': {},
                 'tail_metrics': {}, 'per_scenario': {}, 'training_tail_density': {}}

    print(f"{'arm':7s} {'ckpt':>8s} {'success mean±SD':>17s} {'removal':>8s} "
          f"{'steps':>6s} {'stall_cuts':>10s}")
    for arm in ('base', 'repair'):
        aggregate['arms'][arm] = {}
        for ts in CKPTS:
            per_seed = evals[(arm, ts)]
            if not per_seed:
                continue
            seed_means = [sum(e['success'] for e in eps) / len(eps)
                          for eps in per_seed.values()]
            m, sd = mean_sd(seed_means)
            all_eps = [e for eps in per_seed.values() for e in eps]
            removal = sum(e['removal'] for e in all_eps) / len(all_eps)
            steps = sum(e['steps'] for e in all_eps) / len(all_eps)
            cuts = sum(int(e['stall_cut']) for e in all_eps)
            row = dict(success_mean=m, success_sd=sd, n_seeds=len(seed_means),
                       episodes=len(all_eps), removal=removal, steps=steps,
                       stall_cuts=cuts)
            aggregate['arms'][arm][ts] = row
            print(f"{arm:7s} {ts:8d} {m*100:6.2f} ± {sd*100:4.2f}    "
                  f"{removal:8.3f} {steps:6.1f} {cuts:10d}")

    # Paired per-seed differences (preregistered primary readout).
    print("\npaired repair − base (per seed, success pp):")
    for ts in CKPTS:
        if ts not in evals.get(('repair', ts), {}) or not evals[('base', ts)]:
            continue
        diffs = []
        for s in SEEDS:
            b = evals[('base', ts)].get(s)
            r = evals[('repair', ts)].get(s)
            if not (b and r):
                continue
            diffs.append(sum(e['success'] for e in r) / len(r)
                         - sum(e['success'] for e in b) / len(b))
        if diffs:
            m, sd = mean_sd(diffs)
            t = m / (sd / math.sqrt(len(diffs))) if sd > 0 and len(diffs) > 1 else None
            aggregate['paired'][ts] = dict(diffs=[d * 100 for d in diffs],
                                           mean_pp=m * 100, sd_pp=sd * 100, t=t)
            print(f"  @{ts}: {m*100:+.2f}pp ± {sd*100:.2f} "
                  f"(per-seed {' '.join(f'{d*100:+.1f}' for d in diffs)}"
                  f"{f', t={t:.2f}' if t is not None else ''})")

    # Tail metrics: failed-episode dead time. The preregistered tail metric is
    # "post-stall remaining steps in failed episodes" — approximated from the
    # end-of-episode stall counter and episode length vs horizon 300.
    print("\ntail metrics (failed episodes):")
    for arm in ('base', 'repair'):
        aggregate['tail_metrics'][arm] = {}
        for ts in CKPTS:
            per_seed = evals.get((arm, ts), {})
            fails = [e for eps in per_seed.values() for e in eps if not e['success']]
            if not fails:
                continue
            stall_end = [e['stall_steps'] for e in fails]
            # dead steps after the last progress: stall counter at end, but the
            # episode may also have been cut at K (repair) — report both raw.
            mean_stall = sum(stall_end) / len(stall_end)
            frac_ge_k = sum(1 for x in stall_end if x >= K) / len(stall_end)
            row = dict(n_failures=len(fails), mean_stall_steps_at_end=mean_stall,
                       frac_failures_stall_ge_K=frac_ge_k)
            aggregate['tail_metrics'][arm][ts] = row
            print(f"  {arm:7s} @{ts}: fails={len(fails)}, "
                  f"mean end-stall={mean_stall:.1f}, ≥K: {frac_ge_k*100:.0f}%")

    # Per-scenario at the best checkpoint per arm (by mean success).
    for arm in ('base', 'repair'):
        rows = aggregate['arms'].get(arm, {})
        if not rows:
            continue
        best = max(rows, key=lambda ts: rows[ts]['success_mean'])
        per_seed = evals[(arm, best)]
        by = defaultdict(list)
        for eps in per_seed.values():
            for e in eps:
                by[e['scenario']].append(e['success'])
        table = {k: sum(v) / len(v) for k, v in by.items()}
        aggregate['per_scenario'][arm] = dict(checkpoint=best, success=table)
        print(f"\nper-scenario {arm} @{best} (pooled 3 seeds):")
        for k in sorted(table, key=lambda k: -table[k]):
            print(f"  {k:26s}: {table[k]*100:5.1f}%")

    # Training tail-state density: share of training episodes ended by a cut.
    for arm in ('base', 'repair'):
        aggregate['training_tail_density'][arm] = {}
        for s in SEEDS:
            f = RUN / f'{arm}_s{s}' / 'episodes.jsonl'
            if not f.exists():
                continue
            eps = [json.loads(l) for l in f.open()]
            cuts = sum(1 for e in eps if e.get('stall_cut'))
            aggregate['training_tail_density'][arm][s] = dict(
                episodes=len(eps), stall_cuts=cuts, cut_share=cuts / len(eps))
        d = aggregate['training_tail_density'][arm]
        if d:
            print(f"\ntraining stall-cut share {arm}: "
                  + "  ".join(f"s{s}:{v['cut_share']*100:.1f}%" for s, v in d.items()))

    out = RUN / 'aggregate.json'
    out.write_text(json.dumps(aggregate, indent=1))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
