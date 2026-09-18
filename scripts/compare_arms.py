"""Compare architectures from their independent eval logs.

Reads `experiments/eval_ms_<arch>_<seed>.log` (written by eval_checkpoint.py),
which are all run on the SAME unseen seed set, so the arms are directly
comparable. Training-end numbers in summary.json are not used: those come from
20 episodes on the training env's own RNG stream and are far too noisy to rank
architectures.

    python scripts/compare_arms.py
    python scripts/compare_arms.py --baseline gat
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats

EVAL_DIR = Path("experiments")
SEEDS = (43, 44, 45)
EPISODES = 50  # per seed, as launched

PATTERNS = {
    "success": r"success rate\s+([\d.]+)%",
    "removal": r"removal rate\s+([\d.]+)%",
    "contact_miss": r"contact miss\s+([\d.]+)%",
    "wall_hits": r"wall hits\s+([\d.]+)",
    "steps": r"episode length\s+([\d.]+)",
}


def read_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text()
    out = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if m is None:
            return None  # incomplete run
        out[key] = float(m.group(1))
    return out


def collect(arch: str) -> dict[str, np.ndarray]:
    rows = []
    for s in SEEDS:
        r = read_log(EVAL_DIR / f"eval_ms_{arch}_{s}.log")
        if r is None:
            print(f"  ⚠️  {arch} seed {s}: missing or incomplete")
            continue
        rows.append(r)
    if not rows:
        return {}
    return {k: np.array([r[k] for r in rows]) for k in PATTERNS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="gat",
                    help="arm every other arm is compared against")
    ap.add_argument("--arms", nargs="*",
                    default=["gat", "edge_gat", "edge_bias_gat", "mlp"])
    args = ap.parse_args()

    print("Loading independent-eval logs (same unseen seed set, 50 ep/seed)...")
    data = {a: collect(a) for a in args.arms}
    data = {a: d for a, d in data.items() if d}

    if args.baseline not in data:
        raise SystemExit(f"baseline {args.baseline!r} has no complete logs")

    n_seeds = {a: len(d["success"]) for a, d in data.items()}

    print(f"\n{'='*94}")
    print("ARCHITECTURE COMPARISON  (5 robots, 3 clots, 300k steps)")
    print(f"{'='*94}")
    print(f"{'arch':<16}{'seeds':<7}{'success %':<20}{'removal %':<14}"
          f"{'wall/ep':<14}{'steps':<12}")
    print("-" * 94)

    for a, d in data.items():
        su, re_, wh, st = d["success"], d["removal"], d["wall_hits"], d["steps"]
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{a:<16}{n_seeds[a]:<7}"
              f"{su.mean():5.1f} ± {sd(su):4.1f}       "
              f"{re_.mean():5.1f} ± {sd(re_):4.1f} "
              f"{wh.mean():6.2f} ± {sd(wh):4.2f} "
              f"{st.mean():6.0f} ± {sd(st):3.0f}")
    print("-" * 94)

    # Per-seed values matter more than the summary stats at n=3: non-overlapping
    # ranges are stronger evidence than a p-value from three points.
    print("\nper-seed success % (overlap check):")
    for a, d in data.items():
        print(f"  {a:<16}{sorted(d['success'].tolist())}")

    base = data[args.baseline]["success"]
    print(f"\nvs baseline '{args.baseline}':")
    print(f"  {'arch':<16}{'Δ success':<12}{'seed-level Welch':<26}{'pooled episodes':<30}")
    for a, d in data.items():
        if a == args.baseline:
            continue
        arm = d["success"]
        delta = arm.mean() - base.mean()

        if len(arm) > 1 and len(base) > 1:
            t, p = stats.ttest_ind(arm, base, equal_var=False)
            welch = f"t={t:+.2f}, p={p:.3f}"
        else:
            welch = "n/a (need >=2 seeds)"

        # Pooled over episodes: tighter, but ignores seed clustering, so the
        # p-value is optimistic. Reported alongside, never instead of, Welch.
        na, nb = len(arm) * EPISODES, len(base) * EPISODES
        ka = int(round(arm.mean() / 100 * na))
        kb = int(round(base.mean() / 100 * nb))
        _, pf = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])
        pooled = f"{ka}/{na} vs {kb}/{nb}, Fisher p={pf:.4f}"

        print(f"  {a:<16}{delta:+7.1f}%    {welch:<26}{pooled:<30}")

    print(f"\n{'='*94}")
    print("Note: seed-level Welch respects between-seed variance but has almost no")
    print("power at n=3; pooled-episode Fisher is tighter but ignores seed clustering.")
    print("True evidence strength lies between them. Non-overlapping per-seed ranges")
    print("are the most trustworthy signal at this sample size.")
    print(f"{'='*94}\n")


if __name__ == "__main__":
    main()
