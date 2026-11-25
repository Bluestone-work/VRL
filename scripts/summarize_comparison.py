"""Create auditable tables, paired intervals, and figures for the comparison run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def bootstrap_diff(a, b, key="navigation_success", draws=4000, seed=41):
    amap = {x.get("scenario_hash"): x for x in a if x.get("scenario_hash")}
    bmap = {x.get("scenario_hash"): x for x in b if x.get("scenario_hash")}
    keys = sorted(set(amap) & set(bmap))
    if not keys:
        return {"pairs": 0, "mean_difference": None, "ci95": [None, None]}
    d = np.asarray([float(amap[k].get(key, False)) - float(bmap[k].get(key, False)) for k in keys])
    rng = np.random.default_rng(seed)
    sample = rng.choice(d, size=(draws, len(d)), replace=True).mean(axis=1)
    return {"pairs": len(d), "mean_difference": float(d.mean()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    methods = {"pursuit": root / "classical/pursuit_{split}.json"}
    for name in ("ippo", "mappo"):
        methods[name] = root / f"{name}/s41_{{split}}.json"
    rows = []
    for split, n in (("train180", 180), ("holdout100", 100)):
        split_name = "train" if split.startswith("train") else "holdout_test"
        paths = {"pursuit": root / f"classical/pursuit_{split}.json"}
        for name in ("ippo", "mappo"):
            paths[name] = root / f"{name}/s41_{split}.json"
        for method, path in paths.items():
            if not path.exists():
                continue
            result = load(path)
            overall = result.get("overall", {})
            if method == "pursuit":
                overall = overall.get("pursuit", overall)
            rows.append({"split": split_name, "episodes": overall.get("episodes"), "method": method,
                         "seed": 41, "success": overall.get("navigation_success_rate"),
                         "strict": overall.get("strict_success_rate"), "lost": overall.get("lost_rate"),
                         "wall": overall.get("wall_contact_s"), "collision": overall.get("collision_episode_rate"),
                         "complete": overall.get("complete", False)})
    fixed = root / "fixed_three_pursuit.json"
    fixed_summary = load(fixed) if fixed.exists() else None
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "local_control_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["split"])
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Vascular navigation comparison report", "", "## Protocol", "",
             "N=3 frozen navigation manifest, 13 semantic candidates, legal-action mask, online flow feature, safety parity, zero latency. IPPO and MAPPO are newly retrained arms with seeds 41–43 and 50,000 physical steps. The repository did not contain the historical EXP0141 checkpoint.", "",
             "## Local control table", "", "| split | method | episodes | success | strict | lost | wall robot·s | collision episodes |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def pct(x): return "—" if x is None else f"{100*x:.1f}%"
        def num(x): return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {r['split']} | {r['method']} | {r['episodes']} | {pct(r['success'])} | {pct(r['strict'])} | {pct(r['lost'])} | {num(r['wall'])} | {pct(r['collision'])} |")
    lines += ["", "## Paired differences", "", "The intervals below use deterministic bootstrap resampling of identical scenario hashes. The current run reports the available completed seed-41 evaluation while seeds 42–43 remain separate raw files for audit.", ""]
    for split in ("train180", "holdout100"):
        for method in ("ippo", "mappo"):
            ref = root / f"classical/pursuit_{split}.json"; arm = root / f"{method}/s41_{split}.json"
            if ref.exists() and arm.exists():
                rd = load(ref); ad = load(arm)
                rr = rd.get("episodes", []); aa = ad.get("episodes", [])
                d = bootstrap_diff(aa, rr)
                lines.append(f"- {method} minus pursuit, {split}: {d['pairs']} paired rows, Δsuccess={d['mean_difference']}, 95% CI={d['ci95']}")
    if fixed_summary:
        lines += ["", "## Fixed three-lesion coordination", "", "| clusters | episodes | all three reached | synchronized success (τ=30 s) | last arrival s | arrival span s |", "|---:|---:|---:|---:|---:|---:|"]
        for n, r in sorted(fixed_summary["by_clusters"].items(), key=lambda x: int(x[0])):
            lines.append(f"| {n} | {r['episodes']} | {100*r['all_three_rate']:.1f}% | {100*r['synchronized_rate']:.1f}% | {r['last_arrival_s']:.2f} | {r['arrival_span_s']:.2f} |")
    (a.out / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    if rows:
        fig, axes = plt.subplots(1, 2 if fixed_summary else 1, figsize=(10 if fixed_summary else 6, 4))
        if not isinstance(axes, np.ndarray): axes = [axes]
        ax = axes[0]
        names = [f"{r['method']}\n{r['split']}" for r in rows]
        vals = [100*r['success'] for r in rows]
        ax.bar(names, vals, color=["#2a78d6", "#eb6834", "#1baf7a"] * 2)
        ax.set_ylabel("navigation success (%)"); ax.set_ylim(0, 100); ax.tick_params(axis="x", rotation=35)
        ax.set_title("N=3 frozen manifest")
        if fixed_summary:
            ax = axes[1]
            ns = sorted(fixed_summary["by_clusters"], key=int)
            ax.plot(ns, [100*fixed_summary["by_clusters"][n]["all_three_rate"] for n in ns], "o-", label="all three")
            ax.plot(ns, [100*fixed_summary["by_clusters"][n]["synchronized_rate"] for n in ns], "o-", label="synchronized")
            ax.set_xlabel("clusters"); ax.set_ylabel("episodes (%)"); ax.set_ylim(0, 100); ax.legend()
            ax.set_title("Fixed three-lesion pursuit")
        fig.tight_layout(); fig.savefig(a.out / "comparison_overview.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
