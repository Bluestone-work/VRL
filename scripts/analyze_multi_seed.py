"""Aggregate multi-seed experiment results with statistics.

Computes mean ± std across seeds, runs t-test for significance,
and generates publication-ready tables.

    python scripts/analyze_multi_seed.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def load_seed_results(exp_dir: Path, architecture: str, seeds: List[int]) -> Dict:
    """Load final metrics from all seeds of one architecture."""
    results = {
        "success": [],
        "removal_rate": [],
        "episode_return": [],
        "contact_miss": [],
    }

    for seed in seeds:
        summary_path = exp_dir / f"{architecture}_r5_c3/seed_{seed}/summary.json"
        if not summary_path.exists():
            print(f"⚠️  Missing: {summary_path}")
            continue

        with open(summary_path) as f:
            data = json.load(f)
            final = data["final_metrics"]
            results["success"].append(final["success"])
            results["removal_rate"].append(final["removal_rate"])
            results["episode_return"].append(final["episode_return"])
            results["contact_miss"].append(final["contact_miss"])

    return results


def print_table(gat_results: Dict, edge_results: Dict) -> None:
    """Print comparison table with statistics."""
    print("\n" + "=" * 80)
    print("MULTI-SEED RESULTS (5 robots, 3 clots, 300k steps)")
    print("=" * 80)
    print(f"{'Metric':<20} {'GAT':<25} {'EdgeGAT':<25} {'Δ':<10}")
    print("-" * 80)

    metrics = [
        ("Success Rate", "success", "%"),
        ("Removal Rate", "removal_rate", "%"),
        ("Episode Return", "episode_return", ""),
        ("Contact Miss", "contact_miss", "%"),
    ]

    for label, key, unit in metrics:
        gat_vals = np.array(gat_results[key])
        edge_vals = np.array(edge_results[key])

        if len(gat_vals) == 0 or len(edge_vals) == 0:
            print(f"{label:<20} {'N/A':<25} {'N/A':<25}")
            continue

        # Convert to percentage if needed
        if unit == "%":
            gat_vals = gat_vals * 100
            edge_vals = edge_vals * 100

        gat_mean = np.mean(gat_vals)
        gat_std = np.std(gat_vals, ddof=1) if len(gat_vals) > 1 else 0

        edge_mean = np.mean(edge_vals)
        edge_std = np.std(edge_vals, ddof=1) if len(edge_vals) > 1 else 0

        delta = edge_mean - gat_mean

        # Format strings
        if unit == "%":
            gat_str = f"{gat_mean:5.1f}% ± {gat_std:4.1f}%"
            edge_str = f"{edge_mean:5.1f}% ± {edge_std:4.1f}%"
            delta_str = f"{delta:+5.1f}%"
        else:
            gat_str = f"{gat_mean:6.1f} ± {gat_std:5.1f}"
            edge_str = f"{edge_mean:6.1f} ± {edge_std:5.1f}"
            delta_str = f"{delta:+6.1f}"

        print(f"{label:<20} {gat_str:<25} {edge_str:<25} {delta_str:<10}")

    print("-" * 80)

    # Statistical test
    gat_success = np.array(gat_results["success"]) * 100
    edge_success = np.array(edge_results["success"]) * 100

    if len(gat_success) >= 2 and len(edge_success) >= 2:
        t_stat, p_value = stats.ttest_ind(edge_success, gat_success)
        print(f"\nStatistical Test (Success Rate):")
        print(f"  t-statistic: {t_stat:+.3f}")
        print(f"  p-value:     {p_value:.4f}")

        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        print(f"  Significance: {sig}")
        print(f"  Interpretation: EdgeGAT is ", end="")
        if p_value < 0.05:
            print(f"significantly better (p < 0.05)")
        else:
            print(f"not significantly different (p ≥ 0.05)")

    print("=" * 80)


def print_latex_table(gat_results: Dict, edge_results: Dict) -> None:
    """Generate LaTeX table for paper."""
    print("\n% LaTeX table (copy to paper):")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Multi-seed comparison: GAT vs EdgeGAT (5 robots, 3 clots)}")
    print("\\begin{tabular}{l c c c}")
    print("\\toprule")
    print("Metric & GAT & EdgeGAT & $\\Delta$ \\\\")
    print("\\midrule")

    metrics = [
        ("Success Rate (\\%)", "success", "%"),
        ("Removal Rate (\\%)", "removal_rate", "%"),
        ("Episode Return", "episode_return", ""),
    ]

    for label, key, unit in metrics:
        gat_vals = np.array(gat_results[key])
        edge_vals = np.array(edge_results[key])

        if len(gat_vals) == 0 or len(edge_vals) == 0:
            continue

        if unit == "%":
            gat_vals = gat_vals * 100
            edge_vals = edge_vals * 100

        gat_mean = np.mean(gat_vals)
        gat_std = np.std(gat_vals, ddof=1) if len(gat_vals) > 1 else 0
        edge_mean = np.mean(edge_vals)
        edge_std = np.std(edge_vals, ddof=1) if len(edge_vals) > 1 else 0
        delta = edge_mean - gat_mean

        if unit == "%":
            print(f"{label} & {gat_mean:.1f} $\\pm$ {gat_std:.1f} & "
                  f"{edge_mean:.1f} $\\pm$ {edge_std:.1f} & {delta:+.1f} \\\\")
        else:
            print(f"{label} & {gat_mean:.1f} $\\pm$ {gat_std:.1f} & "
                  f"{edge_mean:.1f} $\\pm$ {edge_std:.1f} & {delta:+.1f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")


def main() -> None:
    exp_dir = Path("experiments/multi_seed")
    seeds = [43, 44, 45]

    print("🔍 Loading multi-seed results...")
    gat_results = load_seed_results(exp_dir, "gat", seeds)
    edge_results = load_seed_results(exp_dir, "edge_gat", seeds)

    print(f"  GAT: {len(gat_results['success'])} / {len(seeds)} seeds")
    print(f"  EdgeGAT: {len(edge_results['success'])} / {len(seeds)} seeds")

    if len(gat_results["success"]) == 0 or len(edge_results["success"]) == 0:
        print("\n⚠️  Not enough data yet. Wait for experiments to complete.")
        return

    print_table(gat_results, edge_results)
    print_latex_table(gat_results, edge_results)

    # Save raw data
    output = {
        "gat": {k: [float(v) for v in vals] for k, vals in gat_results.items()},
        "edge_gat": {k: [float(v) for v in vals] for k, vals in edge_results.items()},
    }

    output_path = exp_dir / "aggregated_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Raw data saved to {output_path}")


if __name__ == "__main__":
    main()
