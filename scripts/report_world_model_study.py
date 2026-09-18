"""Aggregate the anatomical MAPPO/world-model study into auditable reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("success", "removal_rate", "return", "wall_hits")
SEEDS = (42, 43, 44)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


def discover_runs(root: Path):
    specifications = []
    for architecture in ("gat", "edge_bias_gat", "mlp"):
        for seed in SEEDS:
            specifications.append((
                "direct_hard", architecture, architecture, seed,
                root / "baselines" / f"{architecture}_r5_c3" / f"seed_{seed}",
            ))
    for seed in SEEDS:
        specifications.append((
            "curriculum", "gat", "curriculum", seed,
            root / "curriculum" / "gat_r5_c3" / f"seed_{seed}",
        ))
        for arm in ("pure", "mve"):
            specifications.append((
                "equal_budget", "gat", arm, seed,
                root / "comparisons" / f"seed_{seed}" / arm,
            ))

    runs = []
    missing = []
    for phase, architecture, arm, seed, run_dir in specifications:
        path = run_dir / "summary.json"
        if not path.exists():
            missing.append(str(path))
            continue
        summary = load_json(path)
        macro = summary["final_evaluation"]["macro"]
        runs.append({
            "phase": phase,
            "architecture": architecture,
            "arm": arm,
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "real_transitions": int(summary["real_transitions"]),
            "initial_real_transitions": int(summary.get("initial_real_transitions", 0)),
            "added_real_transitions": int(summary.get(
                "added_real_transitions", summary["real_transitions"]
            )),
            **{metric: float(macro[metric]) for metric in METRICS},
            "per_territory": summary["final_evaluation"]["per_territory"],
        })
    return runs, missing


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["phase"], run["architecture"], run["arm"])].append(run)
    aggregates = []
    for (phase, architecture, arm), items in sorted(groups.items()):
        record = {
            "phase": phase,
            "architecture": architecture,
            "arm": arm,
            "n": len(items),
            "seeds": [item["seed"] for item in items],
        }
        for metric in METRICS:
            values = np.asarray([item[metric] for item in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregates.append(record)
    return aggregates


def baseline_plot(reports: Path, aggregates):
    items = {
        item["architecture"]: item for item in aggregates
        if item["phase"] == "direct_hard"
    }
    labels = [name for name in ("gat", "edge_bias_gat", "mlp") if name in items]
    if not labels:
        return
    colors = ["#2878B5", "#D95319", "#3C8D40"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("success", "removal_rate"), ("Success rate", "Clot removal rate")
    ):
        means = [100 * items[label][f"{metric}_mean"] for label in labels]
        stds = [100 * items[label][f"{metric}_std"] for label in labels]
        axis.bar(labels, means, yerr=stds, capsize=4, color=colors[:len(labels)])
        axis.set_ylabel("Percent")
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("Direct hard anatomical baseline (mean +/- SD, 3 seeds)")
    fig.savefig(reports / "baseline_architectures.png", dpi=180)
    plt.close(fig)


def curriculum_plot(root: Path, reports: Path):
    colors = {42: "#2878B5", 43: "#D95319", 44: "#3C8D40"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    found = False
    for seed in SEEDS:
        path = root / "curriculum" / "gat_r5_c3" / f"seed_{seed}" / "eval_metrics.jsonl"
        if not path.exists():
            continue
        by_transition = {}
        for line in path.read_text().splitlines():
            if line:
                item = json.loads(line)
                by_transition[int(item["transitions"])] = item["macro"]
        if not by_transition:
            continue
        found = True
        x = np.asarray(sorted(by_transition)) / 1e6
        axes[0].plot(x, [100 * by_transition[t]["success"] for t in sorted(by_transition)],
                     marker="o", color=colors[seed], label=f"seed {seed}")
        axes[1].plot(x, [100 * by_transition[t]["removal_rate"] for t in sorted(by_transition)],
                     marker="o", color=colors[seed], label=f"seed {seed}")
    if not found:
        plt.close(fig)
        return
    for axis, title in zip(axes, ("Held-out success", "Held-out clot removal")):
        axis.axvspan(0.0, 0.2, color="#BBDDEE", alpha=0.22)
        axis.axvspan(0.2, 0.5, color="#F4C28B", alpha=0.18)
        axis.axvspan(0.5, 1.0, color="#B9D8B5", alpha=0.18)
        axis.set_xlabel("Real transitions (millions)")
        axis.set_ylabel("Percent")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("GAT curriculum with fixed hard anatomical evaluation")
    fig.savefig(reports / "curriculum_learning_curves.png", dpi=180)
    plt.close(fig)


def comparison_plot(reports: Path, runs):
    paired = defaultdict(dict)
    for run in runs:
        if run["phase"] == "equal_budget":
            paired[run["seed"]][run["arm"]] = run
    complete = {seed: arms for seed, arms in paired.items() if {"pure", "mve"} <= set(arms)}
    if not complete:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("success", "removal_rate"), ("Success rate", "Clot removal rate")
    ):
        for seed, arms in sorted(complete.items()):
            values = [100 * arms[arm][metric] for arm in ("pure", "mve")]
            axis.plot((0, 1), values, marker="o", linewidth=1.8, label=f"seed {seed}")
        axis.set_xticks((0, 1), ("Pure MAPPO", "MAPPO + MVE"))
        axis.set_ylabel("Percent")
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Paired equal-real-transition continuation")
    fig.savefig(reports / "equal_budget_paired.png", dpi=180)
    plt.close(fig)


def territory_plot(reports: Path, runs):
    arm_runs = {
        arm: [run for run in runs if run["phase"] == "equal_budget" and run["arm"] == arm]
        for arm in ("pure", "mve")
    }
    if not all(arm_runs.values()):
        return
    territories = list(arm_runs["pure"][0]["per_territory"])
    pure = [np.mean([run["per_territory"][name]["removal_rate"] for run in arm_runs["pure"]])
            for name in territories]
    mve = [np.mean([run["per_territory"][name]["removal_rate"] for run in arm_runs["mve"]])
           for name in territories]
    y = np.arange(len(territories))
    height = 0.38
    fig, axis = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    axis.barh(y - height / 2, 100 * np.asarray(pure), height, color="#2878B5", label="Pure")
    axis.barh(y + height / 2, 100 * np.asarray(mve), height, color="#D95319", label="MVE")
    axis.set_yticks(y, territories)
    axis.invert_yaxis()
    axis.set_xlabel("Mean clot removal (%)")
    axis.set_title("Equal-budget result by held-out anatomical territory")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(reports / "territory_removal_comparison.png", dpi=180)
    plt.close(fig)


def markdown_report(root: Path, aggregates, missing, world_metrics):
    lines = [
        "# Anatomical MAPPO and World-Model Study",
        "",
        "All policy results below are macro averages over 14 held-out anatomical "
        "territories with 20 episodes per territory and seed.",
        "",
        "| Phase | Arm | Seeds | Success (%) | Removal (%) | Return | Wall hits |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['phase']} | {item['arm']} | {item['n']} | "
            f"{100 * item['success_mean']:.2f} +/- {100 * item['success_std']:.2f} | "
            f"{100 * item['removal_rate_mean']:.2f} +/- {100 * item['removal_rate_std']:.2f} | "
            f"{item['return_mean']:.2f} +/- {item['return_std']:.2f} | "
            f"{item['wall_hits_mean']:.2f} +/- {item['wall_hits_std']:.2f} |"
        )
    lines.extend(["", "## World-model validation", ""])
    if world_metrics is None:
        lines.append("World-model metrics are not available.")
    else:
        one = world_metrics["one_step"]
        five = world_metrics["multistep"]["5"]
        lines.extend([
            f"Gate passed: **{world_metrics['gate']['passed']}**",
            "",
            f"- One-step observation RMSE: {one['obs_rmse']:.6f}",
            f"- One-step persistence RMSE: {one['obs_persistence_rmse']:.6f}",
            f"- Reward MAE / zero baseline: {one['reward_relative_to_zero']:.4f}",
            f"- Five-step RMSE / persistence: {five.get('relative_to_persistence', float('nan')):.4f}",
            f"- Held-out uncertainty p95: {one['uncertainty_p95']:.6f}",
        ])
    lines.extend(["", "## Completeness", ""])
    if missing:
        lines.append(f"Incomplete summaries: {len(missing)}")
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("All planned summaries are present.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    root = Path(args.study_root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    runs, missing = discover_runs(root)
    flat_rows = [{key: value for key, value in run.items() if key != "per_territory"}
                 for run in runs]
    fields = (
        "phase", "architecture", "arm", "seed", "run_dir", "real_transitions",
        "initial_real_transitions", "added_real_transitions", *METRICS,
    )
    write_csv(reports / "metrics_by_seed.csv", fields, flat_rows)

    territory_rows = []
    for run in runs:
        for territory, metrics in run["per_territory"].items():
            territory_rows.append({
                "phase": run["phase"], "architecture": run["architecture"],
                "arm": run["arm"], "seed": run["seed"], "territory": territory,
                **{metric: metrics[metric] for metric in METRICS},
            })
    write_csv(
        reports / "metrics_by_territory.csv",
        ("phase", "architecture", "arm", "seed", "territory", *METRICS),
        territory_rows,
    )

    aggregates = aggregate_runs(runs)
    world_path = root / "world_model" / "metrics.json"
    world_metrics = load_json(world_path) if world_path.exists() else None
    result = {
        "complete": not missing,
        "missing": missing,
        "aggregates": aggregates,
        "world_model": world_metrics,
    }
    (reports / "aggregate_metrics.json").write_text(json.dumps(result, indent=2))
    (reports / "report.md").write_text(
        markdown_report(root, aggregates, missing, world_metrics)
    )
    baseline_plot(reports, aggregates)
    curriculum_plot(root, reports)
    comparison_plot(reports, runs)
    territory_plot(reports, runs)
    print(json.dumps({
        "runs": len(runs), "missing": len(missing), "reports": str(reports.resolve())
    }, indent=2))


if __name__ == "__main__":
    main()
