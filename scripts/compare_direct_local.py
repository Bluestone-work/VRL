"""Create the preregistered Direct Local versus flow-guided comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "success", "removal_rate", "steps", "completion_steps", "first_contact_steps",
    "wall_contact_rate", "collision_rate", "mean_action_magnitude", "mean_flow_speed",
    "blood_flow_exposure",
    "mean_simultaneously_contacted_clots",
)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    direct, baseline = read(args.direct), read(args.baseline)
    rows = []
    for metric in METRICS:
        d = direct["aggregate_by_training_seed"][metric]
        b = baseline["aggregate_by_training_seed"][metric]
        delta = d["mean"] - b["mean"]
        rows.append({
            "metric": metric,
            "baseline_mean": b["mean"], "baseline_sample_sd": b["sample_sd"],
            "direct_local_mean": d["mean"], "direct_local_sample_sd": d["sample_sd"],
            "absolute_difference": delta,
            "relative_difference": delta / b["mean"] if b["mean"] else None,
        })
    result = {
        "baseline": {"experiment_id": "EXP_0004", "arm": "geodesic_v", "split": baseline["split"]},
        "direct_local": {"experiment_id": direct["experiment_id"], "split": direct["split"]},
        "statistics_unit": "training_seed",
        "sealed_test_accessed": False,
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# EXP_0005 Direct Local versus Flow-guided Baseline", "",
             "All values are means over three training seeds; SD is sample SD over seed-level metrics.", "",
             "| Metric | Flow-guided baseline mean +/- SD | Direct Local mean +/- SD | Absolute diff | Relative diff |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['baseline_mean']:.6f} +/- {row['baseline_sample_sd']:.6f} | "
                     f"{row['direct_local_mean']:.6f} +/- {row['direct_local_sample_sd']:.6f} | "
                     f"{row['absolute_difference']:+.6f} | {row['relative_difference']:+.2%} |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
