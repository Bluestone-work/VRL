"""Create compact tables, plots, and a browsable media index for the matrix."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/full_matrix_20260906")


def wilson(successes: int, total: int):
    if total == 0:
        return [None, None]
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def main():
    scenarios = [line.strip() for line in (ROOT / "scenarios.txt").read_text().splitlines() if line.strip()]
    labels = [line.split("|", 1)[0] for line in (ROOT / "methods.tsv").read_text().splitlines() if line.strip()]
    rows = []
    heatmap = []
    for label in labels:
        summary = json.loads((ROOT / "methods" / label / "summary.json").read_text())
        episodes = summary["episodes_completed"]
        successes = summary["successes"]
        macro = summary["macro_completed_episodes"]
        rows.append({"method": label, "episodes": episodes, "successes": successes,
                     "success_rate": macro["success"], "removal_rate": macro["removal_rate"],
                     "wilson95_low": wilson(successes, episodes)[0],
                     "wilson95_high": wilson(successes, episodes)[1]})
        heatmap.append([summary["per_scenario"].get(scenario, {}).get("success", np.nan)
                        for scenario in scenarios])

    with (ROOT / "method_ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: row["success_rate"], reverse=True))

    with (ROOT / "scenario_success_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["method", *scenarios])
        writer.writerows([[label, *values] for label, values in zip(labels, heatmap)])

    array = np.asarray(heatmap, dtype=float)
    figure, axis = plt.subplots(figsize=(18, 8))
    image = axis.imshow(array, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(scenarios)), scenarios, rotation=65, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=9)
    axis.set_title("Full method × scenario success rate (20 episodes per cell)")
    figure.colorbar(image, ax=axis, label="success rate")
    figure.tight_layout()
    figure.savefig(ROOT / "scenario_success_heatmap.png", dpi=180)
    plt.close(figure)

    index_lines = ["# 全量媒体索引", "", "每个方法每个场景各有一个 GIF 和一个 MP4；路径按方法/场景分层。", ""]
    for label in labels:
        index_lines.extend([f"## {label}", ""])
        for scenario in scenarios:
            gif = ROOT / "media" / label / scenario / f"{label}.gif"
            mp4 = ROOT / "media" / label / scenario / f"{label}.mp4"
            index_lines.append(f"- `{scenario}` — [{label}.gif]({gif.relative_to(ROOT)}) · [{label}.mp4]({mp4.relative_to(ROOT)})")
        index_lines.append("")
    (ROOT / "MEDIA_INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    print(json.dumps({"methods": len(labels), "scenarios": len(scenarios),
                      "ranking": str(ROOT / "method_ranking.csv"),
                      "heatmap": str(ROOT / "scenario_success_heatmap.png"),
                      "media_index": str(ROOT / "MEDIA_INDEX.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
