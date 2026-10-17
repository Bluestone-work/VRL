"""Read the frozen navigation_dataset_v1 manifest without resampling scenes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/datasets/NAVIGATION_V1/manifest.jsonl"


def iter_rows(*, split=None, anatomy=None, clusters=None, manifest=MANIFEST):
    """Yield immutable manifest rows filtered by registered fields."""
    with Path(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split is not None and row["split"] != split:
                continue
            if anatomy is not None and row["anatomy"] != anatomy:
                continue
            if clusters is not None and int(row["clusters"]) != int(clusters):
                continue
            yield row


def reset_options(row):
    """Convert one manifest row to the CurriculumVascularEnv reset contract."""
    if row.get("dataset_version") != "navigation_dataset_v1":
        raise ValueError("unsupported navigation dataset row")
    return {"dataset_row": dict(row)}


def condition_key(row):
    return (row["anatomy"], int(row["clusters"]), row["flow_mode"],
            row["clot_pattern"], row["obstacle_density"], row["obstacle_layout"])
