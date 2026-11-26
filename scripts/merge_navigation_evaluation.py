"""Resume and merge isolated navigation evaluation artifacts.

The command is deliberately strict. It only writes a merged result after every
manifest row is present exactly once and every artifact agrees on model hash,
scenario identity and target assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IDENTITY = ("scenario_hash", "accepted_seed", "realized_scenario_hash", "navigation_targets")


def _norm(row):
    return tuple(tuple(x) if k == "navigation_targets" and x is not None else x
                 for k, x in ((key, row.get(key)) for key in IDENTITY))


def merge(model: Path, scene_list: Path, out: Path):
    manifest = json.loads(scene_list.read_text())
    expected = []
    for entry in manifest.get("rows", []):
        row = dict(entry.get("row", entry))
        row["accepted_seed"] = entry.get("pursuit_accepted_seed", row.get("accepted_seed"))
        row["realized_scenario_hash"] = entry.get("pursuit_realized_scenario_hash", row.get("realized_scenario_hash"))
        row["navigation_targets"] = entry.get("pursuit_navigation_targets", row.get("navigation_targets"))
        expected.append(row)
    expected_hashes = [row.get("scenario_hash") for row in expected]
    if any(not x for x in expected_hashes) or len(set(expected_hashes)) != len(expected):
        raise ValueError("scene list contains missing or duplicate scenario_hash")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    prefix = out.stem
    candidates = sorted(out.parent.glob(prefix + "_chunk*.json")) + sorted(out.parent.glob(prefix + "_single*.json"))
    episodes = {}
    inputs = []
    for path in candidates:
        data = json.loads(path.read_text())
        if data.get("model_sha256") != model_sha:
            continue
        if not data.get("overall", {}).get("complete"):
            continue
        for row in data.get("episodes", []):
            key = row.get("scenario_hash")
            if not key or any(row.get(field) is None for field in IDENTITY):
                raise ValueError(f"missing identity fields in {path}")
            if key in episodes:
                if episodes[key] != row:
                    raise ValueError(f"conflicting duplicate scenario identity in {path}: {key}")
                continue
            episodes[key] = row
        inputs.append(str(path))
    missing = [row.get("scenario_hash") for row in expected if row.get("scenario_hash") not in episodes]
    extra = [key for key in episodes if key not in set(expected_hashes)]
    mismatches = []
    expected_by_hash = {row.get("scenario_hash"): row for row in expected}
    for key, row in episodes.items():
        for field in IDENTITY[1:]:
            expected_value = expected_by_hash[key].get(field)
            actual_value = row.get(field)
            if field == "navigation_targets":
                expected_value, actual_value = tuple(expected_value or ()), tuple(actual_value or ())
            if expected_value != actual_value:
                mismatches.append({"scenario_hash": key, "field": field,
                                   "expected": expected_value, "actual": actual_value})
    if missing or extra or mismatches or len(episodes) != len(expected):
        raise ValueError(json.dumps({"missing": missing, "extra": extra, "identity_mismatches": mismatches,
                                     "found": len(episodes), "expected": len(expected)}, indent=2))
    ordered = [episodes[row.get("scenario_hash")] for row in expected]
    from scripts.evaluate_navigation_dataset import aggregate, grouped_aggregates
    base = json.loads(Path(inputs[0]).read_text())
    for key in ("cache_key",):
        base.pop(key, None)
    base["episodes"] = ordered
    base["overall"] = aggregate(ordered)
    base.update(grouped_aggregates(ordered))
    base["scene_list"] = str(scene_list)
    base["model"] = str(model)
    base["model_sha256"] = model_sha
    base["merge_inputs"] = inputs
    base["merge_identity_key"] = list(IDENTITY)
    base["protocol"] = "strict_manifest_resume_merge"
    out.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "episodes": len(ordered), "model_sha256": model_sha,
                      "inputs": len(inputs), "complete": True}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scene-list", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    merge(a.model, a.scene_list, a.out)


if __name__ == "__main__":
    main()
