"""Freeze the exact manifest rows used by an existing pursuit evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.navigation_dataset import MANIFEST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pursuit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest = [json.loads(line) for line in Path(MANIFEST).read_text().splitlines() if line.strip()]
    by_hash = {row["scenario_hash"]: (i, row) for i, row in enumerate(manifest)}
    episodes = json.loads(a.pursuit.read_text())["episodes"]
    seen = set()
    selected = []
    errors = []
    for episode_index, episode in enumerate(episodes):
        key = episode.get("scenario_hash")
        if key in seen:
            errors.append({"episode_index": episode_index, "scenario_hash": key, "error": "duplicate"})
            continue
        seen.add(key)
        if key not in by_hash:
            errors.append({"episode_index": episode_index, "scenario_hash": key, "error": "missing_manifest_row"})
            continue
        manifest_index, row = by_hash[key]
        identity = {field: row.get(field) for field in
                    ("split", "clusters", "scene_seed", "scenario_hash")}
        selected.append({"episode_index": episode_index, "manifest_index": manifest_index,
                         "row": row, "pursuit_identity": identity,
                         "pursuit_accepted_seed": episode.get("accepted_seed"),
                         "pursuit_realized_scenario_hash": episode.get("realized_scenario_hash"),
                         "pursuit_navigation_targets": episode.get("navigation_targets")})
    payload = {
        "source": str(a.pursuit),
        "manifest": str(Path(MANIFEST)),
        "manifest_sha256": hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(a.pursuit.read_bytes()).hexdigest(),
        "episodes": len(episodes), "selected": len(selected), "errors": errors,
        "rows": selected,
    }
    if errors or len(selected) != len(episodes):
        raise SystemExit(json.dumps(payload, ensure_ascii=False))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": len(selected), "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
