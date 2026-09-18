"""Recover an append-only dataset to a checkpoint-aligned transition count."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--keep-transitions", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    manifest = root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    kept = []
    orphaned = []
    cumulative = 0
    for row in rows:
        next_count = cumulative + int(row["transitions"])
        if next_count <= args.keep_transitions:
            kept.append(row)
            cumulative = next_count
        else:
            orphaned.append(row)
    if cumulative != args.keep_transitions:
        raise ValueError(
            f"requested {args.keep_transitions} transitions, but shard boundary "
            f"lands at {cumulative}"
        )

    archive = root / f"orphaned_after_crash_{int(time.time())}"
    if orphaned:
        archive.mkdir(parents=True, exist_ok=False)
        for row in orphaned:
            source = root / row["shard"]
            if source.exists():
                shutil.move(str(source), str(archive / source.name))
    temporary = manifest.with_suffix(".jsonl.recovering")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept))
    temporary.replace(manifest)
    record = {
        "kept_transitions": cumulative,
        "kept_shards": [row["shard"] for row in kept],
        "orphaned_shards": [row["shard"] for row in orphaned],
        "archive": str(archive) if orphaned else None,
    }
    (root / f"recovery_{int(time.time())}.json").write_text(
        json.dumps(record, indent=2)
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
