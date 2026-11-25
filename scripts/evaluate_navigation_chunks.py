"""Run frozen learned-policy evaluation in short simulator processes.

The vascular integrator can segfault after a long sequence of resets even when
individual episodes are valid. This wrapper keeps each chunk auditable, merges
only complete JSON outputs, and preserves the original filtered-manifest index.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--kind", choices=("ppo", "dqn", "sac"), required=True)
    p.add_argument("--action-set", default="semantic_reduced")
    p.add_argument("--split", required=True)
    p.add_argument("--clusters", type=int, required=True)
    p.add_argument("--total", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--flow-feature", action="store_true")
    p.add_argument("--centralized-critic", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.total <= 0 or args.chunk_size <= 0:
        p.error("total and chunk-size must be positive")
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    chunks = []
    for start in range(0, args.total, args.chunk_size):
        indices = list(range(start, min(args.total, start + args.chunk_size)))
        chunk_out = args.out.with_name(args.out.stem + f"_chunk{start:04d}.json")
        cmd = [sys.executable, str(Path(__file__).with_name("evaluate_navigation_dataset.py")),
               "--model", str(args.model), "--kind", args.kind, "--action-set", args.action_set,
               "--split", args.split, "--clusters", str(args.clusters), "--index-list", ",".join(map(str, indices)),
               "--out", str(chunk_out), "--device", args.device]
        if args.flow_feature:
            cmd.append("--flow-feature")
        if args.centralized_critic:
            cmd.append("--centralized-critic")
        print("running", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)
        data = json.loads(chunk_out.read_text())
        if not data["overall"].get("complete", False) or len(data.get("episodes", [])) != len(indices):
            raise RuntimeError(f"incomplete chunk {start}: {chunk_out}")
        chunks.append(data)
    episodes = [episode for chunk in chunks for episode in chunk["episodes"]]
    from scripts.evaluate_navigation_dataset import aggregate
    merged = dict(chunks[0])
    merged["model"] = str(args.model)
    merged["split"] = args.split
    merged["clusters"] = args.clusters
    merged["protocol"] = "frozen_manifest_evaluation; isolated simulator chunks"
    merged["overall"] = aggregate(episodes)
    merged["episodes"] = episodes
    merged["chunks"] = [str(args.out.with_name(args.out.stem + f"_chunk{start:04d}.json"))
                        for start in range(0, args.total, args.chunk_size)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(merged["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
