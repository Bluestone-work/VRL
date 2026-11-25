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
    p.add_argument("--scene-list", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.total <= 0 or args.chunk_size <= 0:
        p.error("total and chunk-size must be positive")
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    scene_payload = json.loads(args.scene_list.read_text())
    entries = scene_payload.get("rows", [])
    if args.total != len(entries):
        p.error(f"total={args.total} does not match scene-list rows={len(entries)}")
    chunks = []
    for start in range(0, args.total, args.chunk_size):
        indices = list(range(start, min(args.total, start + args.chunk_size)))
        chunk_scene = args.out.with_name(args.out.stem + f"_scene{start:04d}.json")
        chunk_scene.write_text(json.dumps({**scene_payload, "rows": [entries[i] for i in indices]},
                                           ensure_ascii=False, indent=2) + "\n")
        chunk_out = args.out.with_name(args.out.stem + f"_chunk{start:04d}.json")
        if chunk_out.exists():
            cached = json.loads(chunk_out.read_text())
            if (cached.get("overall", {}).get("complete") and
                    len(cached.get("episodes", [])) == len(indices) and
                    cached.get("manifest_sha256") == scene_payload.get("manifest_sha256") and
                    cached.get("model_sha256") == __import__('hashlib').sha256(args.model.read_bytes()).hexdigest() and
                    [x.get("scenario_hash") for x in cached.get("episodes", [])] ==
                    [x["row"].get("scenario_hash") for x in entries[start:start + len(indices)]]):
                chunks.append(cached)
                print("reusing", chunk_out, flush=True)
                continue
        cmd = [sys.executable, str(Path(__file__).with_name("evaluate_navigation_dataset.py")),
               "--model", str(args.model), "--kind", args.kind, "--action-set", args.action_set,
               "--split", args.split, "--clusters", str(args.clusters), "--scene-list", str(chunk_scene),
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
    from scripts.evaluate_navigation_dataset import aggregate, grouped_aggregates
    merged = dict(chunks[0])
    merged["model"] = str(args.model)
    merged["split"] = args.split
    merged["clusters"] = args.clusters
    merged["protocol"] = "frozen_manifest_evaluation; isolated simulator chunks"
    merged["overall"] = aggregate(episodes)
    merged.update(grouped_aggregates(episodes))
    merged["episodes"] = episodes
    merged["chunks"] = [str(args.out.with_name(args.out.stem + f"_chunk{start:04d}.json"))
                        for start in range(0, args.total, args.chunk_size)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(merged["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
