"""Repair only origin-transition metadata in an already completed branch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--origin-transitions", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    paths = sorted(run_dir.glob("checkpoint_*.pt"))
    final = run_dir / "final_policy.pt"
    if final.exists():
        paths.append(final)
    changed = 0
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("training_state")
        if state is None:
            continue
        before = state.get("origin_transitions")
        if before == args.origin_transitions:
            continue
        state["origin_transitions"] = args.origin_transitions
        temporary = path.with_suffix(path.suffix + ".origin_repair")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        changed += 1
        print(f"repaired {path}: {before} -> {args.origin_transitions}")
    print(f"changed={changed} run_dir={run_dir}")


if __name__ == "__main__":
    main()
