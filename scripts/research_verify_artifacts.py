"""Read-only final integrity and paired checkpoint checks for EXP_0001."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/runs/EXP_0001"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    destination = RUN / "checks/final_integrity.json"
    if destination.exists():
        raise RuntimeError("Integrity record exists; refusing to overwrite")
    snapshot = read(RUN / "provenance/snapshot.json")
    frozen = Path(snapshot["directory"])
    inventory = read(RUN / "provenance/source_inventory.json")
    changed_frozen = [row["path"] for row in inventory
                      if digest(frozen / row["path"]) != row["sha256"]]
    original_python = [row for row in inventory if row["path"].endswith(".py")]
    changed_original_python = [row["path"] for row in original_python
                               if digest(ROOT / row["path"]) != row["sha256"]]
    config_hash = digest(ROOT / "configs/experiments/EXP_0001.json")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=frozen, text=True).strip()
    comparisons = []
    torch.set_num_threads(1)
    for reference in read(RUN / "provenance/reference.json"):
        directory = Path(reference["directory"])
        seed = read(directory / "summary.json")["seed"]
        previous_path = directory / "final_policy.pt"
        current_path = RUN / f"training/seed_{seed}/final_policy.pt"
        old = torch.load(previous_path, map_location="cpu", weights_only=False)
        new = torch.load(current_path, map_location="cpu", weights_only=False)
        modules = {}
        for module in ("actor", "critic"):
            keys_match = old[module].keys() == new[module].keys()
            unequal = []
            maximum = 0.0
            if keys_match:
                for key in old[module]:
                    a, b = old[module][key], new[module][key]
                    if not torch.equal(a, b):
                        unequal.append(key)
                        maximum = max(maximum, float((a.double() - b.double()).abs().max()))
            modules[module] = {"keys_match": keys_match, "tensor_count": len(old[module]),
                               "unequal_tensors": unequal, "maximum_absolute_difference": maximum,
                               "exactly_equal": keys_match and not unequal}
        comparisons.append({"seed": seed, "archived_checkpoint": str(previous_path),
                            "reproduced_checkpoint": str(current_path),
                            "archived_sha256": digest(previous_path),
                            "archive_unchanged": digest(previous_path) == reference["checkpoint_sha256"],
                            "reproduced_sha256": digest(current_path), "modules": modules})
    result = {"experiment_id": "EXP_0001", "checked_at": datetime.now(timezone.utc).isoformat(),
              "execution_commit": actual_commit, "commit_matches": actual_commit == snapshot["git_commit"],
              "frozen_files_checked": len(inventory), "changed_frozen_files": changed_frozen,
              "original_python_files_checked": len(original_python),
              "changed_original_python_files": changed_original_python,
              "configuration_sha256": config_hash, "configuration_matches": config_hash == snapshot["config_sha256"],
              "original_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "paired_checkpoints": comparisons,
              "scope": "Actor/critic tensor equality only; full serialized files contain other run state."}
    result["source_integrity_passed"] = (result["commit_matches"] and result["configuration_matches"]
        and not changed_frozen and not changed_original_python and all(r["archive_unchanged"] for r in comparisons))
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    if not result["source_integrity_passed"]:
        raise AssertionError("Source integrity mismatch; preserve evidence and investigate")


if __name__ == "__main__":
    main()
