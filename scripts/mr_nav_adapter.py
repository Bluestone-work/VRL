"""Record and validate the upstream mr-nav interface before a VRL adapter run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    license_path = a.upstream / "LICENSE"
    record = {
        "upstream": "https://github.com/yinghansun/mr-nav",
        "commit": commit,
        "license": str(license_path),
        "license_sha256": sha(license_path) if license_path.exists() else None,
        "raw_observation": "72 virtual-LiDAR values + 2-D target + 2-D position + previous scalar action",
        "raw_action": "one scalar heading angle, radian or degree according to VesselEnvCfg",
        "raw_control": "2-D point-mass velocity with Gaussian disturbance",
        "vrl_mapping": {
            "observation": "common 2-D subtask only, LiDAR projected into the vessel cross-section",
            "action": "heading angle mapped to a unit 2-D direction, then VRL safety protection",
            "frequency": "VRL control_dt_s",
            "deadline": "VRL horizon",
        },
        "status": "interface-audited_pending_dataset_and_policy_export",
        "fairness_scope": "Does not claim a 3-D vascular result until the upstream 2-D dataset and policy are mapped and evaluated.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
