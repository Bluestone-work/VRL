"""Decode every study video/GIF and save machine-readable pixel checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from environments.vessel_geometry import ALL_SCENARIOS


def inspect(path: Path, screenshot_path: Path | None = None):
    reader = imageio.get_reader(path)
    count = 0
    first = None
    last = None
    standard_deviations = []
    screenshot = None
    try:
        for frame in reader:
            frame = np.asarray(frame)
            if first is None:
                first = frame.copy()
            if count == 40 and screenshot_path is not None:
                screenshot = frame.copy()
            last = frame.copy()
            standard_deviations.append(float(frame.std()))
            count += 1
    finally:
        reader.close()
    if count == 0 or first is None or last is None:
        raise RuntimeError(f"no frames decoded from {path}")
    if screenshot_path is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(screenshot_path, last if screenshot is None else screenshot)
    return {
        "path": str(path.resolve()),
        "frames": count,
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
        "channels": int(first.shape[2]),
        "pixel_std_min": min(standard_deviations),
        "pixel_std_max": max(standard_deviations),
        "first_last_mean_absolute_delta": float(
            np.abs(first.astype(np.float32) - last.astype(np.float32)).mean()
        ),
        "bytes": path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_root)

    scenario_paths = sorted((root / "all_scenarios").glob("*.mp4"))
    found_scenarios = {path.stem for path in scenario_paths}
    expected_scenarios = set(ALL_SCENARIOS)
    if found_scenarios != expected_scenarios:
        raise RuntimeError(
            f"scenario mismatch: missing={sorted(expected_scenarios - found_scenarios)} "
            f"extra={sorted(found_scenarios - expected_scenarios)}"
        )
    policy_paths = sorted((root / "policy_videos").glob("*.mp4"))
    gif_paths = sorted((root / "policy_gifs").glob("*.gif"))
    if len(policy_paths) != 2 or len(gif_paths) != 2:
        raise RuntimeError(
            f"expected 2 policy MP4 and 2 policy GIF, got "
            f"{len(policy_paths)} and {len(gif_paths)}"
        )

    screenshot_path = root / "screenshots" / "best_mve_mca_m1_lvo_frame_40.png"
    records = []
    for path in scenario_paths + policy_paths + gif_paths:
        screenshot = screenshot_path if path.name == "best_mve_mca_m1_lvo.mp4" else None
        record = inspect(path, screenshot)
        if record["width"] != 1280 or record["height"] != 720:
            raise RuntimeError(f"unexpected dimensions: {record}")
        if record["pixel_std_min"] < 1.0:
            raise RuntimeError(f"blank or nearly uniform frame: {record}")
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    output = {
        "passed": True,
        "scenario_count": len(scenario_paths),
        "policy_mp4_count": len(policy_paths),
        "policy_gif_count": len(gif_paths),
        "files": records,
        "screenshot": str(screenshot_path.resolve()),
    }
    (root / "media_validation.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({key: output[key] for key in (
        "passed", "scenario_count", "policy_mp4_count", "policy_gif_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
