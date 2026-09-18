"""Render matched improvement and remaining-failure examples, then decode assets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

from scripts.validate_study_media import inspect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="experiments/success_study_20260905")
    args = parser.parse_args()
    root = Path(args.study)
    report = json.loads((root / "reports" / "aggregate_metrics.json").read_text())
    label = report["selected_label"]
    selected = [json.loads(line) for line in (root / "heldout" / label / "episodes.jsonl").read_text().splitlines()]
    baseline_dir = root / "heldout" / "baseline_mve43"
    baseline_config = json.loads((baseline_dir / "config.json").read_text())
    baseline = {(row["scenario"], row["seed"]): row for row in (
        json.loads(line) for line in (baseline_dir / "episodes.jsonl").read_text().splitlines()
    )}
    priority = ("mca_m1_lvo", "ica_siphon", "coronary_lm_bifurcation", "coronary_rca")
    selected.sort(key=lambda row: (
        priority.index(row["scenario"]) if row["scenario"] in priority else len(priority),
        row["scenario"], row["seed"],
    ))
    improvement = next(row for row in selected if row["success"] and not baseline[(row["scenario"], row["seed"])]["success"])
    remaining_failure = next(row for row in selected if not row["success"])
    examples = [
        ("residual_success", report["selected_checkpoint"], improvement),
        ("baseline_same_episode", baseline_config["policy"], baseline[(improvement["scenario"], improvement["seed"])]),
        ("residual_failure", report["selected_checkpoint"], remaining_failure),
    ]
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.update(PYTHONPATH=".", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    manifest = []
    for name, policy, record in examples:
        stem = f"{name}_{record['scenario']}_{record['seed']}"
        video = artifacts / f"{stem}.mp4"
        command = [
            sys.executable, "make_gif.py", "--policy", policy,
            "--scenario", record["scenario"], "--seed", str(record["seed"]),
            "--robots", "5", "--clots", "3", "--horizon", "300",
            "--robot-radius", "0.0011", "--reward-mode", "milestone",
            "--views", "triple", "--wall-opacity", "0.10",
            "--max-frames", "151", "--stride", "2", "--fps", "15",
            "--width", "960", "--height", "640", "--out", str(video),
        ]
        for attempt in range(1, 5):
            log = root / "logs" / f"render_{stem}_attempt_{attempt}.log"
            with log.open("w") as stream:
                stream.write(json.dumps(command) + "\n")
                stream.flush()
                result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if result.returncode not in (-11, -6, -4, -7, 139, 134, 132, 135):
                raise RuntimeError(f"render failed: {log}")
        else:
            raise RuntimeError(f"render exhausted retries: {stem}")
        expected = f"success {bool(record['success'])}"
        if expected not in log.read_text():
            raise RuntimeError(f"rendered episode disagrees with held-out result: {log}")
        images = []
        last_frame = None
        with imageio.get_reader(video) as reader:
            for frame in reader:
                image = Image.fromarray(frame)
                image.thumbnail((640, 426))
                images.append(image.copy())
                last_frame = frame
        gif = artifacts / f"{stem}.gif"
        images[0].save(gif, save_all=True, append_images=images[1:], duration=67, loop=0)
        screenshot = artifacts / f"{stem}_final.png"
        imageio.imwrite(screenshot, last_frame)
        for path in (video, gif):
            inspected = inspect(path)
            if inspected["frames"] < 2 or inspected["pixel_std_min"] < 1 or inspected["first_last_mean_absolute_delta"] <= 0:
                raise RuntimeError(f"invalid media: {inspected}")
            manifest.append({**inspected, "example": name, "episode": record, "policy": policy})
    result = {
        "passed": True, "selected_label": label, "mp4_count": 3, "gif_count": 3,
        "selection_note": "Outcome-selected illustrations, not statistical evidence; includes a remaining failure.",
        "files": manifest,
    }
    (artifacts / "media_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("passed", "selected_label", "mp4_count", "gif_count")}))


if __name__ == "__main__":
    main()
