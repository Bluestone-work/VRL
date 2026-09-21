"""Replay the smoke final validation to check diagnostic measurement parity."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.policy_loader import load_policy
from scripts.research_evaluate import environment_settings, episode, validate_checkpoint


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "research/runs/EXP_0001"
    smoke = run / "smoke"
    config = json.loads((smoke / "config.json").read_text())
    checkpoint = smoke / "final_policy.pt"
    validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"], config)
    reference = json.loads((smoke / "summary.json").read_text())["final_evaluation"]["episodes"]
    torch.set_num_threads(1)
    comparisons = []
    for original in reference:
        scenario = original["scenario"]
        env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario], **environment_settings(config))
        try:
            policy = load_policy(str(checkpoint), env, device="cpu")
            measured = episode(policy, env, original["episode_seed"])
            comparisons.append({"scenario": scenario, "episode_seed": original["episode_seed"],
                "success_matches": original["success"] == measured["success"],
                "steps_matches": original["steps"] == measured["steps"],
                "wall_total_matches": original["wall_hits_total"] == measured["wall_hits_total"],
                "removal_absolute_difference": abs(original["removal_rate"] - measured["removal_rate"]),
                "original": original, "measured": measured})
        finally:
            env.close()
    result = {"checkpoint": str(checkpoint), "episodes": len(comparisons),
              "all_success_matches": all(r["success_matches"] for r in comparisons),
              "all_steps_match": all(r["steps_matches"] for r in comparisons),
              "all_wall_totals_match": all(r["wall_total_matches"] for r in comparisons),
              "max_removal_absolute_difference": max(r["removal_absolute_difference"] for r in comparisons),
              "comparisons": comparisons,
              "purpose": "measurement sanity check; no training, tuning, model or test-set selection"}
    target = run / "checks/diagnostic_replay.json"
    with target.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
