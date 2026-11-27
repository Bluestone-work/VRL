"""Summarize final checkpoints only, with one vote per frozen scene and training seed."""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def paired(rows, other, metric):
    vals = [(a[metric], b[metric]) for a, b in zip(rows, other)]
    if any(type(x) is not bool or type(y) is not bool for x, y in vals):
        raise ValueError(f"missing or invalid {metric}")
    wins = sum(x and not y for x, y in vals)
    losses = sum(y and not x for x, y in vals)
    return dict(n=len(vals), ippo_rate=sum(x for x, _ in vals)/len(vals),
                comparator_rate=sum(y for _, y in vals)/len(vals),
                ippo_wins=wins, ties=len(vals)-wins-losses, ippo_losses=losses,
                difference=(wins-losses)/len(vals),
                exact_mcnemar_p=binomtest(wins, wins+losses).pvalue if wins+losses else 1.0)


def report(root, manifests):
    results = {}
    for label in ("train180", "holdout100"):
        expected = json.loads((manifests / f"paired_{label}.json").read_text())
        episodes = {}
        for method in ("ppo", "a2c", "mappo"):
            run = json.loads((root / method / f"paired_{label}.json").read_text())
            rows = run["episodes"]
            if (not rows or not run["overall"]["complete"] or run["manifest_sha256"] != expected["manifest_sha256"]
                    or len(rows) != len(expected["rows"])):
                raise ValueError(f"{method}/{label}: partial or different manifest")
            for row, entry in zip(rows, expected["rows"]):
                key = (row["manifest_index"], row["scenario_hash"], row["accepted_seed"],
                       row["realized_scenario_hash"], row["navigation_targets"])
                want = (entry["manifest_index"], entry["row"]["scenario_hash"], entry["pursuit_accepted_seed"],
                        entry["pursuit_realized_scenario_hash"], entry["pursuit_navigation_targets"])
                if key != want or row["integration_failure"]:
                    raise ValueError(f"{method}/{label}: scenario mismatch or integrator failure at {key}")
            episodes[method] = rows
        results[label] = {m: {k: paired(episodes["ppo"], episodes[m], k)
                              for k in ("strict_success", "navigation_success")} for m in ("a2c", "mappo")}
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--manifests", required=True, type=Path)
    a = p.parse_args()
    output = a.root / "paired_summary.json"
    output.write_text(json.dumps(report(a.root, a.manifests), indent=2) + "\n")
    print(output)
