"""
Comparative experiment: GNN-MAPPO vs MADDPG vs MLP-MAPPO

This script runs multiple algorithms on the vascular navigation task to compare:
1. GNN-MAPPO (our implementation inspired by DGR_VDS)
2. MADDPG (baseline from the original project)
3. MLP-MAPPO (MAPPO without graph structure, for ablation)

The goal is to demonstrate the advantages of graph-based modeling for
multi-agent coordination in blood clot removal tasks.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Comparative experiment for vascular MARL")

    parser.add_argument("--robots", type=int, default=3, help="Number of robots")
    parser.add_argument("--clots", type=int, default=3, help="Number of clots")
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps per method")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-dir", type=str, default="experiments", help="Experiment directory")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip MADDPG baseline")

    return parser.parse_args()


def run_gnn_mappo(log_dir: Path, robots: int, clots: int, timesteps: int, seed: int, device: str):
    """Run GNN-MAPPO training."""
    print(f"\n{'='*80}")
    print(f"Running GNN-MAPPO (seed={seed})")
    print(f"{'='*80}\n")

    cmd = [
        "python", "scripts/train_gnn_mappo.py",
        "--robots", str(robots),
        "--clots", str(clots),
        "--timesteps", str(timesteps),
        "--use-gat",
        "--curriculum",
        "--seed", str(seed),
        "--device", device,
        "--log-dir", str(log_dir),
        "--exp-name", f"gnn_mappo_seed{seed}",
    ]

    subprocess.run(cmd, check=True)


def run_mlp_mappo(log_dir: Path, robots: int, clots: int, timesteps: int, seed: int, device: str):
    """Run MLP-MAPPO training (ablation without GAT)."""
    print(f"\n{'='*80}")
    print(f"Running MLP-MAPPO (seed={seed})")
    print(f"{'='*80}\n")

    cmd = [
        "python", "scripts/train_gnn_mappo.py",
        "--robots", str(robots),
        "--clots", str(clots),
        "--timesteps", str(timesteps),
        "--no-gat",
        "--curriculum",
        "--seed", str(seed),
        "--device", device,
        "--log-dir", str(log_dir),
        "--exp-name", f"mlp_mappo_seed{seed}",
    ]

    subprocess.run(cmd, check=True)


def run_maddpg(log_dir: Path, robots: int, clots: int, timesteps: int, seed: int, device: str):
    """Run MADDPG baseline."""
    print(f"\n{'='*80}")
    print(f"Running MADDPG (seed={seed})")
    print(f"{'='*80}\n")

    cmd = [
        "python", "scripts/train_vascular_maddpg.py",
        "--robots", str(robots),
        "--clots", str(clots),
        "--timesteps", str(timesteps),
        "--curriculum",
        "--seed", str(seed),
        "--device", device,
        "--obs-mode", "geometric",
    ]

    subprocess.run(cmd, check=True)


def load_results(exp_dir: Path) -> Dict:
    """Load experiment results."""
    results = {}

    for method_dir in exp_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        results[method_name] = []

        # Load each seed's results
        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            summary_file = seed_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    results[method_name].append(summary)

    return results


def plot_comparison(results: Dict, output_dir: Path):
    """Plot comparison of different methods."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    methods = []
    success_rates = []
    success_stds = []
    returns = []
    return_stds = []
    removal_rates = []
    removal_stds = []

    for method_name, seed_results in results.items():
        if not seed_results:
            continue

        methods.append(method_name)

        # Extract metrics
        successes = [r['final_metrics']['success'] for r in seed_results]
        rets = [r['final_metrics']['episode_return'] for r in seed_results]
        removals = [r['final_metrics']['removal_rate'] for r in seed_results]

        success_rates.append(np.mean(successes))
        success_stds.append(np.std(successes))
        returns.append(np.mean(rets))
        return_stds.append(np.std(rets))
        removal_rates.append(np.mean(removals))
        removal_stds.append(np.std(removals))

    # Plot success rate comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Success rate
    axes[0].bar(range(len(methods)), success_rates, yerr=success_stds, capsize=5)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    # Episode return
    axes[1].bar(range(len(methods)), returns, yerr=return_stds, capsize=5)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].set_ylabel('Episode Return')
    axes[1].set_title('Episode Return Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    # Removal rate
    axes[2].bar(range(len(methods)), removal_rates, yerr=removal_stds, capsize=5)
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].set_ylabel('Clot Removal Rate')
    axes[2].set_title('Clot Removal Rate Comparison')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {output_dir / 'comparison.png'}")

    # Create summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Success Rate':<20} {'Return':<20} {'Removal Rate':<20}")
    print("-"*80)

    for i, method in enumerate(methods):
        print(f"{method:<20} "
              f"{success_rates[i]:.2%} ± {success_stds[i]:.2%}    "
              f"{returns[i]:>6.1f} ± {return_stds[i]:>4.1f}    "
              f"{removal_rates[i]:.2%} ± {removal_stds[i]:.2%}")

    print("="*80)

    # Save table to file
    with open(output_dir / 'results_table.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<20} {'Success Rate':<20} {'Return':<20} {'Removal Rate':<20}\n")
        f.write("-"*80 + "\n")

        for i, method in enumerate(methods):
            f.write(f"{method:<20} "
                   f"{success_rates[i]:.2%} ± {success_stds[i]:.2%}    "
                   f"{returns[i]:>6.1f} ± {return_stds[i]:>4.1f}    "
                   f"{removal_rates[i]:.2%} ± {removal_stds[i]:.2%}\n")

        f.write("="*80 + "\n")


def main():
    args = parse_args()

    print("="*80)
    print("Comparative Experiment: GNN-MAPPO vs Baselines")
    print("="*80)
    print(f"Configuration:")
    print(f"  Robots: {args.robots}")
    print(f"  Clots: {args.clots}")
    print(f"  Timesteps per method: {args.timesteps}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print("="*80)

    exp_dir = Path(args.log_dir) / f"comparison_{int(time.time())}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment directory: {exp_dir}\n")

    # Save configuration
    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Run experiments for each method and seed
    for seed in range(args.seeds):
        print(f"\n{'#'*80}")
        print(f"# SEED {seed + 1}/{args.seeds}")
        print(f"{'#'*80}\n")

        # 1. GNN-MAPPO (main method)
        try:
            run_gnn_mappo(exp_dir, args.robots, args.clots, args.timesteps, seed, args.device)
        except Exception as e:
            print(f"Error running GNN-MAPPO: {e}")

        # 2. MLP-MAPPO (ablation)
        try:
            run_mlp_mappo(exp_dir, args.robots, args.clots, args.timesteps, seed, args.device)
        except Exception as e:
            print(f"Error running MLP-MAPPO: {e}")

        # 3. MADDPG (baseline)
        if not args.skip_baseline:
            try:
                run_maddpg(exp_dir, args.robots, args.clots, args.timesteps, seed, args.device)
            except Exception as e:
                print(f"Error running MADDPG: {e}")

    # Analyze and plot results
    print("\n" + "="*80)
    print("All experiments completed. Analyzing results...")
    print("="*80)

    results = load_results(exp_dir)
    plot_comparison(results, exp_dir)

    print(f"\n✓ All results saved to {exp_dir}")


if __name__ == "__main__":
    main()
