#!/bin/bash
# Batch experiment runner for architecture comparison
# Usage: bash scripts/run_experiments.sh

set -e

DEVICE=${1:-cuda}
BASE_STEPS=200000
SEEDS=3

echo "=================================="
echo "MARL Architecture Comparison Suite"
echo "=================================="
echo "Device: $DEVICE"
echo "Base timesteps: $BASE_STEPS"
echo "Seeds: $SEEDS"
echo "=================================="

# Create results directory
RESULTS_DIR="experiments/architecture_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/experiments.log"
echo "Experiment log - $(date)" > "$LOG_FILE"

# Function to run one experiment
run_experiment() {
    local name=$1
    shift
    local args="$@"

    echo ""
    echo "===================================="
    echo "Running: $name"
    echo "Args: $args"
    echo "===================================="
    echo "" | tee -a "$LOG_FILE"

    for seed in $(seq 1 $SEEDS); do
        echo "[Seed $seed/$SEEDS] $name" | tee -a "$LOG_FILE"

        PYTHONPATH=. python scripts/train_advanced.py \
            $args \
            --seed $((42 + seed)) \
            --device $DEVICE \
            --log-dir "$RESULTS_DIR" \
            2>&1 | tee -a "$LOG_FILE"

        echo "✓ Completed seed $seed" | tee -a "$LOG_FILE"
    done
}

# ============================================================================
# Experiment 1: Architecture Comparison (Core)
# ============================================================================
echo ""
echo "############################################"
echo "# EXPERIMENT 1: Architecture Comparison"
echo "############################################"

# 1.1 Standard GAT (baseline)
run_experiment "GAT-Standard" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 1.2 Edge Feature GAT
run_experiment "GAT-EdgeFeatures" \
    --architecture edge_gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 1.3 Transformer (full attention)
run_experiment "Transformer-Full" \
    --architecture transformer \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 1.4 Sparse Transformer
run_experiment "Transformer-Sparse-k8" \
    --architecture sparse_transformer \
    --k-neighbors 8 \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 1.5 Hierarchical GNN
run_experiment "HierarchicalGNN" \
    --architecture hierarchical_gnn \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 1.6 MLP (ablation floor)
run_experiment "MLP-Baseline" \
    --architecture mlp \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# ============================================================================
# Experiment 2: Scalability (Large Swarms)
# ============================================================================
echo ""
echo "############################################"
echo "# EXPERIMENT 2: Scalability"
echo "############################################"

# 2.1 GAT with 12 robots
run_experiment "GAT-12robots" \
    --architecture gat \
    --robots 12 --clots 6 --timesteps $((BASE_STEPS * 2))

# 2.2 Sparse Transformer with 12 robots
run_experiment "SparseTransformer-12robots" \
    --architecture sparse_transformer \
    --k-neighbors 8 \
    --robots 12 --clots 6 --timesteps $((BASE_STEPS * 2))

# ============================================================================
# Experiment 3: Adaptive Curriculum
# ============================================================================
echo ""
echo "############################################"
echo "# EXPERIMENT 3: Adaptive Curriculum"
echo "############################################"

# 3.1 Fixed curriculum (baseline from original)
run_experiment "Curriculum-Fixed" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 3.2 Adaptive curriculum
run_experiment "Curriculum-Adaptive" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS \
    --adaptive-curriculum

# ============================================================================
# Experiment 4: Pre-training
# ============================================================================
echo ""
echo "############################################"
echo "# EXPERIMENT 4: Pre-training Effect"
echo "############################################"

# 4.1 No pre-training
run_experiment "No-Pretrain" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 4.2 With pre-training
run_experiment "With-Pretrain" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS \
    --pretrain

# ============================================================================
# Experiment 5: Exploration Bonus
# ============================================================================
echo ""
echo "############################################"
echo "# EXPERIMENT 5: Exploration Bonus"
echo "############################################"

# 5.1 No exploration bonus
run_experiment "No-Exploration" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS

# 5.2 With exploration bonus
run_experiment "Exploration-0.01" \
    --architecture gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS \
    --exploration-bonus 0.01

# ============================================================================
# Experiment 6: Combined Best Methods
# ============================================================================
echo ""
echo "############################################"
echo "# EXPERIMENT 6: Combined Best Methods"
echo "############################################"

# 6.1 EdgeGAT + Adaptive + Exploration
run_experiment "EdgeGAT-Full-Pipeline" \
    --architecture edge_gat \
    --robots 5 --clots 3 --timesteps $BASE_STEPS \
    --adaptive-curriculum \
    --exploration-bonus 0.01

# 6.2 Transformer + Pre-training + Adaptive
run_experiment "Transformer-Full-Pipeline" \
    --architecture transformer \
    --robots 5 --clots 3 --timesteps $BASE_STEPS \
    --pretrain \
    --adaptive-curriculum

# ============================================================================
# Analysis and Visualization
# ============================================================================
echo ""
echo "############################################"
echo "# Generating Analysis"
echo "############################################"

# Create analysis script
cat > "$RESULTS_DIR/analyze.py" << 'EOF'
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

results_dir = Path(__file__).parent
experiments = {}

# Load all results
for exp_dir in results_dir.glob("*/seed_*/summary.json"):
    with open(exp_dir) as f:
        data = json.load(f)
        exp_name = exp_dir.parent.parent.name
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(data)

# Compute statistics
print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"{'Experiment':<40} {'Success Rate':<20} {'Final Return':<20}")
print("-"*80)

for exp_name, runs in sorted(experiments.items()):
    successes = [r['final_metrics']['success'] for r in runs]
    returns = [r['final_metrics']['episode_return'] for r in runs]

    print(f"{exp_name:<40} {np.mean(successes):.2%} ± {np.std(successes):.2%}   "
          f"{np.mean(returns):>6.1f} ± {np.std(returns):>4.1f}")

print("="*80)

# Save to file
with open(results_dir / "results_summary.txt", "w") as f:
    f.write("EXPERIMENT RESULTS SUMMARY\n")
    f.write("="*80 + "\n")
    for exp_name, runs in sorted(experiments.items()):
        successes = [r['final_metrics']['success'] for r in runs]
        returns = [r['final_metrics']['episode_return'] for r in runs]
        f.write(f"{exp_name}: success={np.mean(successes):.2%}±{np.std(successes):.2%}, "
                f"return={np.mean(returns):.1f}±{np.std(returns):.1f}\n")

print(f"\nResults saved to: {results_dir / 'results_summary.txt'}")
EOF

# Run analysis
echo "Running analysis..."
PYTHONPATH=. python "$RESULTS_DIR/analyze.py"

# ============================================================================
# Done
# ============================================================================
echo ""
echo "============================================"
echo "✓ ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo "Results directory: $RESULTS_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "1. Review results: cat $RESULTS_DIR/results_summary.txt"
echo "2. Plot learning curves: python scripts/plot_results.py $RESULTS_DIR"
echo "3. Generate paper figures: python scripts/make_figures.py $RESULTS_DIR"
echo "============================================"
