"""手动分析自动科研结果"""

import json
import numpy as np
from pathlib import Path

def analyze_experiment(exp_dir):
    """分析单个实验"""
    log_file = exp_dir / "training_log.jsonl"

    if not log_file.exists():
        return None

    episodes = []
    with open(log_file) as f:
        for line in f:
            episodes.append(json.loads(line))

    if not episodes:
        return None

    # 最后100个episode的统计
    recent = episodes[-100:]

    result = {
        "name": exp_dir.name,
        "total_episodes": len(episodes),
        "final_success_rate": np.mean([ep["success"] for ep in recent]),
        "final_removal_rate": np.mean([ep["removal_rate"] for ep in recent]),
        "final_return": np.mean([ep["return"] for ep in recent]),
        "contact_miss_rate": np.mean([ep.get("contact_miss", False) for ep in recent]),
        "wall_collisions": np.mean([ep["wall_collisions"] for ep in recent]),
        "peak_success_rate": max([ep["success"] for ep in episodes]),
        "episodes": episodes,
    }

    return result

def main():
    results_dir = Path("auto_research_results")

    print("=" * 80)
    print("🔍 自动科研结果分析")
    print("=" * 80)
    print()

    experiments = []
    for exp_dir in sorted(results_dir.glob("*_*/")):
        result = analyze_experiment(exp_dir)
        if result:
            experiments.append(result)

    if not experiments:
        print("❌ 没有找到实验结果")
        return

    # 打印对比表
    print("📊 实验对比\n")
    print(f"{'实验':<25} {'Episode数':<10} {'成功率':<10} {'溶解率':<10} {'回报':<10} {'未接触率':<10}")
    print("-" * 80)

    for exp in experiments:
        print(f"{exp['name']:<25} "
              f"{exp['total_episodes']:<10} "
              f"{exp['final_success_rate']:<10.1%} "
              f"{exp['final_removal_rate']:<10.1%} "
              f"{exp['final_return']:<+10.2f} "
              f"{exp['contact_miss_rate']:<10.1%}")

    print()

    # 详细分析
    print("=" * 80)
    print("📈 详细分析")
    print("=" * 80)
    print()

    for exp in experiments:
        print(f"\n### {exp['name']}")
        print(f"  总Episode数: {exp['total_episodes']}")
        print(f"  最终成功率: {exp['final_success_rate']:.1%}")
        print(f"  最终溶解率: {exp['final_removal_rate']:.1%}")
        print(f"  最终回报: {exp['final_return']:+.2f}")
        print(f"  峰值成功率: {exp['peak_success_rate']}")
        print(f"  接触失败率: {exp['contact_miss_rate']:.1%}")
        print(f"  平均碰壁: {exp['wall_collisions']:.1f}")

        # 学习趋势
        episodes = exp['episodes']
        if len(episodes) >= 100:
            early = episodes[:100]
            late = episodes[-100:]

            early_success = np.mean([ep["success"] for ep in early])
            late_success = np.mean([ep["success"] for ep in late])

            print(f"  学习趋势: {early_success:.1%} → {late_success:.1%} "
                  f"({'✓ 提升' if late_success > early_success else '✗ 未提升'})")

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
