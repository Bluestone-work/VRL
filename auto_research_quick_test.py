"""自动科研系统 - 快速测试版

用极少的训练步数快速验证整个流程是否正常工作。
预计运行时间：5-10分钟（6个实验 × 1-2分钟）

使用方法：
    python auto_research_quick_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_research import AutoResearcher, ExperimentConfig


def define_quick_test_plan() -> list[ExperimentConfig]:
    """定义快速测试计划（每个实验只跑10k步）"""

    base_config = {
        "n_envs": 16,  # 减少并行环境
        "timesteps": 16000,  # 16 envs × 1000 steps = 16k transitions
        "robots": 3,
        "clots": 2,  # 减少血栓数
        "horizon": 200,  # 缩短horizon
        "seed": 42,
        "device": "cuda",
    }

    experiments = [
        ExperimentConfig(
            name="baseline_quick",
            description="快速基线测试",
            innovation="无（基线）",
            scenario="bifurcation",  # 最简单的场景
            **base_config,
        ),
        ExperimentConfig(
            name="curriculum_quick",
            description="快速课程学习测试",
            innovation="2阶段课程",
            scenario="bifurcation",
            curriculum_stages=2,
            **base_config,
        ),
        ExperimentConfig(
            name="gat_quick",
            description="快速GAT测试",
            innovation="图注意力网络（配置记录）",
            scenario="bifurcation",
            use_gat=True,
            **base_config,
        ),
    ]

    return experiments


def main():
    print("\n" + "=" * 80)
    print("🧪 自动科研系统 - 快速测试")
    print("=" * 80)
    print()
    print("⚠️  这是快速测试版本，每个实验只训练 16k transitions")
    print("   目的：验证整个流程是否正常工作")
    print("   预计时间：5-10分钟")
    print()

    researcher = AutoResearcher(results_dir=ROOT / "auto_research_quick_test")
    plan = define_quick_test_plan()

    print(f"📋 测试计划: {len(plan)} 个实验\n")
    for i, cfg in enumerate(plan, 1):
        print(f"{i}. {cfg.name}: {cfg.innovation}")
    print()

    response = input("开始快速测试？[Y/n]: ")
    if response.lower() == 'n':
        print("❌ 已取消")
        return

    # 运行所有实验
    for i, config in enumerate(plan, 1):
        print(f"\n{'='*80}")
        print(f"进度: {i}/{len(plan)}")
        print(f"{'='*80}")

        try:
            result = researcher.run_experiment(config)

            # 第一个实验设为基线
            if i == 1 and researcher.baseline is None:
                researcher.baseline = result
                print("\n✓ 基线已建立\n")

        except Exception as e:
            print(f"\n❌ 实验 {config.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            break

    # 生成报告
    print("\n" + "=" * 80)
    print("📊 生成测试报告...")
    print("=" * 80)

    researcher.generate_report()

    print("\n✅ 快速测试完成！")
    print(f"📁 结果位置: {researcher.results_dir}")
    print(f"📄 查看报告: {researcher.results_dir / 'research_report.md'}")
    print()
    print("如果测试成功，可以运行完整版：")
    print("    python auto_research.py")


if __name__ == "__main__":
    main()
