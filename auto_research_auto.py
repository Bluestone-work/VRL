"""自动科研系统 - 非交互式版本

直接开始运行，无需确认。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_research import AutoResearcher, ExperimentConfig


def define_quick_test_plan() -> list[ExperimentConfig]:
    """定义快速测试计划"""

    base_config = {
        "n_envs": 16,
        "timesteps": 16000,
        "robots": 3,
        "clots": 2,
        "horizon": 200,
        "seed": 42,
        "device": "cuda",
    }

    experiments = [
        ExperimentConfig(
            name="baseline_quick",
            description="快速基线测试",
            innovation="无（基线）",
            scenario="bifurcation",
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
    ]

    return experiments


def main():
    print("\n" + "=" * 80)
    print("🤖 自动科研系统启动")
    print("=" * 80)
    print()
    print("⚡ 快速测试模式（每个实验 16k transitions）")
    print("📊 预计时间：5-10分钟")
    print()

    researcher = AutoResearcher(results_dir=ROOT / "auto_research_quick_test")
    plan = define_quick_test_plan()

    print(f"📋 测试计划: {len(plan)} 个实验\n")
    for i, cfg in enumerate(plan, 1):
        print(f"{i}. {cfg.name}: {cfg.innovation}")
    print()

    # 直接开始，无需确认
    for i, config in enumerate(plan, 1):
        print(f"\n{'='*80}")
        print(f"🚀 实验 {i}/{len(plan)}: {config.name}")
        print(f"{'='*80}")

        try:
            result = researcher.run_experiment(config)

            if i == 1 and researcher.baseline is None:
                researcher.baseline = result
                print("\n✅ 基线已建立\n")

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            break
        except Exception as e:
            print(f"\n❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            # 继续下一个实验
            continue

    # 生成报告
    print("\n" + "=" * 80)
    print("📊 生成报告...")
    print("=" * 80)

    researcher.generate_report()

    print("\n✅ 测试完成！")
    print(f"📁 结果: {researcher.results_dir}")
    print(f"📄 报告: {researcher.results_dir / 'research_report.md'}")


if __name__ == "__main__":
    main()
