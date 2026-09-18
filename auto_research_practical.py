"""自动科研系统 - 实用版本

使用合理的训练步数，可以看到真实的学习效果。
- 基线和创新点各训练 200k-300k steps
- 预计每个实验 10-15 分钟（GPU）
- 总时间约 1-2 小时
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_research import AutoResearcher, ExperimentConfig


def define_practical_plan() -> list[ExperimentConfig]:
    """定义实用研究计划"""

    base_config = {
        "n_envs": 32,        # 32个并行环境（GPU友好）
        "robots": 3,
        "clots": 2,          # 2个血栓（稍简单，便于快速看到结果）
        "horizon": 250,
        "seed": 42,
        "device": "cuda",
    }

    experiments = [
        # 基线：当前最佳配置
        ExperimentConfig(
            name="baseline",
            description="基线：geometric观测 + milestone奖励",
            innovation="无（基线）",
            scenario="bifurcation",  # 先用简单场景建立基线
            timesteps=200000,        # 200k steps，约10-15分钟
            **base_config,
        ),

        # 创新点1：课程学习
        ExperimentConfig(
            name="curriculum_2stage",
            description="2阶段课程学习",
            innovation="课程：bifurcation(1血栓) → bifurcation(2血栓)",
            scenario="bifurcation",
            timesteps=200000,
            curriculum_stages=2,
            **base_config,
        ),

        # 创新点2：更复杂场景
        ExperimentConfig(
            name="complex_scene",
            description="复杂场景：MCA中风",
            innovation="MCA场景（解剖学真实）",
            scenario="mca_stroke",
            timesteps=250000,        # 更难的场景需要更多训练
            **base_config,
        ),

        # 创新点3：课程学习 + 复杂场景
        ExperimentConfig(
            name="curriculum_mca",
            description="课程学习到MCA场景",
            innovation="2阶段课程：bifurcation → mca_stroke",
            scenario="mca_stroke",
            timesteps=250000,
            curriculum_stages=2,
            **base_config,
        ),
    ]

    return experiments


def main():
    print("\n" + "=" * 80)
    print("🤖 自动科研系统 - 实用版")
    print("=" * 80)
    print()
    print("⚡ 配置：")
    print("   - 每个实验：200k-250k transitions")
    print("   - 预计时间：10-15分钟/实验（GPU）")
    print("   - 总时间：约1-2小时")
    print()

    researcher = AutoResearcher(results_dir=ROOT / "auto_research_results")
    plan = define_practical_plan()

    print(f"📋 研究计划: {len(plan)} 个实验\n")
    for i, cfg in enumerate(plan, 1):
        print(f"{i}. {cfg.name}")
        print(f"   创新点: {cfg.innovation}")
        print(f"   场景: {cfg.scenario}, 训练: {cfg.timesteps} steps")
        print()

    # 直接开始
    print("🚀 开始实验...\n")

    for i, config in enumerate(plan, 1):
        print(f"\n{'='*80}")
        print(f"📊 实验 {i}/{len(plan)}: {config.name}")
        print(f"{'='*80}")

        try:
            result = researcher.run_experiment(config)

            # 第一个实验设为基线
            if i == 1 and researcher.baseline is None:
                researcher.baseline = result
                print("\n✅ 基线已建立")

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            print(f"已完成 {i-1}/{len(plan)} 个实验")
            break
        except Exception as e:
            print(f"\n❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            # 继续下一个实验
            continue

    # 生成报告
    print("\n" + "=" * 80)
    print("📊 生成最终报告...")
    print("=" * 80)

    researcher.generate_report()

    print("\n✅ 自动科研完成！")
    print(f"📁 结果位置: {researcher.results_dir}")
    print(f"📄 报告: {researcher.results_dir / 'research_report.md'}")
    print()
    print("💡 提示：")
    print("   - 查看报告：cat auto_research_results/research_report.md")
    print("   - 观看最佳策略：python watch_gui.py --policy auto_research_results/XXX/best_policy.pt")
    print("   - 训练日志：auto_research_results/XXX/training_log.jsonl")


if __name__ == "__main__":
    main()
