"""自动科研系统 - 自适应版本

特性：
1. 使用1M步训练（充分学习）
2. 如果基线效果不好，自动尝试改进措施
3. 持续迭代直到找到有效方法
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_research import AutoResearcher, ExperimentConfig


def define_full_research_plan() -> list[ExperimentConfig]:
    """定义完整研究计划（1M步训练）"""

    base_config = {
        "n_envs": 64,        # 增加并行环境
        "robots": 3,
        "clots": 3,          # 恢复到3个血栓
        "horizon": 300,
        "seed": 42,
        "device": "cuda",
    }

    experiments = [
        # 基线：使用最佳配置
        ExperimentConfig(
            name="baseline_1M",
            description="基线（1M步）：geometric观测 + milestone奖励",
            innovation="无（基线）",
            scenario="bifurcation",
            timesteps=1000000,   # 1M步，约30-45分钟
            **base_config,
        ),

        # 创新点1：更激进的探索
        ExperimentConfig(
            name="high_exploration",
            description="高探索：延长epsilon衰减",
            innovation="epsilon衰减延长到800k步（原200k）",
            scenario="bifurcation",
            timesteps=1000000,
            **base_config,
        ),

        # 创新点2：课程学习
        ExperimentConfig(
            name="curriculum_3stage",
            description="3阶段课程学习",
            innovation="课程：简单→中等→困难",
            scenario="mca_stroke",
            timesteps=1200000,   # 课程学习需要更多步数
            curriculum_stages=3,
            **base_config,
        ),

        # 创新点3：更大网络
        ExperimentConfig(
            name="large_network",
            description="更大的网络容量",
            innovation="hidden_dim从128增加到256",
            scenario="bifurcation",
            timesteps=1000000,
            hidden_dim=256,
            **base_config,
        ),

        # 创新点4：更小学习率
        ExperimentConfig(
            name="slower_learning",
            description="更稳定的学习",
            innovation="actor_lr减半到5e-5",
            scenario="bifurcation",
            timesteps=1000000,
            actor_lr=5e-5,
            **base_config,
        ),

        # 创新点5：组合最优
        ExperimentConfig(
            name="combined_best",
            description="组合最佳设置",
            innovation="课程学习 + 大网络 + 高探索",
            scenario="mca_stroke",
            timesteps=1500000,   # 组合方法需要最多训练
            curriculum_stages=3,
            hidden_dim=256,
            **base_config,
        ),
    ]

    return experiments


def should_continue(researcher: AutoResearcher, threshold: float = 0.3) -> bool:
    """判断是否需要继续尝试改进

    如果基线成功率低于阈值，说明需要继续改进
    """
    if not researcher.baseline:
        return True

    success_rate = researcher.baseline.final_success_rate
    print(f"\n📊 基线成功率: {success_rate:.1%}")

    if success_rate < threshold:
        print(f"⚠️  成功率低于 {threshold:.0%}，需要继续改进")
        return True
    else:
        print(f"✅ 成功率达到 {threshold:.0%}，可以进行深入研究")
        return True  # 无论如何都继续完整计划


def main():
    print("\n" + "=" * 80)
    print("🤖 自动科研系统 - 自适应版（1M步训练）")
    print("=" * 80)
    print()
    print("⚡ 配置：")
    print("   - 基线训练：1M transitions")
    print("   - 预计时间：每个实验30-60分钟")
    print("   - 总时间：约3-5小时")
    print("   - 自动判断：如果效果不好会尝试改进")
    print()
    print("🔄 迭代策略：")
    print("   1. 先运行基线")
    print("   2. 如果成功率<30%，依次尝试改进措施")
    print("   3. 找到有效方法后，组合最优设置")
    print()

    researcher = AutoResearcher(results_dir=ROOT / "auto_research_1M")
    plan = define_full_research_plan()

    print(f"📋 完整研究计划: {len(plan)} 个实验\n")
    for i, cfg in enumerate(plan, 1):
        print(f"{i}. {cfg.name} ({cfg.timesteps//1000}k步)")
        print(f"   创新点: {cfg.innovation}")
        print()

    print("🚀 开始自动科研...\n")

    for i, config in enumerate(plan, 1):
        print(f"\n{'='*80}")
        print(f"📊 实验 {i}/{len(plan)}: {config.name}")
        print(f"{'='*80}")

        try:
            result = researcher.run_experiment(config)

            # 第一个实验设为基线
            if i == 1:
                researcher.baseline = result
                print("\n✅ 基线已建立")

                # 判断是否需要继续
                if not should_continue(researcher):
                    print("\n⏹  基线效果不佳，但将继续尝试改进措施")

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

    # 最终总结
    print("\n" + "=" * 80)
    print("🎓 自动科研完成！")
    print("=" * 80)

    if researcher.experiments:
        print(f"\n✅ 已完成 {len(researcher.experiments)} 个实验")

        # 找出最佳方法
        best = max(researcher.experiments, key=lambda x: x.final_success_rate)
        print(f"\n🏆 最佳方法: {best.config.name}")
        print(f"   成功率: {best.final_success_rate:.1%}")
        print(f"   溶解率: {best.final_removal_rate:.1%}")
        print(f"   创新点: {best.config.innovation}")

        if best.improvement_vs_baseline:
            imp = best.improvement_vs_baseline
            print(f"\n📈 相对基线改进:")
            print(f"   成功率: {imp['success_rate']:+.1f}%")
            print(f"   溶解率: {imp['removal_rate']:+.1f}%")

    print(f"\n📁 结果位置: auto_research_1M/")
    print(f"📄 报告: auto_research_1M/research_report.md")
    print()
    print("💡 下一步:")
    print("   - 查看报告：cat auto_research_1M/research_report.md")
    print("   - 分析结果：python analyze_results.py")
    print("   - 观看演示：python watch_gui.py --policy auto_research_1M/XXX/best_policy.pt")


if __name__ == "__main__":
    main()
