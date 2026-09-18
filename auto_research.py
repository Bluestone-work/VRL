"""自动科研系统：基线 → 创新迭代 → 性能追踪

工作流程：
1. 建立基线（当前最佳配置）
2. 依次添加创新点
3. 每次实验记录：
   - 配置参数
   - 训练曲线
   - 最终性能指标
   - 改进幅度
4. 生成对比报告

所有结果保存到 auto_research_results/ 目录。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str
    innovation: str  # 创新点描述

    # 训练参数
    scenario: str = "mca_stroke"
    n_envs: int = 64
    robots: int = 3
    clots: int = 3
    horizon: int = 300
    timesteps: int = 500000
    obs_mode: str = "geometric"
    reward_mode: str = "milestone"

    # 网络参数
    hidden_dim: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    # 创新点特定参数
    use_gat: bool = False           # 图注意力网络
    use_communication: bool = False  # 智能体通信
    use_hierarchical: bool = False   # 层次化策略
    curriculum_stages: int = 1       # 课程学习阶段数

    # 其他
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    start_time: str
    end_time: str
    duration_seconds: float

    # 性能指标
    final_success_rate: float
    final_removal_rate: float
    final_return: float
    first_contact_step: float
    wall_collision_rate: float
    robot_collision_rate: float

    # 学习曲线统计
    peak_success_rate: float
    convergence_step: int  # 达到90%最优性能的步数

    # 文件路径
    log_dir: str
    checkpoint_path: str
    training_curve_json: str

    # 相对基线的改进
    improvement_vs_baseline: dict[str, float] | None = None


class AutoResearcher:
    """自动科研系统"""

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or ROOT / "auto_research_results"
        self.results_dir.mkdir(exist_ok=True)

        self.experiments: list[ExperimentResult] = []
        self.baseline: ExperimentResult | None = None

        # 加载历史记录
        self._load_history()

    def _load_history(self):
        """加载历史实验记录"""
        history_file = self.results_dir / "experiment_history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                if data.get("baseline"):
                    self.baseline = self._dict_to_result(data["baseline"])
                self.experiments = [
                    self._dict_to_result(exp) for exp in data.get("experiments", [])
                ]
            print(f"✓ 加载了 {len(self.experiments)} 个历史实验记录")

    def _save_history(self):
        """保存实验记录"""
        history_file = self.results_dir / "experiment_history.json"
        data = {
            "baseline": asdict(self.baseline) if self.baseline else None,
            "experiments": [asdict(exp) for exp in self.experiments],
            "last_updated": datetime.now().isoformat(),
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _dict_to_result(self, d: dict) -> ExperimentResult:
        """字典转ExperimentResult"""
        cfg_dict = d.pop("config")
        config = ExperimentConfig(**cfg_dict)
        return ExperimentResult(config=config, **d)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """运行一个实验"""
        print("\n" + "=" * 80)
        print(f"🚀 实验: {config.name}")
        print(f"📝 描述: {config.description}")
        print(f"💡 创新点: {config.innovation}")
        print("=" * 80)

        # 创建实验目录
        exp_dir = self.results_dir / f"{len(self.experiments):03d}_{config.name}"
        exp_dir.mkdir(exist_ok=True)

        # 保存配置
        with open(exp_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)

        start_time = datetime.now()

        # 构建训练命令
        cmd = self._build_training_command(config, exp_dir)

        print(f"\n▶ 运行命令:")
        print(f"  {' '.join(cmd)}\n")

        # 运行训练
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=7200,  # 2小时超时
            )

            if result.returncode != 0:
                print(f"❌ 训练失败:")
                print(result.stderr)
                raise RuntimeError(f"Training failed with code {result.returncode}")

            # 保存训练日志
            with open(exp_dir / "train_stdout.log", "w") as f:
                f.write(result.stdout)
            with open(exp_dir / "train_stderr.log", "w") as f:
                f.write(result.stderr)

        except subprocess.TimeoutExpired:
            print("⏱ 训练超时（2小时）")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 解析结果
        metrics = self._parse_training_output(exp_dir)

        result = ExperimentResult(
            config=config,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            **metrics,
            log_dir=str(exp_dir),
        )

        # 计算相对基线的改进
        if self.baseline is not None and config.name != "baseline":
            result.improvement_vs_baseline = self._compute_improvement(result)

        # 保存结果
        with open(exp_dir / "result.json", "w") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        self.experiments.append(result)
        self._save_history()

        # 打印结果摘要
        self._print_result_summary(result)

        return result

    def _build_training_command(self, config: ExperimentConfig, exp_dir: Path) -> list[str]:
        """构建训练命令"""
        cmd = [
            sys.executable,
            "scripts/train_vector_enhanced.py",
            "--scenario", config.scenario,
            "--n-envs", str(config.n_envs),
            "--robots", str(config.robots),
            "--clots", str(config.clots),
            "--horizon", str(config.horizon),
            "--timesteps", str(config.timesteps),
            "--obs-mode", config.obs_mode,
            "--reward-mode", config.reward_mode,
            "--hidden-dim", str(config.hidden_dim),
            "--actor-lr", str(config.actor_lr),
            "--critic-lr", str(config.critic_lr),
            "--seed", str(config.seed),
            "--device", config.device,
            "--logdir", str(exp_dir / "logs"),
            "--save-path", str(exp_dir / "best_policy.pt"),
        ]

        # 添加创新点特定参数
        if config.use_gat:
            cmd.extend(["--use-gat"])
        if config.use_communication:
            cmd.extend(["--use-communication"])
        if config.use_hierarchical:
            cmd.extend(["--use-hierarchical"])
        if config.curriculum_stages > 1:
            cmd.extend(["--curriculum", "--curriculum-stages", str(config.curriculum_stages)])

        return cmd

    def _parse_training_output(self, exp_dir: Path) -> dict[str, Any]:
        """从训练输出解析性能指标"""
        # 查找日志文件
        log_dirs = list((exp_dir / "logs").glob("**/training_log.jsonl")) if (exp_dir / "logs").exists() else []

        if not log_dirs:
            # 尝试从stdout解析
            return self._parse_from_stdout(exp_dir / "train_stdout.log")

        log_file = log_dirs[0]

        # 读取训练日志
        episodes = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

        if not episodes:
            return self._default_metrics()

        # 提取最后100个episode的统计
        recent = episodes[-100:]

        metrics = {
            "final_success_rate": float(np.mean([ep.get("success", 0) for ep in recent])),
            "final_removal_rate": float(np.mean([ep.get("removal_rate", 0) for ep in recent])),
            "final_return": float(np.mean([ep.get("return", 0) for ep in recent])),
            "first_contact_step": float(np.mean([ep.get("first_contact_step", 300) for ep in recent if ep.get("first_contact_step", -1) >= 0])) if any(ep.get("first_contact_step", -1) >= 0 for ep in recent) else 300.0,
            "wall_collision_rate": float(np.mean([ep.get("wall_collisions", 0) / max(ep.get("steps", 1), 1) for ep in recent])),
            "robot_collision_rate": float(np.mean([ep.get("collision_rate", 0) for ep in recent])),
        }

        # 计算峰值和收敛步数
        success_rates = [ep.get("success", 0) for ep in episodes]
        metrics["peak_success_rate"] = float(max(success_rates)) if success_rates else 0.0

        # 收敛：达到峰值90%的步数
        if metrics["peak_success_rate"] > 0:
            threshold = 0.9 * metrics["peak_success_rate"]
            for i, sr in enumerate(success_rates):
                if sr >= threshold:
                    metrics["convergence_step"] = i * 100  # 假设每100步记录一次
                    break
            else:
                metrics["convergence_step"] = len(episodes) * 100
        else:
            metrics["convergence_step"] = len(episodes) * 100

        # 文件路径
        metrics["checkpoint_path"] = str(exp_dir / "best_policy.pt")
        metrics["training_curve_json"] = str(log_file)

        return metrics

    def _parse_from_stdout(self, stdout_file: Path) -> dict[str, Any]:
        """从stdout日志解析（备用方案）"""
        if not stdout_file.exists():
            return self._default_metrics()

        with open(stdout_file) as f:
            content = f.read()

        # 简单的正则匹配提取最终指标
        import re

        metrics = self._default_metrics()

        # 查找最后一次报告的指标
        patterns = {
            "final_success_rate": r"success[_\s]*rate[:\s]*([\d.]+)",
            "final_removal_rate": r"removal[_\s]*rate[:\s]*([\d.]+)",
            "final_return": r"return[:\s]*([+-]?[\d.]+)",
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    metrics[key] = float(matches[-1])
                except:
                    pass

        return metrics

    def _default_metrics(self) -> dict[str, Any]:
        """默认指标（训练失败时）"""
        return {
            "final_success_rate": 0.0,
            "final_removal_rate": 0.0,
            "final_return": 0.0,
            "first_contact_step": 300.0,
            "wall_collision_rate": 0.0,
            "robot_collision_rate": 0.0,
            "peak_success_rate": 0.0,
            "convergence_step": 999999,
            "checkpoint_path": "",
            "training_curve_json": "",
        }

    def _compute_improvement(self, result: ExperimentResult) -> dict[str, float]:
        """计算相对基线的改进百分比"""
        baseline = self.baseline
        return {
            "success_rate": 100 * (result.final_success_rate - baseline.final_success_rate) / max(baseline.final_success_rate, 1e-6),
            "removal_rate": 100 * (result.final_removal_rate - baseline.final_removal_rate) / max(baseline.final_removal_rate, 1e-6),
            "return": 100 * (result.final_return - baseline.final_return) / max(abs(baseline.final_return), 1e-6),
            "convergence_speed": 100 * (baseline.convergence_step - result.convergence_step) / max(baseline.convergence_step, 1),
        }

    def _print_result_summary(self, result: ExperimentResult):
        """打印结果摘要"""
        print("\n" + "=" * 80)
        print(f"✅ 实验完成: {result.config.name}")
        print("=" * 80)
        print(f"⏱  训练时间: {result.duration_seconds/60:.1f} 分钟")
        print(f"📊 最终指标:")
        print(f"   - 成功率: {result.final_success_rate:.1%}")
        print(f"   - 溶解率: {result.final_removal_rate:.1%}")
        print(f"   - 回报: {result.final_return:+.2f}")
        print(f"   - 首次接触步数: {result.first_contact_step:.1f}")
        print(f"   - 收敛步数: {result.convergence_step}")

        if result.improvement_vs_baseline:
            print(f"\n📈 相对基线改进:")
            imp = result.improvement_vs_baseline
            print(f"   - 成功率: {imp['success_rate']:+.1f}%")
            print(f"   - 溶解率: {imp['removal_rate']:+.1f}%")
            print(f"   - 回报: {imp['return']:+.1f}%")
            print(f"   - 收敛速度: {imp['convergence_speed']:+.1f}%")

        print(f"\n💾 结果保存到: {result.log_dir}")
        print("=" * 80 + "\n")

    def generate_report(self):
        """生成对比报告"""
        if not self.experiments:
            print("⚠ 没有实验记录")
            return

        report_file = self.results_dir / "research_report.md"

        with open(report_file, "w") as f:
            f.write("# 自动科研报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 基线
            if self.baseline:
                f.write("## 基线实验\n\n")
                self._write_experiment_section(f, self.baseline)

            # 所有实验对比表
            f.write("## 实验对比\n\n")
            f.write("| 实验 | 创新点 | 成功率 | 溶解率 | 回报 | 收敛步数 | 改进(成功率) |\n")
            f.write("|------|--------|--------|--------|------|----------|-------------|\n")

            for exp in self.experiments:
                improvement = ""
                if exp.improvement_vs_baseline:
                    imp = exp.improvement_vs_baseline["success_rate"]
                    improvement = f"{imp:+.1f}%"

                f.write(f"| {exp.config.name} | {exp.config.innovation} | "
                       f"{exp.final_success_rate:.1%} | {exp.final_removal_rate:.1%} | "
                       f"{exp.final_return:+.2f} | {exp.convergence_step} | {improvement} |\n")

            # 详细结果
            f.write("\n## 详细结果\n\n")
            for exp in self.experiments:
                f.write(f"### {exp.config.name}\n\n")
                self._write_experiment_section(f, exp)

        print(f"✓ 报告已生成: {report_file}")

    def _write_experiment_section(self, f, result: ExperimentResult):
        """写入单个实验的详细信息"""
        f.write(f"**描述**: {result.config.description}\n\n")
        f.write(f"**创新点**: {result.config.innovation}\n\n")
        f.write(f"**配置**:\n")
        f.write(f"- 场景: {result.config.scenario}\n")
        f.write(f"- 智能体: {result.config.robots}\n")
        f.write(f"- 血栓: {result.config.clots}\n")
        f.write(f"- 训练步数: {result.config.timesteps}\n")
        f.write(f"- 并行环境: {result.config.n_envs}\n\n")

        f.write(f"**性能指标**:\n")
        f.write(f"- 成功率: {result.final_success_rate:.1%}\n")
        f.write(f"- 溶解率: {result.final_removal_rate:.1%}\n")
        f.write(f"- 平均回报: {result.final_return:+.2f}\n")
        f.write(f"- 首次接触: {result.first_contact_step:.1f} 步\n")
        f.write(f"- 收敛步数: {result.convergence_step}\n")
        f.write(f"- 训练时间: {result.duration_seconds/60:.1f} 分钟\n\n")

        if result.improvement_vs_baseline:
            f.write(f"**相对基线改进**:\n")
            imp = result.improvement_vs_baseline
            f.write(f"- 成功率: {imp['success_rate']:+.1f}%\n")
            f.write(f"- 溶解率: {imp['removal_rate']:+.1f}%\n")
            f.write(f"- 回报: {imp['return']:+.1f}%\n")
            f.write(f"- 收敛速度: {imp['convergence_speed']:+.1f}%\n\n")

        f.write(f"**结果位置**: `{result.log_dir}`\n\n")
        f.write("---\n\n")


def define_research_plan() -> list[ExperimentConfig]:
    """定义研究计划"""

    # 基线：当前最佳配置
    baseline = ExperimentConfig(
        name="baseline",
        description="当前最佳配置：geometric观测 + milestone奖励 + MCA场景",
        innovation="无（基线）",
        scenario="mca_stroke",
        timesteps=500000,
        curriculum_stages=1,
    )

    # 创新点1：课程学习
    curriculum = ExperimentConfig(
        name="curriculum_learning",
        description="多阶段课程学习：从简单到复杂场景",
        innovation="3阶段课程：bifurcation(1血栓) → multilevel(2血栓) → mca_stroke(3血栓)",
        scenario="mca_stroke",
        timesteps=500000,
        curriculum_stages=3,
    )

    # 创新点2：图注意力网络
    gat = ExperimentConfig(
        name="graph_attention",
        description="使用GAT替代MLP处理智能体关系",
        innovation="图注意力网络：自适应感受野 + 排列不变性",
        scenario="mca_stroke",
        timesteps=500000,
        use_gat=True,
    )

    # 创新点3：智能体通信
    comm = ExperimentConfig(
        name="communication",
        description="智能体间显式通信机制",
        innovation="通信协议：意图广播 + 邻居响应",
        scenario="mca_stroke",
        timesteps=500000,
        use_communication=True,
    )

    # 创新点4：层次化策略
    hierarchical = ExperimentConfig(
        name="hierarchical",
        description="层次化多智能体：高层分配 + 低层执行",
        innovation="两层架构：目标分配器 + 子群协调器",
        scenario="mca_stroke",
        timesteps=500000,
        use_hierarchical=True,
    )

    # 创新点5：组合最优
    combined = ExperimentConfig(
        name="combined_best",
        description="组合表现最好的创新点",
        innovation="课程学习 + GAT + 通信（假设这三个最优）",
        scenario="mca_stroke",
        timesteps=600000,
        curriculum_stages=3,
        use_gat=True,
        use_communication=True,
    )

    return [baseline, curriculum, gat, comm, hierarchical, combined]


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🤖 自动科研系统")
    print("=" * 80)
    print()

    researcher = AutoResearcher()

    # 定义研究计划
    plan = define_research_plan()

    print(f"📋 研究计划: {len(plan)} 个实验\n")
    for i, cfg in enumerate(plan, 1):
        print(f"{i}. {cfg.name}: {cfg.innovation}")
    print()

    # 询问确认
    response = input("是否开始自动科研？这将运行所有实验（预计数小时）[y/N]: ")
    if response.lower() != 'y':
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

            # 询问是否继续
            cont = input("\n是否继续下一个实验？[y/N]: ")
            if cont.lower() != 'y':
                break

    # 生成最终报告
    print("\n" + "=" * 80)
    print("📊 生成最终报告...")
    print("=" * 80)

    researcher.generate_report()

    print("\n✅ 自动科研完成！")
    print(f"📁 所有结果保存在: {researcher.results_dir}")
    print(f"📄 查看报告: {researcher.results_dir / 'research_report.md'}")


if __name__ == "__main__":
    main()
