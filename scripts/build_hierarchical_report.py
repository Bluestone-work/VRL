"""Build an iteration report for the hierarchical allocation research line."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("experiments/hierarchical_marl_research_20260906")


def read_summary(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    summary_paths = [
        ROOT / "evaluation/matrix/nearest_flow/summary.json",
        ROOT / "evaluation/matrix/hungarian_flow/summary.json",
        ROOT / "evaluation/matrix/learned_allocator/summary.json",
        ROOT / "evaluation/matrix_v2/learned_allocator/summary.json",
        ROOT / "evaluation/matrix_v3_seed46/learned_allocator/summary.json",
        ROOT / "evaluation/matrix_v3_seed47/learned_allocator/summary.json",
        ROOT / "evaluation/stratified/hungarian_flow/summary.json",
    ]
    for path in summary_paths:
        data = read_summary(path)
        if data is None:
            continue
        rows.append((path.parent.parent.name, path.parent.name, data))

    lines = [
        "# 分层多机器人任务分配实验迭代报告",
        "",
        f"> 自动生成时间：{datetime.now().astimezone().isoformat()}",
        "> 本报告记录第一轮任务分配覆盖层、教师分配迭代和分散初始化测试，不覆盖原有 full matrix 结果。",
        "",
        "## 1. 当前实现",
        "",
        "- 旧环境默认行为保持不变：未设置任务分配时仍使用最近活动血栓。",
        "- 新模式通过 `set_task_assignments()` 写入显式目标，低层继续使用现有 `flow_guided` 控制。",
        "- Hungarian 版本先保证不同血栓被覆盖，再把剩余机器人按边际代价分配为辅助机器人。",
        "- 学习版本使用高层 PPO，每 20 步分配一次，低层每步执行运动控制。",
        "",
        "## 2. 统一评测结果",
        "",
        "| 评测目录 | 方法 | 回合 | 成功率 | 平均步数 | 平均清除率 | 任务切换 | 撞壁 | 机器人碰撞 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for matrix, method, data in rows:
        lines.append(
            f"| `{matrix}` | `{method}` | {data['episodes']} | "
            f"{data['success_rate']:.2%} | {data['mean_steps']:.1f} | "
            f"{data['mean_removal_rate']:.2%} | {data['mean_task_switches']:.1f} | "
            f"{data['mean_wall_hits']:.1f} | {data['mean_robot_collisions']:.1f} |"
        )

    lines.extend([
        "",
        "## 3. 第一轮结论",
        "",
        "1. Hungarian 分配在 legacy 初始化下优于最近血栓规则：重复追踪减少，平均完成步数和机器人碰撞下降。",
        "2. 第一版学习分配器没有超过 Hungarian，说明仅依靠原始环境回报学习高层任务分配不够稳定。",
        "3. v2 加入 Hungarian 教师和并行覆盖奖励后训练更稳定，但当前仍未超过规则分配基线。",
        "4. 分支分散初始化在简单场景有效，但在部分解剖场景触发大量撞壁，说明低层导航需要先适应新的出生分布。",
        "5. v3 接入原有 `flow_guided_43.pt` 低层策略后，seed 46/47 的统一测试明显改善，说明此前主要瓶颈是高层分配和低层控制耦合，而不是高层网络容量本身。",
        "6. 当前学习分配仍使用 Hungarian 教师和独立 categorical 选择，尚未实现严格的一对一可微匹配；后续应增加 Sinkhorn/auction 约束并单独报告重复目标比例。",
        "",
        "## 4. 媒体与日志",
        "",
        "- 慢速媒体根目录：`media_slow/`",
        "- 快速媒体根目录：`media_fast/`",
        "- 训练日志：`logs/train_hierarchical_*.log`",
        "- 评测日志：`logs/eval_*.log`",
        "- 媒体日志：`logs/media/`",
        "- 媒体汇总：`manifests/media_manifest.json`",
        "",
        "## 5. 训练口径",
        "",
        "- 旧的 `flow_guided_gat` 全量结果仍是原方法结果，不与本研究线混合。",
        "- 当前学习分配器只学习高层任务选择，低层仍为固定 flow-guided 控制；因此不能把本轮结果称为端到端纯 MARL。",
        "- v1 和 v2 checkpoint、日志及配置均保留，负结果也保留用于后续消融。",
    ])
    output = ROOT / "reports" / "ITERATION_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
