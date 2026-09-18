"""Build a readable project overview from the recorded full-matrix artifacts."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("experiments/full_matrix_20260906")


def load_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rel(path: Path):
    return path.relative_to(ROOT).as_posix()


def main():
    ranking = load_csv(ROOT / "method_ranking.csv")
    scenarios = [line.strip() for line in (ROOT / "scenarios.txt").read_text().splitlines() if line.strip()]
    methods = [line.split("|", 1)[0] for line in (ROOT / "methods.tsv").read_text().splitlines() if line.strip()]
    slow_root = ROOT / "media_slow_v2" if (ROOT / "media_slow_v2").exists() else ROOT / "media_slow"
    slow_gifs = sorted(slow_root.glob("*/*/*.gif"))
    slow_mp4s = sorted(slow_root.glob("*/*/*.mp4"))
    original_gifs = sorted(ROOT.glob("media/*/*/*.gif"))
    original_mp4s = sorted(ROOT.glob("media/*/*/*.mp4"))

    lines = [
        "# 血管多智能体强化学习与世界模型项目总览",
        "",
        f"> 自动生成时间：{datetime.now().astimezone().isoformat()}",
        "> 这是项目说明与实验索引，不替代逐回合 JSONL 和原始训练日志。",
        "",
        "## 1. 一页摘要",
        "",
        "本项目研究 5 个微型机器人在三维血管网络中导航、抵抗血流、接触血栓并协同完成溶栓。环境使用参数化合成血管树和简化连续性/Poiseuille 风格流场；策略使用几何观测、多智能体 PPO 式更新、图注意力网络和流速感知残差控制。",
        "",
        "当前最强实验条目是 `flow_guided_gat`：流速感知基础控制器提供可行动作，GAT-MAPPO 只学习有界残差。全量矩阵中它在 20 个场景、400 个回合上成功率为 **83.75%**，平均质量清除率为 **96.08%**。",
        "",
        "> 重要口径：全量矩阵复用了已经训练好的 checkpoint 做统一重测；它不是 11 个条目全部重新训练。训练来源、checkpoint 和原始日志仍保留在 `training/` 及历史实验目录。",
        "",
        "## 2. 快速入口",
        "",
        f"- 全方法全场景报告：[`full_matrix_report.md`]({rel(ROOT / 'full_matrix_report.md')})",
        f"- 训练/评测/媒体审计：[`full_matrix_audit.json`]({rel(ROOT / 'full_matrix_audit.json')})",
        f"- 方法排名：[`method_ranking.csv`]({rel(ROOT / 'method_ranking.csv')})",
        f"- 场景成功率矩阵：[`scenario_success_matrix.csv`]({rel(ROOT / 'scenario_success_matrix.csv')})",
        f"- 成功率热力图：[`scenario_success_heatmap.png`]({rel(ROOT / 'scenario_success_heatmap.png')})",
        f"- 原始媒体索引：[`MEDIA_INDEX.md`]({rel(ROOT / 'MEDIA_INDEX.md')})",
        f"- 慢速媒体索引：[`MEDIA_SLOW_INDEX.md`]({rel(ROOT / 'MEDIA_SLOW_INDEX.md')})" if (ROOT / 'MEDIA_SLOW_INDEX.md').exists() else f"- 慢速媒体索引：正在生成 `{slow_root.name}/`，完成后运行媒体索引生成脚本。",
        "",
        "## 3. 研究任务",
        "",
        "每个回合包含：",
        "",
        "1. 在血管树中初始化机器人和 2–4 个随机质量血栓。",
        "2. 每个机器人输出一个三维连续速度控制动作。",
        "3. 动力学叠加主动控制、血流、随机扰动和机器人分离修正。",
        "4. 机器人接触血栓后按距离衰减模型清除血栓质量。",
        "5. 所有血栓质量清零则成功；否则最多运行 300 步。",
        "",
        "成功率和质量清除率不是同一个指标：清除 99% 但剩余质量未归零，仍可能被判为失败。",
        "",
        "## 4. 仿真环境",
        "",
        "### 4.1 场景组成",
        "",
        f"当前全量测试包含 **{len(scenarios)} 个场景**：4 个 legacy 几何、2 个 generated 几何和 14 个 anatomical 具名血管区域。",
        "",
        "- Legacy：`straight`、`bifurcation`、`anastomosis`、`stenotic`。",
        "- Generated：`multilevel`、`mca_stroke`。",
        "- Anatomical：肺动脉、冠脉、颈动脉、脑血管、髂静脉、腘静脉、肾动脉、肠系膜动脉和股腘动脉等参数化区域。",
        "",
        "解剖模板保存血管段直径、长度、分支角度、弯曲程度和血栓易发位置；它们是合成 morphometry，不等价于患者影像或 CFD 网格。",
        "",
        "### 4.2 物理与奖励",
        "",
        "- 最大主动速度：`0.018`（仿真归一化单位）。",
        "- 入口流速基准：`0.004`。",
        "- 血栓接触半径：`0.035`。",
        "- 贴壁润滑阻力会削弱主动推力。",
        "- 血栓清除具有饱和项，鼓励机器人分散处理多个血栓。",
        "- 奖励包含质量清除、接近目标、里程碑、完整清除奖励，以及撞壁和机器人碰撞惩罚。",
        "",
        "## 5. 观测与控制",
        "",
        "几何观测为每个机器人 36 维节点特征，包括位置、局部速度、沿目标路径的路由前瞻、管壁方向、管腔和阻塞信息、局部血流、贴壁润滑、目标血栓方向/距离/质量以及邻居信息。",
        "",
        "方向信息在血管 Frenet 局部坐标系中表达，再转换到世界坐标执行。这样同样的“沿切线前进”在不同血管位置具有一致含义。",
        "",
        "流速感知残差控制的核心形式为：",
        "",
        "```text",
        "executed_action = local_to_world(flow_guidance + 0.2 * learned_residual)",
        "```",
        "",
        "其中流速控制器负责基本可行导航，强化学习主要修正接触时机、分支选择和多机器人行为。",
        "",
        "## 6. 算法路线",
        "",
        "| 层次 | 实现 | 作用 |",
        "|---|---|---|",
        "| 更新算法 | MAPPO/PPO 式 clipped objective + GAE | 根据轨迹更新 actor/critic |",
        "| 节点编码 | GAT | 让机器人利用邻居和图邻接信息 |",
        "| 对照结构 | MLP、Edge-Bias GAT | 消融图交互和边信息 |",
        "| 控制增强 | guided、flow-guided、flow-spread | 注入几何/血流先验 |",
        "| 世界模型 | GAT dynamics ensemble | 预测观测、状态、奖励和终止概率 |",
        "| MVE | 3 步短程 imagination + uncertainty gate | 混合模型价值目标与真实目标 |",
        "",
        "当前矩阵的 11 个条目不是仓库全部注册架构；Transformer、稀疏 Transformer 和 hierarchical GNN 已注册但不在这张全量矩阵中。",
        "",
        "## 7. 全量测试结果",
        "",
        "测试协议：每个方法、每个场景 20 回合，共 400 回合；每回合独立进程执行，最终完成 4400 条记录。",
        "",
        "| 排名 | 方法 | 回合 | 成功率 | 质量清除率 | Wilson 95% CI |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(ranking, 1):
        lines.append(f"| {index} | `{row['method']}` | {row['episodes']} | {float(row['success_rate']):.2%} | {float(row['removal_rate']):.2%} | [{float(row['wilson95_low']):.2%}, {float(row['wilson95_high']):.2%}] |")
    lines.extend([
        "",
        "### 7.1 如何解读",
        "",
        "- `flow_guided_gat` 明显优于无学习残差的 `flow_controller`，说明学习残差在当前模拟器中提供了额外收益。",
        "- `flow_spread` 与 `flow_controller` 接近，说明显式分散启发式没有替代学习残差。",
        "- `mve_43` 没有超过 flow-guided 策略，不能宣称当前世界模型已经带来控制性能提升。",
        "- 低平均成功率的方法仍有可能在简单 legacy/generated 场景上成功，在 anatomical 场景上明显下降，因此必须查看热力图而非只看总平均。",
        "",
        "## 8. 世界模型",
        "",
        "世界模型不是视频生成器，而是结构化的一步/多步动力学模型：输入观测、动作、位置、速度、邻接、全局血栓状态、几何特征和场景 embedding，输出下一步观测变化、状态变化、位置、速度、奖励和终止概率。",
        "",
        "两套模型都通过当前 gate：五步预测相对 persistence 的误差不超过 0.8，奖励相对 zero baseline 的误差不超过 0.9。模型指标见 `experiments/overnight_research_20260906/world_models/`。",
        "",
        "但当前 MVE 结果只能作为探索性对照：模型集成不确定性是成员分歧，不是校准概率；并且后续增量 MVE 运行曾出现 native 崩溃，没有被算作成功改进。",
        "",
        "## 9. 媒体怎么读",
        "",
        "慢速媒体 HUD 使用以下字段：",
        "",
        "| HUD 字段 | 含义 |",
        "|---|---|",
        "| `reward` | 当前仿真步的总奖励 |",
        "| `return` | 从回合开始累计的奖励 |",
        "| `team` | 团队清除、里程碑、覆盖和成功奖励等团队项 |",
        "| `agent` | 当前步各机器人个体奖励均值 |",
        "| `lysis` | 当前步清除的血栓质量 |",
        "| `remaining` | 当前剩余血栓总质量 |",
        "| `wall hits` | 当前步撞壁数量 |",
        "| `peer` | 当前步机器人间碰撞数量 |",
        "| `speed` | 机器人实际位移速度相对最大主动速度的比例 |",
        "| `geo` | 机器人到当前分配血栓的血管测地距离 |",
        "",
        "慢速版本每个仿真步保留，固定镜头、5 FPS、最多 300 帧，约 60 秒；原来的高速媒体也保留。",
        "",
        "## 10. 文件结构",
        "",
        "```text",
        "full_matrix_20260906/",
        "├── methods/<method>/episodes/       逐回合 JSONL",
        "├── methods/<method>/logs/           每次评测/重试日志",
        "├── training/<method>/               checkpoint 与训练来源索引",
        "├── media/<method>/<scenario>/       原始速度 GIF/MP4",
        f"├── {slow_root.name}/<method>/<scenario>/  慢速、逐步 HUD GIF/MP4",
        "├── full_matrix_report.md             简版结果报告",
        "├── PROJECT_OVERVIEW.md               本项目总览",
        "├── full_matrix_audit.json            机器可读审计",
        "├── method_ranking.csv                方法排名",
        "└── scenario_success_heatmap.png     场景热力图",
        "```",
        "",
        "## 11. 复现入口",
        "",
        "```bash",
        "cd /home/wj/桌面/vascular_marl_local.tar./vascular_marl_local",
        "export PY=/home/wj/miniconda3/envs/v/bin/python",
        "env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\",
        "  $PY scripts/eval_single_episode.py --scenario mca_m1_lvo --seed 700000 \\",
        "  --policy experiments/success_study_20260905/selection/flow_guided_43.pt \\",
        "  --robots 5 --clots 3 --horizon 300 --robot-radius 0.0011 \\",
        "  --output /tmp/one_episode.jsonl",
        "```",
        "",
        "## 12. 局限与科研边界",
        "",
        "- 血管是参数化合成几何，不是患者级真实血管。",
        "- 流场是解析近似，不是离线 CFD 插值。",
        "- 每个机器人拥有独立三维控制，不等同于现实共享磁场控制。",
        "- `flow_spread` 使用环境内部血栓/路由信息，应视作 oracle-style 消融而不是严格局部观测策略。",
        "- 全量矩阵是统一重测，不等于每个方法都重新训练了 20 场景。",
        "- 当前全量矩阵实际每个场景的 seed 为 `700000–700019`；旧报告中声明的场景偏移 seed 公式与实际记录存在差异，不能忽略这一审计问题。",
        "",
    ])

    (ROOT / "PROJECT_OVERVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"path": str(ROOT / "PROJECT_OVERVIEW.md"),
                      "methods": len(methods), "scenarios": len(scenarios),
                      "slow_gif": len(slow_gifs), "slow_mp4": len(slow_mp4s),
                      "original_gif": len(original_gifs), "original_mp4": len(original_mp4s)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
