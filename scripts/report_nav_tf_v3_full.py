"""Aggregate every V5 physiology result for nav_tf_v3, baselines and ablations.

The raw files contain 14 anatomies x 10 scenes for each cluster count.  This
report averages only within a registered (variation strength, N, method) cell,
keeps the completed-episode count, and writes both readable Markdown tables and
machine-readable CSV/JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LABEL = {
    "nav_tf_v3": "本文 nav_tf_v3",
    "nav_tf_v2": "本文 v2 消融",
    "nav_tf_v4": "本文 v4 消融",
    "nav_tf_ecg": "本文 ECG 消融",
    "classical_settle": "经典 Fixed/Settle",
    "classical_adaptive": "Adaptive Settle",
    "stpg": "STPG (AAAI 2025)",
    "pac_nmpc": "PAC-NMPC (CASE 2025)",
    "turbo_v5": "Turbo (NMI 2026)",
    "oracle_flow": "FlowOracle 上界",
    "oracle_noflow": "Oracle 无流估计",
    "oracle_noresp": "Oracle 无响应估计",
    "oracle_sensing_only": "Oracle 仅精确感知",
}

STRENGTHS = ("s0", "s1", "s15")
METRICS = [
    ("relaxed_success", "relaxed %", 100.),
    ("task_success", "raw/full %", 100.),
    ("strict_success", "strict %", 100.),
    ("removal", "removal %", 100.),
    ("removal_auc", "AUC %", 100.),
    ("t50_300", "T50_300 s", 1.),
    ("t90_300", "T90_300 s", 1.),
    ("t100_300", "T100_300 s", 1.),
    ("wall_contact_s", "wall s", 1.),
    ("wall_ge_1s", "wall >=1s %", 100.),
    ("wall_ge_5s", "wall >=5s %", 100.),
    ("max_continuous_wall_contact_s", "max wall s", 1.),
    ("lost", "lost / ep", 1.),
    ("spacing_violation_pair_s", "spacing pair-s", 1.),
    ("robot_pair_contact_s", "robot pair s", 1.),
    ("coupling_exposure", "coupling s", 1.),
    ("tpg_hold_s", "TPG hold s", 1.),
    ("path_mm", "path mm", 1.),
    ("elapsed_s", "elapsed s", 1.),
    ("walltime_s", "walltime s", 1.),
    ("clots_cleared", "clots cleared", 1.),
    ("clots", "clots", 1.),
    ("min_pair_distance_mm", "min pair mm", 1.),
]


def finite_mean(rows, key):
    vals = []
    for r in rows:
        x = r.get(key)
        if x is None:
            continue
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else None


def horizon_value(r, key):
    x = r.get(key)
    if x is None:
        return float(r.get("horizon_s", 300.))
    return float(x)


def load_cells(root):
    cells = defaultdict(list)
    for strength in STRENGTHS:
        for path in sorted(root.glob(f"{strength}_*.jsonl")):
            method = path.stem[len(strength) + 1:]
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error"):
                    continue
                cells[(strength, int(row["clusters"]), method)].append(row)
    return cells


def aggregate(cells):
    out = []
    for (strength, n, method), rows in sorted(cells.items()):
        d = {"strength": strength, "clusters": n, "method": method,
             "label": LABEL.get(method, method), "n": len(rows)}
        for key, _, _ in METRICS:
            if key == "t50_300":
                d[key] = float(np.mean([horizon_value(r, "t50_s") for r in rows]))
            elif key == "t90_300":
                d[key] = float(np.mean([horizon_value(r, "t90_s") for r in rows]))
            elif key == "t100_300":
                d[key] = float(np.mean([horizon_value(r, "t100_s") for r in rows]))
            elif key in ("wall_ge_1s", "wall_ge_5s"):
                d[key] = float(np.mean([bool(r.get(key, False)) for r in rows]))
            elif key == "min_pair_distance_mm" and n == 1:
                d[key] = None
            else:
                d[key] = finite_mean(rows, key)
        # Preserve method-specific diagnostics where present.
        for key in ("pac_bound_fail_rate", "stpg_infeasible", "stpg_solves", "stpg_switches"):
            value = finite_mean(rows, key)
            if value is not None:
                d[key] = value
        out.append(d)
    return out


def fmt(value, key):
    if value is None:
        return "—"
    scale = next(scale for name, _, scale in METRICS if name == key)
    if key in ("relaxed_success", "task_success", "strict_success", "removal", "removal_auc", "wall_ge_1s", "wall_ge_5s"):
        return f"{float(value)*scale:.1f}"
    if key in ("lost", "clots_cleared", "clots"):
        return f"{float(value):.2f}"
    return f"{float(value):.2f}"


def write_csv(rows, out):
    fields = ["strength", "clusters", "method", "label", "n"] + [x[0] for x in METRICS]
    with (out / "aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: r.get(k) for k in fields} for r in rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path("research/validation/V5_PHYSIO_20261008")); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(a.root); rows = aggregate(cells); write_csv(rows, a.out)
    (a.out / "aggregate.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    lines = [
        "# nav_tf_v3 全场景、全集群数、全指标对比", "",
        "数据：`V5_PHYSIO_20261008` 开发集；每个完整单元为 14 个解剖 × 10 个场景 = 140 回合。",
        "s=0 是名义动力学，s=1 是登记的生理扰动范围，s=1.5 是分布外扰动。",
        "百分比按回合平均；未达到 T50/T90/T100 的回合按 300 s 计入对应 `_300` 指标；n 显示实际完成回合。", "",
        "## 方法实现", "",
        "`nav_tf_v3` 是一个因果 Transformer PPO 导航器。它不直接读取仿真真值流场、真实响应参数或真实位置，"
        "只读取术前 CTA 路线和延迟、带噪的成像估计。每个机器人维护 32 步（3.2 s）的历史窗口。", "",
        "- 动作：`a0` 调整沿路线速度，`a1/a2` 在路线 Frenet 法向/副法向上调整瞄准点；动作先裁剪到 [-1,1]。",
        "- 动作映射：v3 实际使用 `speed=clip(s_prior+a0,0,1)`，其中 `s_prior` 是 Adaptive Settle 速度先验；无先验的旧接口才是 `clip(1+a0,0,1)`。瞄准点为 carrot 加上 `0.7*(管腔半径-机器人半径)*(a1*n+a2*b)`。",
        "- 观测：carrot 方向、当前和前方 1/2 mm 路线切向、地图半径/壁间隙、径向偏移、估计速度及路线分量、上一步动作、"
        "到目标距离、分叉标志、TPG hold、剩余时间、最近两个同伴；可选 ECG 只用于 ECG 消融。",
        "- 低层：所有方法共同使用 WallGuard 和间距 shield；TPG hold 是硬约束，策略不能越过。",
        "- 奖励：路线进度 + 血栓去除 − 壁接触 − 丢失 − 间距违规 − 动作变化 − 存活时间；γ=0.995，GAE λ=0.95。",
        "- 训练：PPO，3 层 Transformer，32 步历史，训练解剖随机采样，N∈{1,2,3}，s=0 或 U[0,1.25]，图像延迟 U{1,2}，"
        "流速取登记的 0.025/0.05/0.1 mm/s；策略均值受 Adaptive Settle 速度先验正则约束 β=1→0.05。",
        "- v1 从零直接学速度会塌缩为停止；v3 改成路线坐标和速度先验后才稳定。v4 加强困难课程后反而退化。", "",
        "## 方法身份和对比关系", "",
        "| 方法 | 类型 | 作用 | 信息权限 |",
        "|---|---|---|---|",
        "| nav_tf_v3 | 本文学习方法 | Transformer PPO 学速度修正和横向瞄准 | 可部署地图 + 延迟成像估计 |",
        "| nav_tf_v2 / nav_tf_v4 / nav_tf_ecg | 本文消融 | 分别检验旧动作映射、困难课程、ECG 相位 | 与 v3 相同，ECG 消融多 2 个相位量 |",
        "| classical_settle | 规则基线 | Fixed/Settle 追踪、壁护、TPG/停车 | 可部署估计，但动作由规则给出 |",
        "| classical_adaptive | 规则消融 | 追踪 + 自适应停滞解除 | 可部署估计 |",
        "| STPG | 外部论文风格基线 | 在线可切换 TPG 顺序协调 | 仓库适配实现 |",
        "| PAC-NMPC | 外部论文风格基线 | 分布式采样 NMPC 与安全约束 | 仓库适配实现 |",
        "| Turbo | 外部论文风格学习基线 | 直接 Transformer PPO 输出速度 | 仓库适配实现；同样受成像延迟影响 |",
        "| oracle_* | 特权上界 | 分解流场、响应和感知误差的可达上限 | 使用仿真真值，不是可部署方法 |",
        "",
        "旧稿主表中的“追踪+TPG”是规则管线（对应当前 `classical_settle` 的家族），不能把它的约 80% Safe Success 直接归因于学习器。"
        "当前论文若要把学习方法作为主方法，应以 `nav_tf_v3` 的 V5 结果和上述消融为主，并单独报告旧稿主基准。", "",
        "## 汇总表：成功、安全与清除", "",
        "以下表格覆盖所有已有方法、消融、oracle 上界、三种扰动强度和三个集群数。", "",
    ]
    success_cols = ["relaxed_success", "task_success", "strict_success", "removal", "removal_auc", "wall_contact_s", "wall_ge_1s", "wall_ge_5s", "lost", "spacing_violation_pair_s", "robot_pair_contact_s", "coupling_exposure"]
    eff_cols = ["t50_300", "t90_300", "t100_300", "max_continuous_wall_contact_s", "tpg_hold_s", "path_mm", "elapsed_s", "walltime_s", "clots_cleared", "clots", "min_pair_distance_mm"]
    for strength in STRENGTHS:
        lines += [f"### {strength}", ""]
        for n in (1, 2, 3):
            subset = [r for r in rows if r["strength"] == strength and r["clusters"] == n]
            lines += [f"#### {strength}, N={n}", ""]
            h = "| 方法 | n | " + " | ".join(next(label for name, label, _ in METRICS if name == k) for k in success_cols) + " |"
            sep = "|---|---:|" + "---:|" * len(success_cols) + "|"
            lines += [h, sep]
            for r in subset:
                lines.append("| " + r["label"] + " | " + str(r["n"]) + " | " + " | ".join(fmt(r.get(k), k) for k in success_cols) + " |")
            lines += ["", "| 方法 | n | " + " | ".join(next(label for name, label, _ in METRICS if name == k) for k in eff_cols) + " |",
                      "|---|---:|" + "---:|" * len(eff_cols) + "|"]
            for r in subset:
                lines.append("| " + r["label"] + " | " + str(r["n"]) + " | " + " | ".join(fmt(r.get(k), k) for k in eff_cols) + " |")
            lines.append("")
    lines += [
        "## 完整性和解释", "",
        "- STPG 的 N=1 没有运行，因为该方法是多集群时序协调基线；相应单元标为缺失而不是 0。",
        "- `nav_tf_v4` 在 s=1.5 的 N=2/N=3 评估被中止，n 会显示为不完整。",
        "- `oracle_*` 只用于分解“流场估计、响应估计、感知误差”的可达上限，不是可部署对比方法。",
        "- Turbo、STPG、PAC-NMPC 是仓库中的适配实现；不能表述为外部论文作者代码的精确复现。",
        "- 该报告是开发集，不是密封测试集；nav_tf_v3 为单训练种子，尚不能据此宣称稳定优于经典方法。", "",
        "## 原始来源", "",
        "- `scripts/train_lysis_nav.py`：训练和动作/观测定义。",
        "- `research/validation/V5_PHYSIO_20261008/REPORT.md`：实验协议和结论。",
        "- `research/validation/V5_PHYSIO_20261008/*.jsonl`：逐回合原始数据。",
    ]
    (a.out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
