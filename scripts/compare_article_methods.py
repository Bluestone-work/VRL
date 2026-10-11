"""Build a reproducible comparison of paper methods and all registered baselines.

The repository contains two deliberately different protocols.  The manuscript
benchmark measures the deployable tracking+TPG method over 14 anatomies and
N=1/2/3.  EXP0073 is a hard dynamics screening matrix for delay/flow stress.
This report keeps them separate and only compares methods within each protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MAIN = [
    # N, method, information, safe, raw, removal, t100, auc, wall
    (1, "本文：追踪", "可部署", 81.0, 90.2, 96.7, 173, 67.6, 8.2),
    (1, "传统规划", "特权真值", 56.9, 58.6, 77.2, 201, 53.4, .1),
    (1, "规则路线跟随", "特权真值", 41.7, 42.6, 64.3, 300, 47.5, .0),
    (2, "本文：追踪+TPG", "可部署", 81.2, 95.5, 98.3, 101, 81.0, 8.9),
    (2, "本文：仅追踪", "可部署", 77.1, 97.6, 99.2, 96, 81.8, 5.4),
    (2, "传统规划", "特权真值", 66.9, 74.5, 90.4, 132, 73.0, 3.9),
    (2, "规则路线跟随", "特权真值", 49.3, 54.0, 82.5, 247, 68.6, 1.5),
    (2, "Codex 分层RL/规则TPG", "实测", 4.0, 41.2, 69.3, 300, 56.3, 23.0),
    (3, "本文：追踪+TPG", "可部署", 80.2, 96.7, 98.9, 73, 86.2, 6.0),
    (3, "本文：仅追踪", "可部署", 61.7, 93.1, 98.9, 75, 86.1, 9.8),
    (3, "传统规划", "特权真值", 63.6, 83.8, 95.2, 95, 81.1, 8.5),
    (3, "规则路线跟随", "特权真值", 58.8, 70.0, 90.7, 110, 78.2, 14.2),
    (3, "Codex 分层RL", "实测", 4.4, 59.0, 83.0, 200, 70.0, 28.0),
]


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard", type=Path,
                    default=Path("research/validation/EXP0073_HARD_BASELINES_20261009/aggregate.json"))
    ap.add_argument("--sched", type=Path,
                    default=Path("research/figures/SCHED_AUTO_20261010/flowres_s5/aggregate.json"))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    hard = load_json(a.hard); sched = load_json(a.sched)
    lines = [
        "# 本文方法与其他论文/传统对比方法的统一查阅报告", "",
        "本报告保留两个评估协议，不把不同任务、观测权限或指标混成一个排名。",
        "主基准来源：`research/MANUSCRIPT_DRAFT_20261006.md`；困难动力学来源：`EXP0073_HARD_BASELINES_20261009`；",
        "最新调度残差来源：`SCHED_AUTO_20261010/flowres_s5`。", "",
        "## 1. 论文主基准：14 解剖 × 30 场景 × N=1/2/3",
        "",
        "主指标是 Safe Success；本文方法使用可部署观测，传统规划/路线跟随使用仿真真值，因此传统方法拥有信息特权。", "",
        "| N | 方法 | 信息 | Safe % | Raw % | 清除 % | T100 s | AUC % | 壁接触 s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, m, info, safe, raw, rem, t100, auc, wall in MAIN:
        lines.append(f"| {n} | {m} | {info} | {safe:.1f} | {raw:.1f} | {rem:.1f} | {t100:.0f} | {auc:.1f} | {wall:.1f} |")
    lines += [
        "", "结论：论文主方法 `追踪+TPG` 在 N=1/2/3 为 **81.0/81.2/80.2%**，",
        "高于特权传统规划的 **56.9/66.9/63.6%**。这个 80% 结论对应的是这套主基准。", "",
        "## 2. 困难动力学筛查：同场景、四个 stress panel",
        "",
        "这里比较当前路线跟随/调度实现、PAC-NMPC、STPG、固定/自适应规则和 Transformer；所有方法 42 个配对场景。", "",
        "| 方法 | 平均 Strict % | 平均 Raw % | 平均清除 % | 平均 AUC | 平均 T90_300 s | 平均壁接触 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    hard_rows = hard["tables"]
    methods = sorted({r["method"] for r in hard_rows})
    for method in methods:
        rr = [r["metrics"] for r in hard_rows if r["method"] == method]
        mean = lambda k: sum(x[k] for x in rr) / len(rr)
        label = {"flow_aux": "本文：flow auxiliary Transformer", "nav_tf_v3": "本文原始 Transformer",
                 "adaptive_settle": "Adaptive Settle", "no_settle": "No Settle",
                 "pac_nmpc": "PAC-NMPC", "settle": "Fixed Settle", "stpg": "STPG"}.get(method, method)
        lines.append(f"| {label} | {100*mean('strict_success'):.1f} | {100*mean('task_success'):.1f} | {100*mean('removal'):.1f} | {mean('removal_auc'):.3f} | {mean('T90_300'):.1f} | {mean('wall_contact_s'):.2f} |")
    lines += ["", "困难筛查的逐面板 Strict：", "", "| 面板 | 本文 flow_aux | 最强其他方法 | 差距 |", "|---|---:|---|---:|"]
    for panel in sorted({r["panel"] for r in hard_rows}):
        rr = [r for r in hard_rows if r["panel"] == panel]
        ours = next(r for r in rr if r["method"] == "flow_aux")["metrics"]["strict_success"]
        best = max((r for r in rr if r["method"] != "flow_aux"), key=lambda r: r["metrics"]["strict_success"])
        lines.append(f"| {panel} | {100*ours:.1f} | {best['method']} ({100*best['metrics']['strict_success']:.1f}) | {100*(ours-best['metrics']['strict_success']):+.1f} pp |")
    lines += [
        "", "## 3. 当前 FlowResidual 与 SwitchSettle",
        "", "| 面板 | SwitchSettle Strict | FlowResidual Strict | 差距 |",
        "|---|---:|---:|---:|",
    ]
    sr = sched["tables"]
    panels = sorted({r["panel"] for r in sr})
    for panel in panels:
        r = {x["method"]: x for x in sr if x["panel"] == panel}
        if "SwitchSettle" in r and "FlowResidual" in r:
            s = r["SwitchSettle"]["metrics"]["strict_success"]; f = r["FlowResidual"]["metrics"]["strict_success"]
            lines.append(f"| {panel} | {100*s:.1f} | {100*f:.1f} | {100*(f-s):+.1f} pp |")
    so = sched["overall"]
    lines += ["", f"整体：SwitchSettle **{100*so['rule_strict']:.1f}%**，FlowResidual **{100*so['learner_strict']:.1f}%**，差值 **{so['delta_strict_pp']:+.1f} pp**。", "",
              "## 4. 解释", "",
              "1. 论文主方法的 80% 结果是真实存在的，但它属于主基准的 Safe Success，不能直接套到困难流场/延迟矩阵。",
              "2. 在困难矩阵中，当前学习方法主要改善高延迟和变量响应，低延迟仍落后 Adaptive Settle，强流仍落后原始 Transformer 或规则方法。",
              "3. 当前 FlowResidual 相对 SwitchSettle 只获得边际提升，说明它还没有恢复论文主方法的 80% 水平。",
              "4. PAC-NMPC、STPG 等结果是仓库中的适配实现，不应写成原论文作者代码的精确复现；主基准中的传统规划也使用了特权真值。",
              "",
              "## 数据来源", "",
              "- 主基准与论文表：`research/MANUSCRIPT_DRAFT_20261006.md` §7.1。",
              "- 同场景困难筛查：`research/validation/EXP0073_HARD_BASELINES_20261009/REPORT.md`。",
              "- FlowResidual 对照：`research/figures/SCHED_AUTO_20261010/flowres_s5/REPORT.md`。",
    ]
    (a.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (a.out / "sources.json").write_text(json.dumps({"main": "research/MANUSCRIPT_DRAFT_20261006.md", "hard": str(a.hard), "sched": str(a.sched)}, indent=2))


if __name__ == "__main__":
    main()
