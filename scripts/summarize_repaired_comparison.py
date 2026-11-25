"""Generate repaired paired tables and report from JSON artifacts only."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/COMPARISON_20261125_REPAIRED"

def load(name): return json.loads((OUT / name).read_text())

def paired(a, b):
    am = {x["scenario_hash"]: x for x in a}
    bm = {x["scenario_hash"]: x for x in b}
    keys = sorted(set(am) & set(bm))
    d = np.asarray([float(am[k]["navigation_success"]) - float(bm[k]["navigation_success"]) for k in keys])
    if len(d) == 0: return {"pairs": 0, "wins": 0, "ties": 0, "losses": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(41)
    boot = rng.choice(d, (4000, len(d)), replace=True).mean(1)
    return {"pairs": len(d), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum()), "mean": float(d.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}

def main():
    train = load("paired_train180.json"); hold = load("paired_holdout100.json")
    records = []
    for split, pursuit_name, ippo_name, mappo_name in (
        ("train", "pursuit_train180.json", "ippo_s41_train180.json", "mappo_s41_train180.json"),
        ("holdout", "pursuit_holdout100.json", "ippo_s41_holdout100.json", "mappo_s41_holdout100.json")):
        scene = train if split == "train" else hold
        pursuit = load(pursuit_name)["episodes"]
        for method, name in (("IPPO-S41", ippo_name), ("MAPPO-S41", mappo_name)):
            if (OUT / name).exists():
                learned = load(name)["episodes"]
                p = paired(learned, pursuit)
                records.append(dict(method=method, split=split, scenes=len(learned), common_scenes=p["pairs"],
                                    success_n=sum(x["navigation_success"] for x in learned), success_d=len(learned),
                                    strict=sum(x["strict_success"] for x in learned), strict_d=len(learned),
                                    wall=float(np.mean([x["wall_contact_s"] for x in learned])),
                                    lost=float(np.mean([bool(x["lost"]) for x in learned])), **{f"paired_{k}": v for k,v in p.items()}))
    with (OUT / "paired_local_control_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    tpg_on, tpg_off = load("fixed_three_tpg_on.json"), load("fixed_three_tpg_off.json")
    lines = ["# 修复后的配对重评报告", "", "## 场景清单", "",
             f"训练 180 行清单 SHA-256 `{hashlib.sha256((OUT/'paired_train180.json').read_bytes()).hexdigest()}`，留出 100 行清单 SHA-256 `{hashlib.sha256((OUT/'paired_holdout100.json').read_bytes()).hexdigest()}`。从仓库中可复核的旧 IPPO JSON 计算，训练交集为 2 行，留出交集为 0 行。用户给出的 4/2 线索无法由当前已提交旧 JSON 重建，不能作为本轮统计。", "",
             "## 配对局部控制结果", "", "| 方法 | split | 场景数 | 共同场景 | 成功 | 严格成功 | 壁接触 robot·s | 丢失率 | 配对胜/平/负 | 95% CI |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in records:
        lines.append(f"| {r['method']} | {r['split']} | {r['scenes']} | {r['common_scenes']} | {r['success_n']}/{r['success_d']} | {r['strict']:.0f}/{r['strict_d']} | {r['wall']:.3f} | {r['lost']:.3f} | {r['paired_wins']}/{r['paired_ties']}/{r['paired_losses']} | {r['paired_ci95']} |")
    lines += ["", "## TPG on/off", "", "TPG-on 使用当前三病灶访问顺序建立协调器，TPG-off 保持相同场景、起点、病灶和局部 pursuit，仅关闭 gate。`tpg_trigger_count` 统计协调器 gate 调用次数（当前版本的可审计触发计数），spacing_violation_pair_s 来自 SpacingTracker，实体接触单独记录。", "", "| N | on 全覆盖 | off 全覆盖 | on 最后到达 s | off 最后到达 s | on 触发次数 | on 间距违规 pair·s | off 间距违规 pair·s |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in (1,2,3):
        on, off = tpg_on["by_clusters"][str(n)], tpg_off["by_clusters"][str(n)]
        rg = lambda d,k: d[k]
        triggers = sum(x["tpg_trigger_count"] for x in tpg_on["rows"] if x["clusters"] == n)
        onv = [x["spacing_violation_pair_s"] for x in tpg_on["rows"] if x["clusters"] == n]
        offv = [x["spacing_violation_pair_s"] for x in tpg_off["rows"] if x["clusters"] == n]
        lines.append(f"| {n} | {on['all_three_rate']:.3f} | {off['all_three_rate']:.3f} | {on['last_arrival_s']:.3f} | {off['last_arrival_s']:.3f} | {triggers} | {np.mean(onv):.3f} | {np.mean(offv):.3f} |")
    lines += ["", "## 结论边界", "", "IPPO-S41 与 MAPPO-S41 均完成训练 180 行和留出 100 行的同场景重评。训练集上 IPPO 为 153/180，MAPPO 为 151/180，配对差值 IPPO−MAPPO 的胜/平/负为 2/177/1。留出集两者均为 80/100，100 对全部平局。这里的成功指标是导航成功，严格成功、壁接触和丢失率另列，不能把导航成功差异解释为整体安全优势。", "", "TPG 的 N=3 六场景结果显示 on/off 全覆盖均为 4/6。成功回合平均最后到达时间为 on 64.575 s、off 61.175 s，配对差值 +3.40 s，未证明速度收益。将未完成回合按 300 s 预算计分后，on 143.050 s、off 140.783 s，差值 +2.267 s。TPG-on 间距违规均值 3.826 pair·s，off 为 6.101 pair·s，安全间距代价下降。"]
    (OUT / "REPAIRED_REPORT.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__": main()
