"""Merge the HCC paired-seed shards and report paired statistics.

R3 reported point estimates over 3 paired seeds per difficulty, where one
episode moves the rate by 33 pp.  This script merges the 12-repeat expansion and
reports, for every baseline, the paired difference against ours_hcc with a
bootstrap interval over seeds plus an exact McNemar p-value on the binary
success pairs.  Completion time is paired only over episodes both methods
solved, because an unsolved episode has no time (right-censored).
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

OURS = "ours_hcc"
B = 10000


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b wins for ours, c against)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    boot = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    a = ap.parse_args()

    rows, meta, errors = [], {}, []
    for s in a.shards:
        d = json.loads(s.read_text())
        meta = {k: v for k, v in d.items() if k != "rows"}
        for r in d["rows"]:
            (errors if "error" in r else rows).append(r)

    by = defaultdict(dict)   # method -> (repeat, difficulty) -> row
    for r in rows:
        by[r["method"]][(r["repeat"], r["difficulty"])] = r
    methods = sorted(by)
    assert OURS in by, "ours_hcc missing from the merged shards"
    keys = sorted(by[OURS])
    rng = np.random.default_rng(20261026)

    def agg(m, dfilter=None):
        v = [r for k, r in by[m].items() if dfilter is None or k[1] == dfilter]
        if not v:
            return None
        solved = [x["navigation_time_s"] for x in v if x.get("navigation_time_s") is not None]
        return dict(episodes=len(v),
                    success=float(np.mean([x["navigation_success"] for x in v])),
                    strict=float(np.mean([x["strict_navigation_success"] for x in v])),
                    wall=float(np.mean([x.get("wall_contact_s", 0.) for x in v])),
                    lost=float(np.mean([x.get("lost", 0) > 0 for x in v])),
                    collision=float(np.mean([x.get("particle_collision_events", 0) > 0 for x in v])),
                    time=float(np.mean(solved)) if solved else None)

    overall = {m: agg(m) for m in methods}
    per_diff = {m: {d: agg(m, d) for d in range(4)} for m in methods}

    comparisons = {}
    for m in methods:
        if m == OURS:
            continue
        shared = [k for k in keys if k in by[m]]
        o = np.array([by[OURS][k]["navigation_success"] for k in shared], float)
        x = np.array([by[m][k]["navigation_success"] for k in shared], float)
        ow = np.array([by[OURS][k].get("wall_contact_s", 0.) for k in shared], float)
        xw = np.array([by[m][k].get("wall_contact_s", 0.) for k in shared], float)
        both = [k for k in shared
                if by[OURS][k].get("navigation_time_s") is not None
                and by[m][k].get("navigation_time_s") is not None]
        ot = np.array([by[OURS][k]["navigation_time_s"] for k in both], float)
        xt = np.array([by[m][k]["navigation_time_s"] for k in both], float)
        sm, slo, shi = paired_ci((o - x) * 100, rng)
        wm, wlo, whi = paired_ci(xw - ow, rng)          # positive = ours contacts less
        tm, tlo, thi = paired_ci(xt - ot, rng)          # positive = ours faster
        comparisons[m] = dict(
            n_paired=len(shared), n_both_solved=len(both),
            success_pp=sm, success_lo=slo, success_hi=shi,
            mcnemar_p=mcnemar_exact(int(((o == 1) & (x == 0)).sum()), int(((o == 0) & (x == 1)).sum())),
            ours_wins=int(((o == 1) & (x == 0)).sum()), ours_losses=int(((o == 0) & (x == 1)).sum()),
            wall_reduction=wm, wall_lo=wlo, wall_hi=whi,
            time_reduction=tm, time_lo=tlo, time_hi=thi)

    result = dict(meta=meta, n_rows=len(rows), n_errors=len(errors),
                  errors=[dict(method=e["method"], difficulty=e["difficulty"],
                               repeat=e.get("repeat"), error=e["error"]) for e in errors],
                  overall=overall, per_difficulty=per_diff, comparisons=comparisons,
                  bootstrap_resamples=B, paired_unit="(repeat, difficulty) scene seed")
    a.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n")

    L = ["# HCC 新构造导航:12 配对种子扩展评估", "",
         f"合并 {len(a.shards)} 个分片,{len(rows)} 个有效回合,{len(errors)} 个错误。",
         f"配对单位 = (repeat, difficulty) 场景种子,每难度 {len(keys)//4} 个配对种子。",
         f"ours_hcc 权重:`{meta.get('ours_checkpoint','?')}`",
         f"置信区间 = 按种子配对 bootstrap({B} 次重采样)的 95% 区间;p 值 = 精确 McNemar。", "",
         "## 总体(四难度合并)", "",
         "| 算法 | 回合 | 成功率 | Strict | 壁接触 robot·s | 丢失率 | 碰撞回合率 | 完成时间 s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in sorted(methods, key=lambda z: -overall[z]["success"]):
        s = overall[m]
        t = f"{s['time']:.1f}" if s["time"] is not None else "—"
        star = " **(本文)**" if m == OURS else ""
        L.append(f"| {m}{star} | {s['episodes']} | {s['success']*100:.1f}% | {s['strict']*100:.1f}% | "
                 f"{s['wall']:.2f} | {s['lost']*100:.1f}% | {s['collision']*100:.1f}% | {t} |")
    L += ["", "## 相对本文方法的配对差值(正值 = 本文更优)", "",
          "| 对比算法 | n | 成功率差 pp | 95% CI | 胜/负回合 | McNemar p | 壁接触减少 | 95% CI | 完成时间减少 s | n(双方成功) |",
          "|---|---:|---:|---|---:|---:|---:|---|---:|---:|"]
    for m in sorted(comparisons, key=lambda z: -comparisons[z]["success_pp"]):
        c = comparisons[m]
        tr = f"{c['time_reduction']:+.1f}" if not math.isnan(c["time_reduction"]) else "—"
        L.append(f"| {m} | {c['n_paired']} | {c['success_pp']:+.1f} | "
                 f"[{c['success_lo']:+.1f}, {c['success_hi']:+.1f}] | {c['ours_wins']}/{c['ours_losses']} | "
                 f"{c['mcnemar_p']:.3f} | {c['wall_reduction']:+.2f} | "
                 f"[{c['wall_lo']:+.2f}, {c['wall_hi']:+.2f}] | {tr} | {c['n_both_solved']} |")
    L += ["", "## 分难度成功率", "",
          "| 算法 | 难度0 (4障碍/.025) | 难度1 (8/.05) | 难度2 (16/.10) | 难度3 (32/.15) |", "|---|---:|---:|---:|---:|"]
    for m in methods:
        cells = " | ".join(f"{per_diff[m][d]['success']*100:.1f}%" if per_diff[m][d] else "—" for d in range(4))
        L.append(f"| {m} | {cells} |")
    L += ["", "## 边界", "",
          "- 这是 procedural nominal HCC 构造的导航仿真,不是患者特异性临床验证。",
          "- ours_hcc 为单个训练种子(S41)的 HCC 专用策略,确定性推理;IPPO 同为单种子。",
          "  跨训练种子的方差未在此量化。",
          "- 完成时间只在双方都成功的回合上配对,未解出的回合为右删失,不计入。",
          "- 此评估为 HCC 开发集迁移复测,不是密封测试;注册门禁状态未因此改变。"]
    a.out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
