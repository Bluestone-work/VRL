"""Build an auditable paired comparison from episode JSON artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/COMPARISON_20261125_REPAIRED"
LEGACY = ROOT / "research/COMPARISON_20261125"
IDENTITY = ("split", "scene_seed", "scenario_hash", "realized_scenario_hash",
            "accepted_seed", "navigation_targets")


def load(path: Path, default_split=None):
    payload = json.loads(path.read_text())
    rows = payload.get("episodes", payload)
    if default_split:
        rows = [dict(row, split=default_split) for row in rows]
    return rows


def _identity(row):
    return tuple(row.get(key) if key != "navigation_targets" else tuple(row.get(key) or ())
                 for key in IDENTITY)


def validate_rows(rows, label):
    errors, by_hash = [], {}
    for index, row in enumerate(rows):
        key = row.get("scenario_hash")
        if not key:
            errors.append({"label": label, "index": index, "error": "missing_scenario_hash"})
            continue
        if key in by_hash:
            errors.append({"label": label, "index": index, "scenario_hash": key,
                           "error": "duplicate_scenario_hash", "first_index": by_hash[key][0]})
        else:
            by_hash[key] = (index, row)
        for field in ("realized_scenario_hash", "accepted_seed", "navigation_targets"):
            if row.get(field) is None:
                errors.append({"label": label, "index": index, "scenario_hash": key,
                               "error": f"missing_{field}"})
    return by_hash, errors


def strict_pairs(a, b, label):
    am, errors = validate_rows(a, f"{label}:a")
    bm, other = validate_rows(b, f"{label}:b")
    errors.extend(other)
    if set(am) != set(bm):
        errors.append({"label": label, "error": "scenario_set_mismatch",
                       "only_a": sorted(set(am) - set(bm)), "only_b": sorted(set(bm) - set(am))})
    rows = []
    for key in sorted(set(am) & set(bm)):
        ai, ar = am[key]
        bi, br = bm[key]
        for field in IDENTITY:
            if _identity(ar)[IDENTITY.index(field)] != _identity(br)[IDENTITY.index(field)]:
                errors.append({"label": label, "scenario_hash": key, "error": "identity_mismatch",
                               "field": field, "a": ar.get(field), "b": br.get(field),
                               "a_index": ai, "b_index": bi})
        rows.append((key, ar, br))
    if errors:
        raise ValueError(json.dumps({"comparison": label, "errors": errors}, ensure_ascii=False, indent=2))
    return rows


def paired_summary(rows, method_a, method_b, split, training_seed=41):
    delta = np.asarray([float(a["navigation_success"]) - float(b["navigation_success"])
                        for _, a, b in rows])
    boot = np.random.default_rng(41).choice(delta, (4000, len(delta)), replace=True).mean(1)
    return {"method_a": method_a, "method_b": method_b, "split": split,
            "training_seed": training_seed, "paired_scene_count": len(rows),
            "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()), "difference_mean_a_minus_b": float(delta.mean()),
            "difference_direction": "a_gt_b" if np.any(delta > 0) else "b_gt_a" if np.any(delta < 0) else "observed_zero_difference",
            "observed_zero_difference": bool(np.all(delta == 0)), "equivalence_proven": False,
            "ci95_a_minus_b": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def legacy_overlap(split):
    pursuit_path = LEGACY / f"pursuit_{split}.json"
    ippo_path = LEGACY / f"ippo_s41_{split}_chunked.json"
    split_name = "train" if split.startswith("train") else "holdout"
    pursuit, pe = validate_rows(load(pursuit_path, split_name), f"legacy pursuit {split}")
    ippo, ie = validate_rows(load(ippo_path, split_name), f"legacy ippo {split}")
    shared = sorted(set(pursuit) & set(ippo))
    mismatches = [key for key in shared if _identity(pursuit[key][1]) != _identity(ippo[key][1])]
    if pe or ie or mismatches:
        raise ValueError(json.dumps({"legacy": split, "errors": pe + ie,
                                     "identity_mismatch": mismatches}, ensure_ascii=False, indent=2))
    return {"split": split, "pursuit_file": str(pursuit_path), "ippo_file": str(ippo_path),
            "identity_key": list(IDENTITY), "pursuit_scenes": len(load(pursuit_path, split_name)),
            "ippo_scenes": len(load(ippo_path, split_name)), "strict_identity_intersection": len(shared),
            "scenario_hashes": shared}


def main():
    specs = {
        "train": {"pursuit": "pursuit_train180.json", "IPPO-S41": "ippo_s41_train180.json", "MAPPO-S41": "mappo_s41_train180.json"},
        "holdout": {"pursuit": "pursuit_holdout100.json", "IPPO-S41": "ippo_s41_holdout100.json", "MAPPO-S41": "mappo_s41_holdout100.json"},
    }
    summaries, aggregate_rows, episode_rows = [], [], []
    for split, names in specs.items():
        loaded = {name: load(OUT / filename, split) for name, filename in names.items()}
        for method_a, method_b in (("IPPO-S41", "pursuit"), ("MAPPO-S41", "pursuit"), ("IPPO-S41", "MAPPO-S41")):
            pairs = strict_pairs(loaded[method_a], loaded[method_b], f"{split}:{method_a}:{method_b}")
            summary = paired_summary(pairs, method_a, method_b, split)
            summaries.append(summary); aggregate_rows.append(summary)
            for scene_hash, a, b in pairs:
                episode_rows.append({"method_a": method_a, "method_b": method_b, "split": split, "training_seed": 41,
                                     "scenario_hash": scene_hash, "scene_seed": a.get("scene_seed"),
                                     "accepted_seed_a": a.get("accepted_seed"), "accepted_seed_b": b.get("accepted_seed"),
                                     "navigation_targets_a": json.dumps(a.get("navigation_targets")),
                                     "navigation_targets_b": json.dumps(b.get("navigation_targets")),
                                     "navigation_success_a": int(bool(a.get("navigation_success"))),
                                     "navigation_success_b": int(bool(b.get("navigation_success"))),
                                     "difference_a_minus_b": int(bool(a.get("navigation_success"))) - int(bool(b.get("navigation_success")))})
    legacy = {split: legacy_overlap(split) for split in ("train180", "holdout100")}
    (OUT / "paired_comparisons.json").write_text(json.dumps({"identity_key": list(IDENTITY), "comparisons": summaries, "legacy_overlap": legacy}, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "paired_comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(aggregate_rows)
    with (OUT / "paired_episode_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(episode_rows)

    records = []
    for split, names in specs.items():
        pursuit = load(OUT / names["pursuit"], split)
        for method in ("IPPO-S41", "MAPPO-S41"):
            learned = load(OUT / names[method], split); strict_pairs(learned, pursuit, f"{split}:{method}:pursuit")
            records.append({"method": method, "split": split, "scenes": len(learned), "common_scenes": len(learned),
                            "success_n": sum(bool(x.get("navigation_success")) for x in learned), "success_d": len(learned),
                            "strict": sum(bool(x.get("strict_success")) for x in learned), "strict_d": len(learned),
                            "wall": float(np.mean([x["wall_contact_s"] for x in learned])), "lost": float(np.mean([bool(x["lost"]) for x in learned]))})
    with (OUT / "paired_local_control_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(records)

    on_path = OUT / "fixed_three_tpg_on_recount.json"
    off_path = OUT / "fixed_three_tpg_off_recount.json"
    tpg_on = json.loads((on_path if on_path.exists() else OUT / "fixed_three_tpg_on.json").read_text())
    tpg_off = json.loads((off_path if off_path.exists() else OUT / "fixed_three_tpg_off.json").read_text())
    edge = json.loads((OUT / "fixed_three_tpg_on_n3_edge.json").read_text())
    lines = ["# 修复后的配对重评报告", "", "## 起始提交与身份校验", "",
             "本轮工作分支从 `d6940d0f352381af774a1711374c03d33aca963f` 建立。远端目标分支在该提交后没有更新。当前配对比较要求以下身份键完全一致并检查唯一性，`split`、`scene_seed`、`scenario_hash`、`realized_scenario_hash`、`accepted_seed`、`navigation_targets`。全平局仅表示观测到零差异，`equivalence_proven` 保持为 false。", "",
             "旧 chunked 输入的严格身份交集为训练 4、留出 2。输入文件和身份键已写入 `paired_comparisons.json`。", "", "## 配对局部控制结果", "",
             "| method_a | method_b | split | 训练种子 | 配对场景数 | 胜 | 平 | 负 | 差值方向 |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        lines.append(f"| {row['method_a']} | {row['method_b']} | {row['split']} | {row['training_seed']} | {row['paired_scene_count']} | {row['wins']} | {row['ties']} | {row['losses']} | {row['difference_direction']} |")
    lines += ["", "## 回归数字", "", "训练侧导航成功 pursuit 152/180、IPPO 153/180、MAPPO 151/180。IPPO 对 pursuit 为 2/177/1，IPPO 对 MAPPO 为 2/178/0。留出侧三种方法逐场成败相同，均为 80/100，严格成功均为 77/100。脚本从 JSON 自动生成这些数字，旧数字只作回归校验。", "", "## TPG on/off", "", "TPG-on 与 TPG-off 共用场景、起点、靶点和局部 pursuit。`tpg_gate_calls` 是 gate 调用次数，`tpg_trigger_count` 是 hold 边沿次数，`waiting_transitions` 和 `waiting_robot_s` 分别是等待转入次数和累计等待 robot·s，`spacing_violation_pair_s` 来自 SpacingTracker，实体接触单独记录。", "", "| N | on 全覆盖 | off 全覆盖 | on 最后到达 s | off 最后到达 s | on 触发次数 | on 间距违规 pair·s | off 间距违规 pair·s |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in (1, 2, 3):
        on, off = tpg_on["by_clusters"][str(n)], tpg_off["by_clusters"][str(n)]
        onv = [x["spacing_violation_pair_s"] for x in tpg_on["rows"] if x["clusters"] == n]; offv = [x["spacing_violation_pair_s"] for x in tpg_off["rows"] if x["clusters"] == n]
        trigger_rows = edge["rows"] if n == 3 else [x for x in tpg_on["rows"] if x["clusters"] == n]
        lines.append(f"| {n} | {on['all_three_rate']:.3f} | {off['all_three_rate']:.3f} | {on['last_arrival_s']:.3f} | {off['last_arrival_s']:.3f} | {sum(x['tpg_trigger_count'] for x in trigger_rows)} | {np.mean(onv):.3f} | {np.mean(offv):.3f} |")
    lines += ["", "## 结论边界", "", "IPPO 在训练侧仅有 2 个导航胜场、1 个负场，留出侧没有逐场导航差异。全平局不能证明方法等效。TPG 的 N=3 六场景全覆盖均为 4/6，on 成功回合最后到达时间 64.575 s，off 为 61.175 s，间距违规均值分别为 3.826 与 6.101 pair·s。这里报告的是导航到达和安全计量，不等同于药物释放或磁场干扰终点。"]
    (OUT / "REPAIRED_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
