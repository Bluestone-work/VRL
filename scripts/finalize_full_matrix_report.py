"""Aggregate the full method x scenario matrix without hiding failures."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/full_matrix_20260906")
    args = parser.parse_args()
    root = Path(args.root)
    scenarios = [line.strip() for line in (root / "scenarios.txt").read_text().splitlines() if line.strip()]
    methods = []
    for line in (root / "methods.tsv").read_text().splitlines():
        if line.strip():
            label, checkpoint, kind, source, controller = line.split("|")
            methods.append({"label": label, "checkpoint": checkpoint or None, "kind": kind,
                            "source": source, "controller": controller})

    rows = []
    for method in methods:
        label = method["label"]
        summary = read_json(root / "methods" / label / "summary.json", {}) or {}
        episodes = sorted((root / "methods" / label / "episodes").glob("*.jsonl"))
        completed_scenarios = sorted({path.name.rsplit("_", 1)[0] for path in episodes})
        media_dir = root / "media" / label
        gifs = sorted(str(path.relative_to(root)) for path in media_dir.glob("*/*.gif"))
        mp4s = sorted(str(path.relative_to(root)) for path in media_dir.glob("*/*.mp4"))
        methods_record = {
            **method,
            "training": read_json(root / "training" / label / "record.json", {}),
            "summary": summary,
            "completed_episode_files": len(episodes),
            "completed_scenarios": len(completed_scenarios),
            "missing_scenarios": sorted(set(scenarios) - set((summary.get("per_scenario") or {}).keys())),
            "gif_count": len(gifs), "mp4_count": len(mp4s),
            "media_gif": gifs, "media_mp4": mp4s,
        }
        rows.append(methods_record)

    failures = []
    failure_path = root / "logs" / "failed_episodes.tsv"
    if failure_path.exists():
        for line in failure_path.read_text().splitlines():
            if line.strip():
                label, scenario, seed, status, attempts = line.split("\t")
                failures.append({"method": label, "scenario": scenario, "seed": int(seed),
                                 "exit_code": int(status), "attempts": int(attempts)})
    media_failures = []
    media_failure_path = root / "logs" / "media_failed.tsv"
    if media_failure_path.exists():
        for line in media_failure_path.read_text().splitlines():
            if line.strip():
                label, scenario, extension, status, attempts = line.split("\t")
                media_failures.append({"method": label, "scenario": scenario, "extension": extension,
                                       "exit_code": int(status), "attempts": int(attempts)})

    audit = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {"methods": len(methods), "scenarios": len(scenarios), "episodes_per_cell": 20,
                     "expected_episode_records": len(methods) * len(scenarios) * 20,
                     "expected_gif": len(methods) * len(scenarios),
                     "expected_mp4": len(methods) * len(scenarios)},
        "methods": rows, "evaluation_failures": failures, "media_failures": media_failures,
        "process_logs": sorted(str(path.relative_to(root)) for path in (root / "logs").glob("*.log")),
    }
    (root / "full_matrix_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    lines = ["# 全方法 × 全场景训练测试矩阵", "", f"生成时间：{audit['generated_at']}", "",
             "## 协议", "", "- 20 个场景：legacy 4、generated 2、anatomical 14。",
             "- 11 个方法；每个方法每个场景 20 回合，固定 seed 规则为 `700000 + 场景序号×10000 + 回合序号`。",
             "- 每回合隔离进程执行；native 崩溃单独记录，不从成功率中删除或补写。", "",
             "## 完成状态", "", "| 方法 | 训练来源 | 完成回合 | 场景数 | 成功率 | 清除率 | GIF | MP4 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        summary = row["summary"] or {}
        macro = summary.get("macro_completed_episodes") or {}
        training = row.get("training") or {}
        lines.append(f"| {row['label']} | {training.get('training_status', '-')} | {row['completed_episode_files']} | "
                     f"{row['completed_scenarios']} | {macro.get('success', 0):.2%} | {macro.get('removal_rate', 0):.2%} | "
                     f"{row['gif_count']} | {row['mp4_count']} |")
    lines.extend(["", "## 失败与审计", "", f"- 评测失败记录：{len(failures)} 条。",
                  f"- 媒体失败记录：{len(media_failures)} 条。",
                  f"- 逐方法 JSON 汇总：`{root / 'methods'}`。",
                  f"- 训练来源清单：`{root / 'training_manifest.json'}`。",
                  f"- 完整审计 JSON：`{root / 'full_matrix_audit.json'}`。",
                  f"- 方法排名 CSV：`{root / 'method_ranking.csv'}`；场景成功率矩阵：`{root / 'scenario_success_matrix.csv'}`。",
                  f"- 场景成功率热力图：`{root / 'scenario_success_heatmap.png'}`。",
                  f"- GIF/MP4 媒体索引：`{root / 'MEDIA_INDEX.md'}`。",
                  f"- 全部日志：`{root / 'logs'}`。", "",
                  "训练状态说明：本轮使用已有已训练 checkpoint 进行全场景统一测试；checkpoint 来源、SHA256 与原始训练日志保留在 training/ 和既有研究目录中。",
                  ""])
    (root / "full_matrix_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"methods": len(rows), "evaluation_failures": len(failures),
                      "media_failures": len(media_failures),
                      "report": str(root / "full_matrix_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
