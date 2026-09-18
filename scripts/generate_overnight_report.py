"""Build an auditable index for the overnight research run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def sha256(path: Path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path):
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def process_status(pattern: str):
    result = subprocess.run(
        ["bash", "-lc", f"pgrep -af {pattern!r} || true"],
        capture_output=True, text=True, check=False,
    )
    return [line for line in result.stdout.splitlines() if "pgrep -af" not in line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/overnight_research_20260906")
    parser.add_argument("--success-study", default="experiments/success_study_20260905")
    args = parser.parse_args()
    root = Path(args.root)
    success_root = Path(args.success_study)
    root.mkdir(parents=True, exist_ok=True)

    success_report = success_root / "reports" / "aggregate_metrics.json"
    success_data = load_json(success_report, {})
    flow_spread = load_json(root / "controller_pilots" / "flow_spread" / "summary.json", {})

    world_models = {}
    world_model_root = root / "world_models"
    for model_dir in sorted(world_model_root.glob("retrained_*")):
        if not model_dir.is_dir():
            continue
        metrics = load_json(model_dir / "metrics.json")
        config = load_json(model_dir / "config.json", {})
        world_models[model_dir.name] = {
            "status": "gate_passed" if metrics and metrics.get("gate", {}).get("passed") else (
                "metrics_pending" if not metrics else "gate_failed"
            ),
            "config": config,
            "metrics": metrics,
            "checkpoint": file_record(model_dir / "best_world_model.pt", root)
            if (model_dir / "best_world_model.pt").exists() else None,
            "training_log": str((world_model_root / "logs" / f"{model_dir.name}.log").relative_to(root)),
        }

    media_manifest = load_json(root / "media_methods" / "manifest.json", {"count": 0, "files": []})
    media_files = [item for item in media_manifest.get("files", []) if Path(item["path"]).exists()]
    media_by_kind = {
        "gif": sorted(item["path"] for item in media_files if item["path"].lower().endswith(".gif")),
        "mp4": sorted(item["path"] for item in media_files if item["path"].lower().endswith(".mp4")),
    }

    training_dirs = []
    training_root = root / "training"
    if training_root.exists():
        for path in sorted(training_root.glob("*/*")):
            if path.is_dir():
                checkpoints = sorted(path.glob("checkpoint_*.pt"))
                training_dirs.append({
                    "path": str(path.relative_to(root)),
                    "checkpoints": [file_record(item, root) for item in checkpoints],
                    "summary": load_json(path / "summary.json"),
                    "log_candidates": [str(item.relative_to(root)) for item in
                                       sorted((root / "logs").glob(f"*{path.parent.name}*{path.name}*.log"))],
                })

    audit = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "root": str(root),
        "success_study": {
            "source_report": str(success_report),
            "selected_label": success_data.get("selected_label"),
            "selected_checkpoint": success_data.get("selected_checkpoint"),
            "learned_three_seed": success_data.get("learned_three_seed"),
            "summaries": {
                label: {
                    "episodes": value.get("episodes"),
                    "successes": value.get("successes"),
                    "macro": value.get("macro"),
                }
                for label, value in success_data.get("summaries", {}).items()
            },
        },
        "flow_spread_pilot": flow_spread,
        "world_models": world_models,
        "training_runs": training_dirs,
        "media": {
            "manifest": str((root / "media_methods" / "manifest.json").relative_to(root)),
            "gif_count": len(media_by_kind["gif"]),
            "mp4_count": len(media_by_kind["mp4"]),
            "gif": media_by_kind["gif"],
            "mp4": media_by_kind["mp4"],
        },
        "processes": {
            "world_model": process_status("train_world_model"),
            "mappo": process_status("train_vector_mappo"),
        },
        "known_failure_logs": sorted(
            str(path.relative_to(root)) for path in (root / "logs").glob("*.log")
            if "attempt" in path.name or "probe" in path.name
        ),
    }
    (root / "audit_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    learned = audit["success_study"].get("learned_three_seed") or {}
    success_mean = learned.get("success", {}).get("mean")
    removal_mean = learned.get("removal_rate", {}).get("mean")
    lines = [
        "# Overnight 深度强化学习与世界模型研究记录",
        "",
        f"生成时间：{audit['generated_at']}",
        "",
        "## 结论摘要",
        "",
        "- 当前正式最佳方法：流速感知残差 GAT-MAPPO。",
        f"- 三训练种子独立测试成功率：{success_mean:.2%} ± {learned['success']['seed_std']:.2%}。"
        if success_mean is not None else "- 三训练种子正式结果尚未读取。",
        f"- 三训练种子平均质量清除率：{removal_mean:.2%}。"
        if removal_mean is not None else "- 平均质量清除率尚未读取。",
        "- `flow_spread` 为预注册控制器消融，结果低于正式最佳方法，不作为改进结论。",
        "- 世界模型只在验证 gate 通过后进入 MVE；未通过或未完成的分支不计入成功结果。",
        "",
        "## 正式测试结果",
        "",
        "| 方法 | 回合数 | 成功数 | 成功率 | 平均清除率 | 来源 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, value in audit["success_study"]["summaries"].items():
        macro = value.get("macro") or {}
        lines.append(
            f"| {label} | {value.get('episodes', '-') } | {value.get('successes', '-')} | "
            f"{macro.get('success', 0):.2%} | {macro.get('removal_rate', 0):.2%} | "
            "success_study_20260905 |"
        )
    if flow_spread:
        lines.append(
            f"| flow_spread pilot | {flow_spread.get('episodes', '-')} | {flow_spread.get('successes', '-')} | "
            f"{flow_spread.get('macro', {}).get('success', 0):.2%} | "
            f"{flow_spread.get('macro', {}).get('removal_rate', 0):.2%} | overnight isolated eval |"
        )
    lines.extend(["", "## 世界模型验证", "", "| 模型 | 状态 | 一步观测 RMSE | 五步相对 persistence | reward 相对 zero |", "|---|---|---:|---:|---:|"])
    for label, item in world_models.items():
        metrics = item.get("metrics") or {}
        one = metrics.get("one_step", {})
        five = metrics.get("multistep", {}).get("5", {})
        lines.append(
            f"| {label} | {item['status']} | {one.get('obs_rmse', '-') } | "
            f"{five.get('relative_to_persistence', '-')} | {one.get('reward_relative_to_zero', '-')} |"
        )
    lines.extend([
        "",
        "## 媒体与审计文件",
        "",
        f"- GIF：{len(media_by_kind['gif'])} 个；MP4：{len(media_by_kind['mp4'])} 个。",
        f"- 媒体清单：`{root / 'media_methods' / 'manifest.json'}`。",
        f"- 审计 JSON：`{root / 'audit_summary.json'}`。",
        f"- 所有训练/评测日志：`{root / 'logs'}`。",
        "- MP4 是用户所说的 `map4` 的规范文件扩展名；这里同时保存 GIF 与 MP4。",
        "",
        "## 可复现性与失败记录",
        "",
        "- 所有独立评测回合保存为 JSONL，并使用固定场景/seed 生成。",
        "- native 崩溃、恢复配置错误和未完成训练保留原始日志，不转写为成功结果。",
        "- 选择 checkpoint 只使用验证集；held-out 测试结果不参与选择。",
        "",
    ])
    (root / "research_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "report": str(root / "research_report.md"),
        "audit": str(root / "audit_summary.json"),
        "gif_count": len(media_by_kind["gif"]),
        "mp4_count": len(media_by_kind["mp4"]),
        "world_models": {key: value["status"] for key, value in world_models.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
