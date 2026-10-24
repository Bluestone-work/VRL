"""Refresh complete physical replays of the three EXP0114 full models."""
from __future__ import annotations
import json
from pathlib import Path
import time
import traceback

from scripts.visualize_sb3_vascular import checkpoint_info, record_replay
from scripts.run_sb3_curriculum_study import ROOT, atomic, read, rows, stamp


def refresh(run, out, manifest):
    algorithms = []
    for algo in ("ppo", "sac", "td3"):
        directory = run / f"{algo}_full_s41"
        status = read(directory / "status.json")
        completion = read(directory / "TRAINING_COMPLETE.json")
        episodes = rows(directory / "episodes.jsonl")
        total = 0
        for row in episodes:
            total += row["environment_steps"]
            row["training_environment_step"] = total
        steps = max(status.get("environment_steps", 0), completion.get("environment_steps", 0))
        algorithms.append(dict(algorithm=algo, steps=steps, target=131072,
            stage="已完成开发评估" if (directory / "RUN_COMPLETE.json").exists() else "训练完成，评估中" if completion else "训练中",
            episode_count=len(episodes), episodes=episodes, strict_success=sum(bool(r["strict_success"]) for r in episodes),
            mean_removal=sum(r["removal"] for r in episodes) / len(episodes) if episodes else None,
            device=read(directory / "config.json").get("device", "cpu"), updates=completion.get("updates"),
            progress_curve=[[status.get("minutes", 0), steps]]))
    status = dict(updated=stamp(), algorithms=algorithms)
    data = dict(status=status, manifest=manifest)
    atomic(out / "status.json", data)
    data["replays"] = {algo: read(out / meta["json"]) for algo, meta in manifest["replays"].items()}
    template = (ROOT / "scripts/sb3_dashboard.html").read_text()
    template = (template.replace("EXP0113", "EXP0114")
                .replace("SB3 训练与三维回放", "课程与预测模型 · 三维回放")
                .replace("3 个机器人 &nbsp;·&nbsp; 16 个动态障碍 &nbsp;·&nbsp; 开发预实验", "困难遭遇场景 · 3 机器人 · 16 动态障碍")
                .replace("从随机参数学习连续动作", "CNN + Transformer · 课程与因果障碍预测")
                .replace("机器人与血栓标记适当放大。", "橙色虚线为选中机器人感知的障碍未来 1 秒预测；标记适当放大。")
                .replace("training_curves.png", "../training_curves.png")
                .replace("fetch('/api/status'", "fetch('status.json'")
                .replace("http://127.0.0.1:8873/", "http://127.0.0.1:8875/replay/")
                .replace("当前为一个训练种子的预实验", "此页为种子 41 完整版；训练难度动态变化。当前为开发预实验"))
    page = template.replace("__EMBEDDED_DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    temporary = out / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(out / "index.html")


def loop(run, output):
    out = output / "replay"
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(out / "manifest.json", dict(replays={}, errors={}))
    while True:
        try:
            refresh(run, out, manifest)
            for algo in ("ppo", "sac", "td3"):
                directory = run / f"{algo}_full_s41"
                paths = list(directory.glob("policy_steps*.zip"))
                paths.extend(p for p in (directory / "initial_model.zip", directory / "final_model.zip") if p.exists())
                models = [meta for path in paths if (meta := checkpoint_info(path)) is not None]
                if not models:
                    continue
                checkpoint = max(models, key=lambda m: m["steps"])
                if checkpoint["sha256"] == manifest["replays"].get(algo, {}).get("sha256"):
                    continue
                checkpoint.update(algorithm=algo, origin="正式课程实验冻结模型" if checkpoint["steps"] else "正式实验随机初始化模型（0 步，尚未学习）")
                manifest["replays"][algo] = record_replay(checkpoint, out)
                atomic(out / "manifest.json", manifest)
                refresh(run, out, manifest)
        except Exception:
            print(traceback.format_exc(), flush=True)
        time.sleep(60)
