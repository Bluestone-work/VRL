"""Supervise the finite EXP0114 experiment queue and serve its live dashboard."""
from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "research/runs/EXP0114_CURRICULUM_20261015"
DEFAULT_OUT = Path("/home/wj/桌面/VRL_课程学习与预测_20261015")


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default if default is not None else {}


def rows(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def plan(steps):
    specs = [("ppo", "full", 41, "cpu"), ("sac", "full", 41, "cuda:0"),
             ("td3", "full", 41, "cuda:1"), ("ppo", "no_prediction", 41, "cpu"),
             ("ppo", "no_curriculum", 41, "cpu"), ("ppo", "no_shaping", 41, "cpu")]
    # Predeclared paired seeds, not a selective rerun of the winning seed.
    specs.extend(("ppo", variant, seed, "cpu") for seed in (42, 43)
                 for variant in ("full", "no_prediction", "no_curriculum"))
    return [dict(id=f"{algo}_{variant}_s{seed}", algorithm=algo, variant=variant,
                 seed=seed, device=device, steps=steps, state="queued")
            for algo, variant, seed, device in specs]


def snapshot(run, jobs):
    result = []
    for job in jobs:
        directory = run / job["id"]
        result.append(dict(job, progress=read(directory / "status.json"),
                           training=read(directory / "TRAINING_COMPLETE.json"),
                           evaluation=read(directory / "RUN_COMPLETE.json"),
                           episodes=rows(directory / "episodes.jsonl"),
                           evaluation_rows=rows(directory / "evaluation.jsonl")))
    comparisons = []
    # Compare each ablation to its same-seed full model on identical scenes.
    for seed in (41, 42, 43):
        full = next((r for r in result if r["id"] == f"ppo_full_s{seed}"), {})
        control = {(r["anatomy"], r["difficulty"]): r for r in full.get("evaluation_rows", [])}
        for variant in ("no_prediction", "no_curriculum", "no_shaping"):
            ablation = next((r for r in result if r["id"] == f"ppo_{variant}_s{seed}"), {})
            paired = [(control[(r["anatomy"], r["difficulty"])], r)
                      for r in ablation.get("evaluation_rows", []) if (r["anatomy"], r["difficulty"]) in control]
            if paired:
                hashes_match = all(a["scenario_hash"] == b["scenario_hash"] for a, b in paired)
                comparisons.append(dict(seed=seed, ablation=variant, paired_scenes=len(paired),
                    scenario_hashes_match=hashes_match,
                    full_minus_ablation={key: sum(float(a[key]) - float(b[key]) for a, b in paired) / len(paired)
                    for key in ("removal", "particle_contact_s", "particle_collision_events", "wall_contact_s", "strict_success")},
                    control_collision_events=sum(b["particle_collision_events"] for a, b in paired),
                    interpretation="development comparison; no reduction evidence when control has zero collisions"))
    return dict(updated=stamp(), jobs=result, comparisons=comparisons,
                note="开发实验；训练回合按难度区分。预测为因果运动外推，尚非训练好的预测网络。")


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    palette = {"full": "#087e8b", "no_prediction": "#bc6356", "no_curriculum": "#8a65ab", "no_shaping": "#bd9a39"}
    for job in data["jobs"]:
        if job["seed"] != 41:
            continue
        label = job["id"].replace("_s41", "")
        color = palette[job["variant"]] if job["algorithm"] == "ppo" else {"sac": "#3779ba", "td3": "#63774d"}[job["algorithm"]]
        episodes = job["episodes"]
        steps, x = 0, []
        for episode in episodes:
            steps += episode["environment_steps"]
            x.append(steps)
        if episodes:
            axes[0, 0].plot(x, [r["removal"] * 100 for r in episodes], ".-", color=color, label=label)
            axes[0, 1].plot(x, [r["particle_contact_s"] for r in episodes], ".-", color=color, label=label)
            axes[1, 0].plot(x, [r["difficulty"] for r in episodes], ".-", color=color, label=label)
        evaluation = job["evaluation"].get("by_difficulty", {})
        if evaluation:
            axes[1, 1].plot([int(k) for k in evaluation], [r["removal"] * 100 for r in evaluation.values()], ".-", color=color, label=label)
    for ax, title, xlabel, ylabel in zip(axes.flat,
            ("Training removal (episode difficulty varies)", "Training dynamic-obstacle contact", "Sampled training difficulty", "Fixed paired development evaluation"),
            ("Environment steps", "Environment steps", "Environment steps", "Difficulty"),
            ("Removal (%)", "Contact (robot seconds)", "Difficulty 0..3", "Removal (%)")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    axes[1, 0].set_yticks(range(4))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def publish(run, out, jobs, graphs=False):
    data = snapshot(run, jobs)
    atomic(out / "status.json", data)
    template = (ROOT / "scripts/sb3_curriculum_dashboard.html").read_text()
    content = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    tmp = out / "index.html.tmp"
    tmp.write_text(content)
    tmp.replace(out / "index.html")
    if graphs:
        plot(data, out / "training_curves.png")
    atomic(run / "comparison.json", dict(updated=data["updated"], comparisons=data["comparisons"]))


def supervise(run, out, steps):
    state = run / "jobs.json"
    existing = read(state)
    jobs = existing.get("jobs") or plan(steps)
    children = {}
    last_plot = 0.
    for job in jobs:
        if job["state"] == "running" and not alive(job.get("pid", -1)):
            job["state"] = "complete" if (run / job["id"] / "RUN_COMPLETE.json").exists() else "failed"
    while True:
        for job in jobs:
            if job["state"] != "running":
                continue
            proc = children.get(job["id"])
            code = proc.poll() if proc is not None else None
            if (proc is not None and code is not None) or (proc is None and not alive(job["pid"])):
                job.update(state="complete" if (run / job["id"] / "RUN_COMPLETE.json").exists() else "failed",
                           exit_code=code, finished=stamp())
        running = [j for j in jobs if j["state"] == "running"]
        busy = {j["device"] for j in running}
        for job in jobs:
            if len(running) >= 3:
                break
            if job["state"] != "queued" or job["device"] in busy:
                continue
            directory = run / job["id"]
            cmd = [sys.executable, "-u", "-m", "scripts.train_sb3_curriculum", "--algorithm", job["algorithm"],
                   "--out", str(directory), "--seed", str(job["seed"]), "--timesteps", str(job["steps"]),
                   "--device", job["device"], "--eval-count", "36"]
            flags = {"no_prediction": "--no-prediction", "no_curriculum": "--no-curriculum", "no_shaping": "--no-shaping"}
            if job["variant"] in flags:
                cmd.append(flags[job["variant"]])
            if directory.exists():
                job.update(state="failed", error="output already exists; refusing overwrite")
                continue
            with (run / (job["id"] + ".log")).open("a") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[job["id"]] = proc
            job.update(state="running", pid=proc.pid, command=cmd, started=stamp())
            running.append(job)
            busy.add(job["device"])
        atomic(state, dict(updated=stamp(), jobs=jobs))
        try:
            publish(run, out, jobs, graphs=time.monotonic() - last_plot >= 60)
            if time.monotonic() - last_plot >= 60:
                last_plot = time.monotonic()
        except Exception:
            print(traceback.format_exc(), flush=True)
        if all(j["state"] in ("complete", "failed", "cancelled") for j in jobs):
            atomic(run / "STUDY_FINISHED.json", dict(finished=stamp(), completed=sum(j["state"] == "complete" for j in jobs),
                                                     failed=sum(j["state"] == "failed" for j in jobs)))
            return
        time.sleep(10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--steps", type=int, default=131072)
    p.add_argument("--port", type=int, default=8875)
    a = p.parse_args()
    if a.steps < 512:
        p.error("at least 512 steps required")
    a.run.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    service = read(a.out / "service.json")
    if alive(service.get("pid", -1)):
        raise RuntimeError("study supervisor already running")
    smoke = ROOT / "research/runs/EXP0114_CURRICULUM_SMOKE_20261015"
    for algo in ("ppo", "sac", "td3"):
        proof = read(smoke / algo / "TRAINING_COMPLETE.json")
        if not proof.get("updates", 0) or not proof.get("parameter_change_l2", 0):
            raise RuntimeError(f"missing successful {algo} smoke update")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), partial(SimpleHTTPRequestHandler, directory=str(a.out)))
    atomic(a.out / "service.json", dict(pid=os.getpid(), url=f"http://127.0.0.1:{a.port}/", started=stamp()))
    worker = threading.Thread(target=supervise, args=(a.run, a.out, a.steps), daemon=True)
    worker.start()
    from scripts.visualize_sb3_curriculum import loop as replay_loop
    threading.Thread(target=replay_loop, args=(a.run, a.out), daemon=True).start()
    print(f"http://127.0.0.1:{a.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
