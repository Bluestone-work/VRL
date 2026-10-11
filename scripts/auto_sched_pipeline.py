"""Detached monitor for Scheduled-Settle experiments.

The monitor waits for an already running training/evaluation shell, produces the
diagnostics and paper scene bundle, and can continue with a GRU history variant
when the learned controller does not beat the paired SwitchSettle rule.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def say(log, message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def run(cmd, cwd, log, timeout=None, env=None):
    say(log, "$ " + " ".join(map(str, cmd)))
    with log_path_for(log).open("a") as out:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
    say(log, f"exit={p.returncode}")
    return p.returncode


def log_path_for(log):
    # ``say`` keeps the file open for timestamps, while subprocesses need a
    # separate append handle.  Derive the path from the monitor's fixed root.
    return Path(os.environ.get("SCHED_MONITOR_LOG", "research/automation/SCHED_AUTO_20261010/monitor.log"))


def pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid, log, poll=30):
    if not pid:
        return
    say(log, f"waiting for existing pipeline pid={pid}")
    while pid_alive(pid):
        time.sleep(poll)
    say(log, f"pipeline pid={pid} finished")


def wait_for_file(path, log, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(10)
    say(log, f"timed out waiting for {path}")
    return False


def snapshot(repo, run_dir):
    dst = run_dir / "source_snapshot_auto"
    dst.mkdir(parents=True, exist_ok=True)
    files = ["marl/sched_settle.py", "scripts/train_sched_settle.py",
             "scripts/evaluate_hard_baselines.py", "scripts/plot_sched_training.py",
             "scripts/report_sched_settle.py", "scripts/render_lysis_paper.py",
             "scripts/auto_sched_pipeline.py"]
    for rel in files:
        src = repo / rel
        if src.exists():
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def evaluate_and_visualize(repo, run_name, log, scene_seed=2501111130):
    run_dir = repo / "research/runs/SCHED_20261010" / run_name
    val_dir = repo / "research/validation/SCHED_20261010" / run_name
    fig_dir = repo / "research/figures/SCHED_AUTO_20261010" / run_name
    val_dir.parent.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    snapshot(repo, run_dir)
    env = os.environ.copy(); env["PYTHONPATH"] = "."
    policy = run_dir / "policy.pt"
    if not policy.exists():
        say(log, f"missing checkpoint: {policy}")
        return None
    episodes = val_dir / "episodes.jsonl"
    if not episodes.exists():
        # evaluate_hard_baselines deliberately requires a fresh output
        # directory; a previous interrupted attempt may have left an empty one.
        if val_dir.exists() and not any(val_dir.iterdir()):
            val_dir.rmdir()
        cmd = [sys.executable, "-m", "scripts.evaluate_hard_baselines",
               "--out", str(val_dir), "--methods", "sched_settle", "--policy", str(policy),
               "--panels", "low_delay,moderate_delay,high_delay,strong_flow,variable_response,ood_flow_delay",
               "--workers", "20"]
        rc = run(cmd, repo, log, timeout=8 * 3600, env=env)
        if rc != 0 and not episodes.exists():
            say(log, "evaluation failed; aborting automatic continuation")
            return None
    if not wait_for_file(episodes, log, timeout=60):
        return None
    run([sys.executable, "-m", "scripts.plot_sched_training", "--log",
         str(run_dir / "log.jsonl"), "--out", str(fig_dir)], repo, log, timeout=1800, env=env)
    rule = repo / "research/validation/SCHED_20261010/switch_rule"
    run([sys.executable, "-m", "scripts.report_sched_settle", "--eval", str(val_dir),
         "--rule", str(rule), "--out", str(fig_dir)], repo, log, timeout=1800, env=env)
    scene = fig_dir / "scene"
    render = [sys.executable, "-m", "scripts.render_lysis_paper", "--out", str(scene),
              "--anatomy", "sma_embolism", "--clusters", "3", "--seed", str(scene_seed),
              "--flow", "0.075", "--latency", "3", "--variation", "0.625",
              "--methods", f"switch,sched:{policy}", "--times", "20,60,120,end", "--video"]
    env["LYSIS_DISABLE_EGL"] = "1"
    rc_render = run(render, repo, log, timeout=3600, env=env)
    (fig_dir / "scene_render_status.json").write_text(json.dumps({"returncode": rc_render,
        "egl_disabled": True, "note": "TinyRenderer fallback selected by automation"}, indent=2))
    try:
        aggregate = json.loads((fig_dir / "aggregate.json").read_text())
        say(log, "paired result " + json.dumps(aggregate.get("overall", {}), sort_keys=True))
        return aggregate
    except Exception as exc:
        say(log, f"could not parse aggregate.json: {exc}")
        return None


def is_good(aggregate):
    if not aggregate:
        return False
    overall = aggregate.get("overall", {})
    if float(overall.get("delta_strict_pp", -999.)) <= 0.0:
        return False
    # Require the learned method to avoid a regression in every hard panel.
    tables = aggregate.get("tables", [])
    rules = {r.get("panel"): float(r.get("metrics", {}).get("strict_success", 0.))
             for r in tables if r.get("method") == "SwitchSettle"}
    for row in tables:
        if row.get("method") == "FlowResidual":
            panel = row.get("panel")
            learner = float(row.get("metrics", {}).get("strict_success", 0.))
            if panel in rules and learner + 1e-12 < rules[panel]:
                return False
    return True


def launch_training(repo, run_name, seed, arch, log, minutes=90):
    out = repo / "research/runs/SCHED_20261010" / run_name
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env["PYTHONPATH"] = "."
    cmd = [sys.executable, "-m", "scripts.train_sched_settle", "--out", str(out),
           "--prior", "flow", "--minutes", str(minutes), "--workers", "18",
           "--seed", str(seed), "--device", "cuda:0", "--lr", "1e-4",
           "--beta0", ".2", "--beta-end", ".02", "--arch", arch]
    say(log, "launching continuation " + " ".join(cmd))
    train_log = out / "train.stdout.log"
    with train_log.open("w") as stream:
        proc = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
    say(log, f"continuation pid={proc.pid}")
    while proc.poll() is None:
        time.sleep(60)
        say(log, f"continuation pid={proc.pid} still running")
    say(log, f"continuation exit={proc.returncode}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--pipeline-pid", type=int, default=0)
    ap.add_argument("--max-continuations", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=90)
    args = ap.parse_args()
    repo = args.repo.resolve(); root = repo / "research/automation/SCHED_AUTO_20261010"
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "monitor.log"
    os.environ["SCHED_MONITOR_LOG"] = str(log_path)
    with log_path.open("a") as log:
        say(log, f"monitor started repo={repo} initial={args.initial}")
        wait_for_pid(args.pipeline_pid, log)
        aggregate = evaluate_and_visualize(repo, args.initial, log)
        current = aggregate
        for k in range(args.max_continuations):
            if is_good(current):
                say(log, "quality gate passed; no continuation required")
                break
            if current is None:
                say(log, "quality gate unavailable because evaluation failed; stopping")
                break
            idx = 6 + k
            while (repo / "research/runs/SCHED_20261010" / f"flowres_gru_s{idx}").exists():
                idx += 1
            name = f"flowres_gru_s{idx}"
            say(log, f"quality gate failed; starting {name}")
            rc = launch_training(repo, name, seed=2+k, arch="gru", log=log, minutes=args.minutes)
            if rc != 0:
                say(log, f"continuation failed with exit={rc}")
                break
            current = evaluate_and_visualize(repo, name, log, scene_seed=2501111130 + k + 1)
        say(log, "monitor finished")


if __name__ == "__main__":
    main()
