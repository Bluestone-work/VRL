"""Snapshot completed EXP29 evaluations without touching the live experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize(path, policy_seed, expected_seeds):
    raw = path.read_bytes()
    data = json.loads(raw)
    episodes = data["episodes"]
    assert [r["seed"] for r in episodes] == expected_seeds, path
    n = len(episodes)
    successes = sum(bool(r["success"]) for r in episodes)
    safe = sum(bool(r["success"]) and r["particle_contact_s"] <= 1e-12
               and r["lost_robots"] == 0 for r in episodes)
    assert safe == sum(bool(r["collision_free_success"]) for r in episodes), path
    collisions = sum(r["particle_contact_s"] > 1e-12 for r in episodes)
    recomputed = {
        "success_rate": successes / n,
        "collision_free_success_rate": safe / n,
        "particle_collision_episode_rate": collisions / n,
        "mean_removal_fraction": statistics.mean(r["removal_fraction"] for r in episodes),
    }
    for key, value in recomputed.items():
        assert abs(data[key] - value) < 1e-9, (path, key)
    reset_hash = digest([r["reset_info"] for r in episodes])
    return dict(policy_seed=policy_seed, transitions=int(path.stem.split("_")[-1]),
                episodes=n, successes=successes, collision_free_successes=safe,
                particle_collision_episodes=collisions,
                robot_loss_episodes=sum(r["lost_robots"] > 0 for r in episodes),
                termination_reasons=dict(Counter(r["reason"] for r in episodes)),
                validation_layouts_sha256=reset_hash, source=str(path.resolve()),
                source_sha256=hashlib.sha256(raw).hexdigest(), **recomputed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    expected = list(range(protocol["validation_seed_base"],
                          protocol["validation_seed_base"] + protocol["eval_episodes"]))
    statuses, results = {}, []
    for seed in protocol["seeds"]:
        seed_dir = args.run_dir / f"seed_{seed}"
        status = json.loads((seed_dir / "status.json").read_text())
        statuses[str(seed)] = status
        for path in sorted(seed_dir.glob("evaluation_*.json")):
            results.append(summarize(path, seed, expected))
    assert results, "No completed evaluations"
    assert len({r["validation_layouts_sha256"] for r in results}) == 1, "Layout mismatch"
    results.sort(key=lambda r: (r["transitions"], r["policy_seed"]))
    steps_by_seed = [{r["transitions"] for r in results if r["policy_seed"] == s}
                     for s in protocol["seeds"]]
    common_steps = sorted(set.intersection(*steps_by_seed))
    assert common_steps, "No common completed checkpoint"
    common = [r for r in results if r["transitions"] == common_steps[-1]]
    aggregate = {}
    for key in ("success_rate", "collision_free_success_rate", "particle_collision_episode_rate"):
        values = [r[key] for r in common]
        aggregate[key] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    snapshot = dict(as_of=now, run_dir=str(args.run_dir.resolve()),
                    user_clearance_target=0.80, statuses=statuses, evaluations=results,
                    common_checkpoint=common_steps[-1], common_checkpoint_aggregate=aggregate,
                    clearance_target_met_all_seeds=all(r["success_rate"] >= .80 for r in common),
                    best_observed_clearance=max(r["success_rate"] for r in results),
                    interpretation="Development validation on one anatomy; the same layouts are reused across policies/checkpoints. Seed means are descriptive; episodes are not 300 independent policy trainings. No sealed test, no eventual-performance guarantee.")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    lines = ["# EXP29 随机布局进度审计", "", f"读取时间：{now}", "",
             "结论：当前已完成评估未达到用户此前提出的80%完整清栓目标；无动态粒子接触且全部机器人保留的清栓率更低。不能承诺剩余训练一定达标。", "",
             "## 最新共同检查点", "", f"各训练seed新增 {common_steps[-1]:,} 环境步，各评估100个相同的未用于训练的布局。", "",
             "| 训练seed | 完整清栓 | 无粒子碰撞且全机器人保留的清栓 | 发生粒子碰撞 | 平均清除质量比例 |",
             "|---|---:|---:|---:|---:|"]
    for r in common:
        lines.append(f"| {r['policy_seed']} | {r['successes']}/{r['episodes']} | {r['collision_free_successes']}/{r['episodes']} | {r['particle_collision_episodes']}/{r['episodes']} | {r['mean_removal_fraction']:.1%} |")
    lines += ["", "统计单位为3个独立训练seed；同一100种布局在策略之间配对复用，不将300次评估当成300次独立训练。", "",
              "## 全部已完成评估", "", "| 环境步 | 训练seed | 完整清栓 | 无粒子碰撞清栓 | 粒子碰撞回合 |",
              "|---:|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['transitions']:,} | {r['policy_seed']} | {r['success_rate']:.0%} | {r['collision_free_success_rate']:.0%} | {r['particle_collision_episode_rate']:.0%} |")
    lines += ["", "## 解释边界", "",
              "- 完整清栓按一回合全部四个血栓清除计数；平均去除质量不能替代该成功率。",
              "- 无粒子碰撞清栓还要求5个机器人全保留；该指标不包含零壁接触或零机器人间接触的保证。",
              "- 血栓、机器人、粒子初始位置与流量随机；血管解剖仍固定。相同seed可复现。",
              "- 当前是开发验证集，反复用于观察检查点，不是密封最终测试集，也未验证跨解剖泛化。",
              "- 当前训练从EXP28权重迁移而来，表中步数是EXP29新增步数，不是累计全部样本预算。",
              "- 不混合不同训练进度来报告三seed均值；未完成评估不填0、不外推。",
              "- 明显的检查点退步说明表现尚不稳定；仅凭这些日志不能确认退步的算法根因。",
              "- 本次只新增审计与研究说明；未修改物理参数、奖励、运行源码、训练进程或检查点。", "",
              "原始文件绝对路径和SHA256、同布局校验、逐回合复算及读取时运行状态见 snapshot.json。", ""]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    colors = ("#2563eb", "#c65d17", "#168068")
    for ax, metric, title in zip(axes, ("success_rate", "collision_free_success_rate"),
                                 ("All four clots cleared", "Cleared, no particle contact, all robots retained")):
        for seed, color in zip(protocol["seeds"], colors):
            rows = [r for r in results if r["policy_seed"] == seed]
            ax.plot([r["transitions"] / 1000 for r in rows], [r[metric] * 100 for r in rows],
                    marker="o", color=color, label=f"Policy seed {seed}")
        ax.set(title=title, xlabel="Additional EXP29 environment steps (thousands)", ylim=(0, 105))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].axhline(80, color="#b91c1c", linestyle="--", alpha=.7)
    axes[0].text(.98, .78, "Requested clearance target: 80%", transform=axes[0].transAxes,
                 ha="right", fontsize=8, color="#b91c1c")
    axes[0].set_ylabel("Completed evaluation success (%)")
    fig.suptitle("EXP29 | 100 matched validation layouts/checkpoint | Fixed anatomy", fontsize=12)
    fig.text(.5, .01, f"Snapshot: {now} | Missing evaluation points are pending, not zero", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(args.out_dir / "evaluation_progress.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: snapshot[k] for k in ("as_of", "common_checkpoint", "common_checkpoint_aggregate", "best_observed_clearance")}, indent=2))


if __name__ == "__main__":
    main()
