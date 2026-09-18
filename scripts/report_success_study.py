"""Aggregate immutable paired evaluations without selecting on test scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def paired_comparison(candidate_records, reference_records, seed=20260905):
    keys = sorted((row["scenario"], row["seed"]) for row in reference_records)
    reference = {(row["scenario"], row["seed"]): row for row in reference_records}
    if not reference or len(reference) != len(reference_records):
        raise ValueError("reference requires nonempty unique episode keys")
    candidates = []
    for records in candidate_records:
        lookup = {(row["scenario"], row["seed"]): row for row in records}
        if len(lookup) != len(records) or set(lookup) != set(reference):
            raise ValueError("paired evaluations must contain identical unique episode keys")
        candidates.append(lookup)
    differences = np.asarray([
        [lookup[key]["success"] - reference[key]["success"] for key in keys]
        for lookup in candidates
    ], dtype=np.float64)
    territories = sorted({key[0] for key in keys})
    groups = [np.asarray([index for index, key in enumerate(keys) if key[0] == territory])
              for territory in territories]
    rng = np.random.default_rng(seed)
    draws = np.empty(5000)
    for draw in range(len(draws)):
        training_indices = rng.integers(0, len(candidates), size=len(candidates))
        episode_indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        draws[draw] = differences[training_indices][:, episode_indices].mean()
    return {
        "success_difference": float(differences.mean()),
        "paired_bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_draws": len(draws),
        "method": "resample training seeds and paired episodes within fixed territories",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="experiments/success_study_20260905")
    args = parser.parse_args()
    root = Path(args.study)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    labels = ["baseline_mve43", "local_42", "guided_42", "flow_controller",
              "flow_guided_42", "flow_guided_43", "flow_guided_44"]
    summaries = {label: read_json(root / "heldout" / label / "summary.json") for label in labels}
    records = {label: read_records(root / "heldout" / label / "episodes.jsonl") for label in labels}
    selections = {label: read_json(root / "selection" / f"{label}.json")
                  for label in labels if (root / "selection" / f"{label}.json").exists()}
    learned_labels = [f"flow_guided_{seed}" for seed in (42, 43, 44)]
    metric_names = tuple(summaries[labels[0]]["macro"])
    aggregate = {}
    for metric in metric_names:
        values = [summaries[label]["macro"][metric] for label in learned_labels]
        aggregate[metric] = {"mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1))}
    comparisons = {
        reference: paired_comparison([records[label] for label in learned_labels], records[reference])
        for reference in ("baseline_mve43", "flow_controller")
    }
    selected_label = max(learned_labels, key=lambda label: (
        selections[label]["selected_evaluation"]["macro"]["success"],
        selections[label]["selected_evaluation"]["macro"]["removal_rate"],
    ))
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "selected_label": selected_label, "selection_source": "validation only",
        "selected_checkpoint": selections[selected_label]["checkpoint"],
        "summaries": summaries, "learned_three_seed": aggregate,
        "paired_comparisons": comparisons, "selections": selections,
    }
    (reports / "aggregate_metrics.json").write_text(json.dumps(result, indent=2))
    rows = [{"arm": label, "episodes": summaries[label]["episodes"],
             "successes": summaries[label]["successes"], **summaries[label]["macro"]}
            for label in labels]
    with (reports / "metrics_by_arm.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    territory_rows = [{"arm": label, "scenario": scenario, **metrics}
                      for label in labels for scenario, metrics in summaries[label]["per_territory"].items()]
    with (reports / "metrics_by_territory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(territory_rows[0]))
        writer.writeheader()
        writer.writerows(territory_rows)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = ["Previous MVE\nseed 43", "Local MAPPO\nseed 42", "Route residual\nseed 42",
             "Flow controller\n(no training)", "Flow residual\n3 training seeds"]
    for axis, metric, title in zip(axes, ("success", "removal_rate", "wall_hits_per_step"),
                                   ("Complete clot clearance (%)", "Removed mass (%)", "Wall hits / step")):
        factor = 100 if metric != "wall_hits_per_step" else 1
        values = [summaries[label]["macro"][metric] * factor for label in labels[:4]]
        values.append(aggregate[metric]["mean"] * factor)
        bars = axis.bar(range(5), values, color=["#7b8794", "#577590", "#8d6cab", "#f4a261", "#2a9d8f"])
        axis.errorbar(4, values[-1], yerr=aggregate[metric]["seed_std"] * factor,
                      color="black", capsize=4, fmt="none")
        axis.set_xticks(range(5), names, rotation=25, ha="right", fontsize=8)
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(0, max(values) * 1.25)
    figure.suptitle("Paired held-out seeds: 14 anatomical scenarios x 20 episodes; unchanged physics")
    figure.tight_layout()
    figure.savefig(reports / "heldout_comparison.png", dpi=160)
    plt.close(figure)

    territories = list(summaries["flow_controller"]["per_territory"])
    values = np.asarray([[summaries[label]["per_territory"][scenario]["success"] * 100
                          for scenario in territories] for label in labels])
    figure, axis = plt.subplots(figsize=(14, 5))
    image = axis.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(range(len(territories)), territories, rotation=50, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.0f}", ha="center", va="center",
                      color="white" if value >= 65 else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label="Episode success (%)")
    axis.set_title("Held-out success by anatomical scenario (20 episodes per cell)")
    figure.tight_layout()
    figure.savefig(reports / "success_by_territory.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label in learned_labels:
        seed = label.rsplit("_", 1)[1]
        training = read_records(root / "training" / "flow_guided" / f"seed_{seed}" / "eval_metrics.jsonl")
        for axis, metric in zip(axes, ("success", "removal_rate")):
            axis.plot([row["transitions"] for row in training],
                      [row["macro"][metric] * 100 for row in training], marker="o", label=label)
            axis.set_xlabel("Real training transitions")
            axis.set_ylabel(f"Validation {metric} (%)")
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle("Validation curves (3 episodes / scenario); not held-out scores")
    figure.tight_layout()
    figure.savefig(reports / "learning_curves.png", dpi=160)
    plt.close(figure)

    lines = ["# 血管多智能体成功率改进实验", "", f"生成时间：{result['generated_at']}", "",
             "## 评估协议", "",
             "- 每个策略在 14 个解剖场景各运行 20 回合；测试 seed 为 700000 + 场景序号×10000 + 回合序号。",
             "- 场景类别与训练相同，但几何/随机种子独立；不是未见过的解剖类别泛化。",
             "- 5 个机器人、名义 3 个血栓（每回合随机 2–4）、horizon=300、半径=0.0011；动力学、奖励、成功条件不变。",
             "- 成功严格定义为本回合所有血栓质量清零；清除质量比例不是成功率。",
             "- checkpoint 和推荐训练 seed 仅按验证集 success/removal 选择，测试集不参与选择。",
             "- 上轮 6.67% 是旧测试集的三训练种子平均，不能直接与本轮单策略比较；下表重测上轮强 seed43 MVE 策略。",
             "", "## 独立测试结果", "",
             "| 策略 | 成功 / 回合 | 成功率 | 清除率 | 撞壁/步 | 无接触率 |",
             "|---|---:|---:|---:|---:|---:|"]
    for label in labels:
        summary = summaries[label]
        metrics = summary["macro"]
        lines.append(f"| {label} | {summary['successes']}/{summary['episodes']} | {metrics['success']:.2%} | "
                     f"{metrics['removal_rate']:.2%} | {metrics['wall_hits_per_step']:.3f} | {metrics['contact_miss']:.2%} |")
    lines += ["", f"流速感知残差 MAPPO 三训练种子成功率：**{aggregate['success']['mean']:.2%} ± "
              f"{aggregate['success']['seed_std']:.2%}**（± 为训练种子样本标准差，不是置信区间）。", ""]
    for reference, comparison in comparisons.items():
        lower, upper = comparison["paired_bootstrap95"]
        lines.append(f"- 对比 {reference}：成功率差 {comparison['success_difference'] * 100:+.2f} 个百分点；"
                     f"配对分层 bootstrap 95% 区间 [{lower * 100:+.2f}, {upper * 100:+.2f}]。")
    lines += ["", "## 改动与归因", "",
              "1. 原几何观测的方向向量在 Frenet 局部坐标系，但旧动作按世界坐标执行。新增显式局部到全局转换，保留旧 checkpoint 的 world 默认值。",
              "2. 中心线巡航并不足够：诊断中部分局部血流约 0.05–0.064，超过最大主动速度 0.018，机器人被冲离接触区。",
              "3. flow_controller 仅用已有观测和固定模拟器标度，估计径向流速变化并选择低流速位置；不读取隐藏血栓状态、不改动力学。",
              "4. flow_guided 在该控制器上训练有界残差，PPO buffer 保留原始采样动作及其 log-prob，环境收到变换后的动作。",
              "5. 单纯几何引导 pilot 仅 1/42 成功；流速感知控制器 pilot 为 15/42。这些失败/探索结果保存在 pilot/，不能当独立测试。",
              "6. 撞壁计数累计整个回合；旧报告的 wall_hits 是末步计数，不能解释为回合累计。修正通用评估脚本的同类统计和 contact-miss。",
              "", "## 训练预算与推荐 checkpoint", "",
              "- local/guided 各 seed42：500032 个真实 transition；flow_guided 各 seed42/43/44：1000000 个真实 transition。",
              "- 所有新训练从头开始；没有把旧 world-frame 权重直接当局部动作权重。",
              "- 当前比较不是全部等预算实验；学习收益需首先对照无训练 flow_controller，而不是仅归因于 MAPPO。",
              "- 旧世界模型仍表示 world-frame 动作，本轮未使用；代码拒绝在局部/残差模式中误用旧模型。",
              f"- 推荐：`{result['selected_checkpoint']}`；由验证集选中 `{selected_label}`。", "",
              "| 训练 seed | 验证选择 transition | 验证成功率 |", "|---|---:|---:|"]
    for label in learned_labels:
        selected = selections[label]["selected_evaluation"]
        lines.append(f"| {label} | {selected['transitions']} | {selected['macro']['success']:.2%} |")
    hardest = sorted((
        (scenario, float(np.mean([
            summaries[label]["per_territory"][scenario]["success"] for label in learned_labels
        ]))) for scenario in territories
    ), key=lambda item: item[1])[:3]
    lines += ["", "## 仍然薄弱的场景", ""]
    lines.extend(f"- {scenario}：三训练种子平均成功率 {success:.2%}。"
                 for scenario, success in hardest)
    retry_path = root / "logs" / "evaluation_retries.log"
    retry_count = len(retry_path.read_text().splitlines()) if retry_path.exists() else 0
    lines += ["", "## 测试与运行稳定性", "",
              "- 本轮曾完整通过全量测试：136 passed、1 skipped，见 logs/pytest_final.log。",
              "- 首次全量测试有一次既有 squashed-Gaussian 浮点一致性断言失败；未修改该测试或放宽阈值，单项和全量复跑通过。",
              f"- 独立评估记录 {retry_count} 次 native 异常退出；所有样本按原 seed 续跑，没有跳过崩溃回合。",
              "- 此轮没有根治底层 native 崩溃；异常栈、退出码和每次重试日志全部保留。"]
    release_logs = [root / "logs" / "pytest_release.log"] + list((root / "logs").glob("pytest_release_attempt_*.log"))
    release_crashes = sum(path.exists() and "Fatal Python error:" in path.read_text()
                          for path in release_logs)
    if release_crashes:
        lines.append(f"- 收尾全量复检另有 {release_crashes} 次 native 崩溃，未取得新的完整通过结果；不能把此前通过记录解释为运行稳定性问题已解决。")
    lines += ["", "## 可复现性与限制", "",
              "- `heldout/*/episodes.jsonl` 保存每回合结果；`config.json` 含 checkpoint SHA256，重跑配置不一致会拒绝混写。",
              "- `selection/*.json` 保存只基于验证集的选择记录；`training/*/seed_*/command.txt`、配置及日志记录训练命令。",
              "- 三训练种子与有限测试样本尚不足以证明临床适用性；当前只是在该模拟器中的结果。",
              "- 血流感知控制利用模拟器的近似流速模型和已有路由观测，属于模型辅助方法；不能宣称端到端从零学习或现实医疗安全。",
              "- 报告使用本地代码和实测数据；本次联网文献检索未取得可核验内容，未将文献当实验证据。",
              "- 演示视频按成功/失败结果选例，仅用于说明行为，不参与成功率估计。", ""]
    (reports / "report.md").write_text("\n".join(lines))

    source_paths = [Path(name) for name in (
        "marl/geometric_control.py", "marl/mappo_advanced.py", "marl/policy_loader.py",
        "scripts/train_vector_mappo.py", "scripts/eval_success_study.py", "scripts/eval_checkpoint.py",
        "scripts/run_success_training.sh", "scripts/run_success_evaluation.sh", "scripts/report_success_study.py",
        "scripts/render_success_study.py", "make_gif.py",
        "SUCCESS_STUDY.md",
        "tests/test_geometric_control.py", "environments/vascular_3d_marl_env.py",
        "environments/vector_env.py", "environments/balanced_vector_env.py", "environments/vessel_geometry.py",
    )]
    hashes = {}
    for source in source_paths:
        target = reports / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (reports / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps({"selected": selected_label, "success": aggregate["success"],
                      "paired_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
