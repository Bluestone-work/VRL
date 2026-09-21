"""Derive EXP_0001 reports and figures from preserved raw results only."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/runs/EXP_0001"
FIGURES = ROOT / "research/figures"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "seeds": len(values)}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def final_metrics(summary):
    evaluation = summary["final_evaluation"]
    records = evaluation["episodes"]
    assert len(records) == 280
    assert len({(r["scenario"], r["episode_seed"]) for r in records}) == 280
    result = dict(evaluation["macro"])
    for key in ("success", "removal_rate"):
        assert abs(np.mean([r[key] for r in records]) - result[key]) < 1e-7
    result["steps"] = float(np.mean([r["steps"] for r in records]))
    result["wall_contact_rate"] = result["wall_hits_per_step"] / 5
    return result


def save_figure(figure, name):
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=160)
    plt.close(figure)


def figures(seeds, training, legacy, metrics, diagnostic):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for seed in seeds:
        run = RUN / f"training/seed_{seed}"
        updates = rows(run / "update_metrics.jsonl")
        validation = rows(run / "eval_metrics.jsonl")
        for ax, key, title in zip(axes.flat[:2], ("success", "removal_rate"), ("Validation success", "Validation mass removal")):
            ax.plot([r["transitions"] for r in validation], [r["macro"][key] for r in validation], label=str(seed))
            ax.set_title(title)
        for ax, key in zip(axes.flat[2:5], ("actor_loss", "critic_loss", "explained_variance")):
            ax.plot([r["transitions"] for r in updates], [r[key] for r in updates], label=str(seed))
            ax.set_title(key)
        episodes = rows(run / "episode_metrics.jsonl")
        buckets = {}
        for record in episodes:
            buckets.setdefault(record["transitions"], []).append(record["return"])
        axes.flat[5].plot(list(buckets), [np.mean(v) for v in buckets.values()], label=str(seed))
        axes.flat[5].set_title("Training scalar return (completed episodes)")
    for ax in axes.flat:
        ax.set_xlabel("Real environment transitions")
        ax.grid(alpha=.25)
        ax.legend(title="Training seed", fontsize=8)
    fig.suptitle("EXP_0001 — fixed baseline reproduction; validation is not a test set")
    save_figure(fig, "EXP_0001_training_curves.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key, factor, title in zip(axes, ("success", "removal_rate", "steps", "wall_contact_rate"),
                                    (100, 100, 1, 100), ("Success (%)", "Removed mass (%)", "Episode length", "Wall contact / robot-step (%)")):
        groups = [[legacy[s][key] for s in seeds], [metrics[s][key] for s in seeds],
                  [diagnostic[s][key] for s in seeds]]
        for x, group in enumerate(groups):
            ax.bar(x, np.mean(group) * factor, yerr=np.std(group, ddof=1) * factor, capsize=4,
                   color=("#8795a1", "#217e84", "#ca8a32")[x])
            ax.scatter(np.full(len(group), x), np.array(group) * factor, color="black", s=13)
        ax.set_xticks(range(3), ["Archived\nvalidation", "Reproduced\nvalidation", "New development\nvalidation"], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("EXP_0001 — points are 3 training seeds; bars mean ± sample SD; different validation sets labeled")
    save_figure(fig, "EXP_0001_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(axes, ("steps", "collision_rate", "wall_contact_rate"),
                              ("Episode length", "Robot-pair collision / pair-step", "Wall contact / robot-step")):
        values = [[r[key] for r in rows(RUN / f"evaluation/seed_{s}/episodes.jsonl")] for s in seeds]
        ax.boxplot(values, tick_labels=list(map(str, seeds)), showfliers=True, showmeans=True,
                   flierprops={"markersize": 3, "alpha": .5})
        ax.set_title(title)
        ax.set_xlabel("Training seed (280 development episodes each)")
    fig.suptitle("EXP_0001 — development episode distributions; IQR boxes, 1.5 IQR whiskers, outliers shown")
    save_figure(fig, "EXP_0001_episode_safety.png")

    # Predetermined first episodes, not hand-picked successes or trajectories.
    selected = [r for r in rows(RUN / "evaluation/seed_42/episodes.jsonl")
                if r["episode_seed"] in (900000, 1020000)]
    case_records = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    trajectory_figure = plt.figure(figsize=(13, 6))
    for index, record in enumerate(selected):
        trace = np.load(RUN / "evaluation/seed_42/traces" / record["trace"])
        masses = trace["clot_masses"]
        initial_mass = float(masses[0].sum())
        recent_index = max(0, len(masses) - 51)
        case_records.append({"training_seed": 42, "scenario": record["scenario"],
            "episode_seed": record["episode_seed"], "success": record["success"], "steps": record["steps"],
            "first_contact_steps": record["first_contact_steps"], "removal_rate": record["removal_rate"],
            "remaining_mass_by_clot": masses[-1].tolist(),
            "last_50_steps_removal_ratio": float((masses[recent_index].sum() - masses[-1].sum()) / initial_mass),
            "wall_contact_rate": record["wall_contact_rate"],
            "mean_simultaneously_active_clots": float((trace["active_per_clot"] > 0).sum(axis=-1).mean()),
            "mean_active_robots_per_clot": trace["active_per_clot"].mean(axis=0).tolist(),
            "trace_file": str((RUN / "evaluation/seed_42/traces" / record["trace"]).relative_to(ROOT))})
        for clot in range(masses.shape[1]):
            axes[index, 0].plot(masses[:, clot], alpha=.8, label=f"Clot {clot}")
        axes[index, 0].plot(masses.sum(axis=1), color="black", linewidth=2, label="Total")
        axes[index, 0].set_title(f"{record['scenario']} — success={int(record['success'])}")
        axes[index, 0].set_ylabel("Clot mass (simulation units)")
        axes[index, 0].legend()
        flow = np.linalg.norm(trace["flow"], axis=-1)
        axes[index, 1].plot(flow.mean(axis=1), label="Mean local speed at robots")
        axes[index, 1].plot(flow.max(axis=1), alpha=.7, label="Max local speed at robots")
        axes[index, 1].set_title(f"seed 42 / episode {record['episode_seed']}")
        axes[index, 1].legend()
        ax = trajectory_figure.add_subplot(1, 2, index + 1, projection="3d")
        points = trace["tree_points"]
        for branch in np.unique(trace["tree_branch_ids"]):
            p = points[trace["tree_branch_ids"] == branch]
            ax.plot(*p.T, color="gray", alpha=.4, linewidth=2)
        for robot in range(trace["positions"].shape[1]):
            ax.plot(*trace["positions"][:, robot].T, label=f"R{robot}")
        ax.scatter(*trace["clot_positions"].T, color="red", marker="x", s=45, label="Clots")
        ax.set_title(record["scenario"])
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel("Environment step")
        ax.grid(alpha=.2)
    fig.suptitle("EXP_0001 — fixed episode IDs; observed mass and flow, no world-model predictions")
    trajectory_figure.suptitle("EXP_0001 — recorded robot trajectories in the same fixed episodes")
    save_figure(fig, "EXP_0001_clot_mass_and_flow.png")
    save_figure(trajectory_figure, "EXP_0001_robot_trajectories.png")
    return case_records


def fmt(value, percent=False):
    factor = 100 if percent else 1
    return f"{value['mean'] * factor:.4f} ± {value['std'] * factor:.4f}" + ("%" if percent else "")


def main():
    config = read(ROOT / "configs/experiments/EXP_0001.json")
    assert read(RUN / "training_gate.json")["passed"]
    assert read(RUN / "evaluation_gate.json")["passed"]
    integrity = read(RUN / "checks/final_integrity.json")
    assert integrity["source_integrity_passed"]
    seeds = config["seeds"]
    training = {s: read(RUN / f"training/seed_{s}/summary.json") for s in seeds}
    reference = {read(p)["seed"]: read(p) for p in (RUN / "provenance/reference").glob("*/summary.json")}
    legacy = {s: final_metrics(reference[s]) for s in seeds}
    metrics = {s: final_metrics(training[s]) for s in seeds}
    diagnostic = {s: read(RUN / f"evaluation/seed_{s}/summary.json")["macro"] for s in seeds}
    diagnostic_records = {s: rows(RUN / f"evaluation/seed_{s}/episodes.jsonl") for s in seeds}
    for s, records in diagnostic_records.items():
        assert len(records) == 280
        assert len({(r["scenario"], r["episode_seed"]) for r in records}) == 280
        assert len({r["scenario"] for r in records}) == 14
        for scenario in {r["scenario"] for r in records}:
            assert sum(r["scenario"] == scenario for r in records) == 20
        for key in ("success", "removal_rate", "steps", "wall_contact_rate", "collision_rate"):
            assert abs(np.mean([r[key] for r in records]) - diagnostic[s][key]) < 1e-7
        before_keys = {(r["scenario"], r["episode_seed"]) for r in reference[s]["final_evaluation"]["episodes"]}
        after_keys = {(r["scenario"], r["episode_seed"]) for r in training[s]["final_evaluation"]["episodes"]}
        assert before_keys == after_keys
    assert read(RUN / "checks/configuration_parity.json")["all_match"]
    common = ("success", "removal_rate", "steps", "wall_contact_rate")
    groups = {"archived_validation": legacy, "reproduced_validation": metrics, "development_validation": diagnostic}
    aggregate = {name: {key: stats([values[s][key] for s in seeds]) for key in common}
                 for name, values in groups.items()}
    for key in diagnostic[seeds[0]]:
        if all(diagnostic[s][key] is not None for s in seeds):
            aggregate["development_validation"][key] = stats([diagnostic[s][key] for s in seeds])
    difference = aggregate["reproduced_validation"]["success"]["mean"] - aggregate["archived_validation"]["success"]["mean"]
    removal_difference = aggregate["reproduced_validation"]["removal_rate"]["mean"] - aggregate["archived_validation"]["removal_rate"]["mean"]
    gate_passed = abs(difference * 100) <= config["gate"]["maximum_success_difference_pp"]
    commit = read(RUN / "provenance/snapshot.json")["git_commit"]
    details = []
    for s in seeds:
        validation = rows(RUN / f"training/seed_{s}/eval_metrics.jsonl")
        x = np.array([r["transitions"] for r in validation])
        y = np.array([r["macro"]["success"] for r in validation])
        details.append({"seed": s, "reference": legacy[s], "reproduced": metrics[s], "development": diagnostic[s],
                        "health": read(RUN / f"training/seed_{s}/health_check.json"),
                        "training_seconds": training[s]["training_seconds"],
                        "archived_training_seconds": reference[s]["training_seconds"],
                        "training_cost_ratio": training[s]["training_seconds"] / reference[s]["training_seconds"],
                        "invocation_seconds": read(RUN / f"training/seed_{s}/attempt_1.json")["seconds"],
                        "validation_success_auc_per_transition": float(np.trapz(y, x) / x[-1]),
                        "gpu_memory": read(RUN / f"training/seed_{s}/attempt_1.health.summary.json")["gpu_memory"]})
    result = {"experiment_id": "EXP_0001", "generated_at": datetime.now(timezone.utc).isoformat(),
              "git_commit": commit, "seeds": seeds, "aggregate": aggregate, "per_seed": details,
              "success_difference_pp": difference * 100,
              "success_relative_difference_percent": difference / aggregate["archived_validation"]["success"]["mean"] * 100,
              "removal_difference_pp": removal_difference * 100,
              "gate": {"passed": gate_passed, "tolerance_pp": 5, "meaning": "engineering reproduction, not statistical equivalence"}}
    result["replay_checks"] = {"cpu": read(RUN / "checks/diagnostic_replay.json"),
                               "gpu": read(RUN / "checks/diagnostic_replay_gpu.json")}
    result["artifact_integrity"] = integrity
    reports = RUN / "analysis"
    reports.mkdir(exist_ok=True)
    (reports / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    territory_rows = []
    for scenario in sorted({r["scenario"] for r in diagnostic_records[seeds[0]]}):
        selected = [r for records in diagnostic_records.values() for r in records if r["scenario"] == scenario]
        territory_rows.append({"scenario": scenario, "episodes": len(selected),
            **{key: float(np.mean([r[key] for r in selected])) for key in
               ("success", "removal_rate", "steps", "wall_contact_rate", "collision_rate")}})
    with (reports / "development_by_territory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(territory_rows[0]))
        writer.writeheader()
        writer.writerows(territory_rows)
    with (reports / "per_seed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "split", *common])
        writer.writeheader()
        for name, values in groups.items():
            for s in seeds:
                writer.writerow({"seed": s, "split": name, **{k: values[s][k] for k in common}})
    case_records = figures(seeds, training, legacy, metrics, diagnostic)
    (reports / "fixed_episode_cases.json").write_text(json.dumps(case_records, indent=2, allow_nan=False))
    manifest = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                for p in sorted(RUN.rglob("*"))
                if p.is_file() and "analysis" not in p.relative_to(RUN).parts
                and p.suffix in (".json", ".jsonl", ".npz", ".pt", ".py", ".xml", ".log")]
    (reports / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    report = ["# EXP_0001 Baseline reproduction report", "", f"Frozen code: `{commit}`.", "",
              "结果由 scripts/research_report.py 从不可替换的原始记录计算；三训练种子，标准差ddof=1。",
              "主比较是相同历史validation协议，额外development validation不是封存test，也不代表未见拓扑。", "",
              "| 数据集/运行 | Success mean ± SD | Clot removal mean ± SD | Episode length mean ± SD | Wall contact/robot-step mean ± SD |",
              "|---|---:|---:|---:|---:|"]
    for name, value in aggregate.items():
        report.append(f"| {name} | {fmt(value['success'], True)} | {fmt(value['removal_rate'], True)} | {fmt(value['steps'])} | {fmt(value['wall_contact_rate'], True)} |")
    report += ["", "## Per-seed results", "", "| Seed | Archived success | Reproduced success | Development success | Reproduced removal | Train seconds |",
               "|---|---:|---:|---:|---:|---:|"]
    for row in details:
        report.append(f"| {row['seed']} | {row['reference']['success']:.4%} | {row['reproduced']['success']:.4%} | {row['development']['success']:.4%} | {row['reproduced']['removal_rate']:.4%} | {row['training_seconds']:.2f} |")
    diag = aggregate["development_validation"]
    old_cost = sum(row["archived_training_seconds"] for row in details)
    new_cost = sum(row["training_seconds"] for row in details)
    report += ["", "## Comparison with baseline", "",
               f"成功率绝对变化 {difference * 100:+.4f} pp，相对变化 {result['success_relative_difference_percent']:+.4f}%；质量清除率变化 {removal_difference * 100:+.4f} pp。",
               f"预登记±5pp工程复现容差：{'PASS' if gate_passed else 'FAIL'}。不以此宣称统计等效或显著提升。", "",
               "## Diagnostics and costs", "",
               f"Completion Rate与Success Rate定义相同。成功回合完成步数 {fmt(diag['completion_steps'])}；有接触回合首次接触步数 {fmt(diag['first_contact_steps'])}。",
               f"Robot-robot collision / pair-step：{fmt(diag['collision_rate'], True)}；每回合pair事件 {fmt(diag['robot_collisions_total'])}。",
               f"Evaluation reward：{fmt(diag['return'])}；平均模拟flow speed：{fmt(diag['mean_flow_speed'])}。",
               f"平均执行动作L2：{fmt(diag['mean_action_magnitude'])}；平方和仅为energy proxy。",
               f"完整载入策略接口CPU推理成本：{fmt(diag['inference_ms_per_step'])} ms/step，包含critic和控制转换，不是纯actor延迟。",
               f"三次训练耗时总和 {sum(d['training_seconds'] for d in details):.2f}s；实际作业elapsed另见attempt JSON。并行运行，不能把总和当wall-clock。",
               f"历史三seed training_seconds总和{old_cost:.2f}s，本轮{new_cost:.2f}s，差{new_cost-old_cost:+.2f}s（{(new_cost/old_cost-1)*100:+.2f}%）。此为描述性成本对照；并发调度、梯度观察同步和系统负载未被单独控制，不能归因为算法变化。",
               "GPU峰值allocated/reserved、validation曲线AUC及所有梯度norm见aggregate和health日志。观察梯度会产生额外同步成本，未作无观察器配对成本实验。",
               "Sample Efficiency：固定各1M真实转移，逐seed validation AUC为描述性值；没有新算法对照，不能验证H1。", "",
               "## Stability and limitations", "",
               "全测试171 passed、1 skipped；先完成16,384-transition smoke。正式每seed的有限loss/梯度/参数、非零actor更新、checkpoint载入及280回合指标复算均通过才生成此报告。",
               f"完整性检查：冻结{integrity['frozen_files_checked']}文件hash未变，原目录审计时{integrity['original_python_files_checked']}个Python文件未变，原分支仍为{integrity['original_branch']}。",
               "冻结期间未改核心算法。只报告已完成seed，脚本要求全部预登记seed存在；失败attempt不删除。",
               "Unexpected Finding：smoke checkpoint在同cuda:0设备重放14回合时success/steps/wall/removal全部精确一致；CPU重放success一致，但最大单回合清除率差2.9065pp。见FAILURE_LOG F003。主复现与CPU开发验证的设备差异必须保留，不能归因为算法。",
               "工程失败：F001为coordinator缺少PYTHONPATH，训练已成功，修正启动路径后检查通过；F002为分析脚本预编译语法错误，未产生错误结果。原始记录均保留。",
               "不同血管/机器人数量的最终泛化、unsafe flow阈值、shear、真实能耗与世界模型预测误差：N/A。未定义的物理量不补造。", "",
               "## Paired checkpoint comparison", "",
               "只读比较历史与本轮最终actor/critic张量；完整checkpoint还含不同运行状态，其文件hash可以不同。", "",
               *[f"- seed{row['seed']}：actor精确一致={row['modules']['actor']['exactly_equal']}，critic精确一致={row['modules']['critic']['exactly_equal']}；旧权重文件未变={row['archive_unchanged']}。" for row in integrity["paired_checkpoints"]], "",
               "## Failure cases and observed behaviors", "",
               "完整失败回合保存在evaluation/seed_*/episodes.jsonl；固定seed42的首个肺动脉与股腘动脉回合作为轨迹图，未按成功挑选。",
               "质量与血流曲线支持对具体回合的描述，但不单凭总成功率推断协作或reward hacking机制。复杂场景逐类结果见summary.json。", "",
               "## Scientific interpretation and innovation", "",
               "What improved / worsened：上表定量差异仅表示复现差异，未改变算法，不能归因为方法改进。",
               "What did not change：physics、observation、reward、actor/critic、控制器与正式训练预算。",
               "Why：原协议的相同指标还得到actor/critic张量相等的直接支持；CPU/GPU轨迹分歧的具体浮点传播机制仍为HYPOTHESIS — NOT VERIFIED。",
               "支持/削弱：此实验仅检验工程复现，不验证H1–H5。新的问题是严格评估划分、接触/安全诊断与旧world model语义。",
               "Engineering Improvement：可追溯基线与仪表。Algorithmic Improvement：无。Scientific Contribution：描述性复现证据。Potential Novel Contribution：无新增主张。",
               "NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION。Literature verification required.", "",
               "## Development validation by territory", "",
               "每个场景60回合，汇总三个训练seed；按成功率排序仅用于展示全部场景，不选择模型。", "",
               "| Territory | Success | Clot removal | Steps | Wall contact / robot-step | Pair collision / pair-step |",
               "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(territory_rows, key=lambda r: r["success"]):
        report.append(f"| {row['scenario']} | {row['success']:.2%} | {row['removal_rate']:.2%} | {row['steps']:.2f} | {row['wall_contact_rate']:.2%} | {row['collision_rate']:.2%} |")
    report += ["", "## Fixed episode observations", ""]
    for case in case_records:
        report.append(f"- seed42 / {case['scenario']} / episode {case['episode_seed']}：成功={int(case['success'])}，长度{case['steps']}步，首次接触{case['first_contact_steps']}步，清除率{case['removal_rate']:.2%}；最后至多50步新增清除占初始总质量{case['last_50_steps_removal_ratio']:.2%}；平均同时处理{case['mean_simultaneously_active_clots']:.3f}个血栓，wall contact/robot-step={case['wall_contact_rate']:.2%}。")
    report += ["", "这些是固定回合的观测，不构成对全部失败原因的因果证明；逐血栓残余与活动机器人见fixed_episode_cases.json。",
               "流速曲线采样在各机器人位置，变化同时受机器人移动及局部阻塞影响，不能单凭这条曲线宣称全血管流量恢复。",
               "股腘动脉低成功率及固定回合消融停滞登记为FAILURE_LOG F004；后续需分别验证局部可控性、分配、接触和投影机制，不先调参掩盖。"]
    report += ["", "## Figures and provenance", ""]
    for filename in ("training_curves", "comparison", "episode_safety", "clot_mass_and_flow", "robot_trajectories"):
        report.append(f"![EXP_0001 {filename}](figures/EXP_0001_{filename}.png)")
        report.append("")
    report += ["机器、pip freeze、代码manifest、原参考文件副本：`research/runs/EXP_0001/provenance/`。",
               "聚合及所有原始产物SHA256：`research/runs/EXP_0001/analysis/`。",
               "", "下一实验：EXP_0002评估协议验证；本轮不启动世界模型创新训练。", ""]
    body = "\n".join(report)
    (ROOT / "research/BASELINE_REPORT.md").write_text(body)
    (ROOT / "research/RESULTS_SUMMARY.md").write_text(
        "# Results summary\n\n" + "\n".join(report[2:13]) +
        f"\n\nEXP_0001工程复现gate：{'PASS' if gate_passed else 'FAIL'}。详见[BASELINE_REPORT](BASELINE_REPORT.md)。\n"
        "\n世界模型、planning、unseen-topology测试均未开展，H1–H5仍为HYPOTHESIS — NOT VERIFIED。\n")
    exp = ROOT / "research/experiments/EXP_0001.md"
    text = exp.read_text().split("## Actual Result")[0]
    exp.write_text(text + "## Actual Result\n\n" + body.replace("(figures/", "(../figures/") +
                   "\n## Conclusion\n\n" + ("工程基线复现通过。" if gate_passed else "复现未通过预登记容差；保留负结果，需诊断。") + "\n")
    review = f"""# Phase 0 Research Gate Review

状态：{'PASS — baseline reproduction only' if gate_passed else 'FAIL — retain negative result'}。

- 阶段目标：PROJECT_AUDIT、冻结版本、预登记、完整测试、smoke、3seed各1M、独立开发验证与归档已完成。
- 稳定性：有限loss/gradient/参数与checkpoint检查通过；跨机器可靠性尚未验证。
- 可复现性：代码`{commit}`、完整config、pip freeze、机器信息、原始JSONL/trace/checkpoint/hash均保留；主成功率差{difference * 100:+.4f}pp。
- 超过variance的提升：未提出性能提升假设，没有算法变量；不宣称显著优越或统计等效。
- Confounders：历史final validation复用调参回合；诊断集只是development validation；gradient observer有成本；CPU/GPU回放敏感性见F003，固定GPU回放14回合已精确一致；native稳定性仅限本轮观察。
- 下一阶段：{'允许继续EXP_0002评估协议核验，再进入完整仪表；不直接跳到WM planning。' if gate_passed else '先诊断复现差异，不进入主要创新。'}
- 当前方向：保留coupled dynamics研究假设；H1–H5尚无证据，NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION。
- 回退：无需删除或重写失败；所有数据留存。若后续改协议/物理/算法，必须新experiment ID。
"""
    (ROOT / "research/reviews/PHASE_0_REVIEW.md").write_text(review)
    registry = ROOT / "research/EXPERIMENT_REGISTRY.csv"
    with registry.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        registry_rows = list(reader)
    for row in registry_rows:
        if row["experiment_id"] == "EXP_0001":
            row.update(success_rate=aggregate["reproduced_validation"]["success"]["mean"],
                       clot_removal=aggregate["reproduced_validation"]["removal_rate"]["mean"],
                       episode_length=aggregate["reproduced_validation"]["steps"]["mean"],
                       collision_rate=diag["collision_rate"]["mean"],
                       sample_efficiency="fixed_1M_per_seed;validation_AUC_in_aggregate",
                       status="complete_reproduced" if gate_passed else "complete_negative",
                       main_result=f"success_delta_pp={difference * 100:+.4f};collision_on_development_validation;no_algorithm_change")
    with registry.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(registry_rows)
    print(json.dumps({"aggregate": aggregate, "gate": result["gate"], "success_delta_pp": difference * 100}, indent=2))


if __name__ == "__main__":
    main()
