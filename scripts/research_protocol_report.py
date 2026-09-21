"""Derive EXP_0002 paired evidence; never modify evaluation records."""
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
RUN = ROOT / "research/runs/EXP_0002"
KEYS = ("success", "removal_rate", "steps", "wall_contact_rate", "collision_rate", "return", "inference_ms_per_step")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "n_training_seeds": len(values)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_difference(a, b):
    count = min(len(a), len(b))
    changed = np.any(a[:count] != b[:count], axis=tuple(range(1, a.ndim)))
    indices = np.flatnonzero(changed)
    return int(indices[0]) if len(indices) else None


def trajectories(a_path, b_path):
    with np.load(a_path) as a, np.load(b_path) as b:
        count = min(len(a["positions"]), len(b["positions"]))
        difference = np.linalg.norm(a["positions"][:count] - b["positions"][:count], axis=-1)
        actions = min(len(a["actions"]), len(b["actions"]))
        return {"first_action_divergence_zero_based": first_difference(a["actions"], b["actions"]),
                "first_position_divergence_transition": first_difference(a["positions"], b["positions"]),
                "initial_action_max_absolute_difference": float(np.max(np.abs(a["actions"][0] - b["actions"][0]))),
                "max_action_absolute_difference_common_prefix": float(np.max(np.abs(a["actions"][:actions] - b["actions"][:actions]))),
                "max_robot_position_l2_common_prefix": float(difference.max()),
                "mean_robot_position_l2_common_prefix": float(difference.mean()),
                "max_clot_mass_absolute_difference_common_prefix": float(np.max(np.abs(a["clot_masses"][:count] - b["clot_masses"][:count]))),
                "common_position_samples": count, "unequal_episode_lengths": len(a["positions"]) != len(b["positions"])}


def figure_save(fig, name):
    fig.tight_layout()
    path = ROOT / "research/figures" / name
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figures(groups, pairs, seeds):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for ax, key, factor, title in zip(axes,
            ("success", "removal_rate", "steps", "wall_contact_rate", "collision_rate"),
            (100, 100, 1, 100, 100),
            ("Success (%)", "Mass removal (%)", "Episode length", "Wall / robot-step (%)", "Collision / pair-step (%)")):
        for seed in seeds:
            ax.plot((0, 1), [groups[d][seed][key] * factor for d in ("cpu", "cuda:0")], marker="o", label=str(seed))
        ax.set_xticks((0, 1), ("CPU", "GPU 0"))
        ax.set_xlim(-.3, 1.3)
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(title="Training seed")
    fig.suptitle("EXP_0002 — same weights and development episodes; device is the only performance variable")
    figure_save(fig, "EXP_0002_paired_device_metrics.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    first = [p["first_position_divergence_transition"] for p in pairs if p["first_position_divergence_transition"] is not None]
    axes[0].hist(first, bins=np.arange(0, 311, 10), color="#217e84")
    axes[0].set(xlabel="First differing position transition", ylabel="Episode pairs", title=f"Divergent positions: {len(first)}/{len(pairs)}")
    axes[1].hist([p["max_robot_position_l2_common_prefix"] for p in pairs], bins=30, color="#ca8a32")
    axes[1].set(xlabel="Max robot position L2 (simulation units)", ylabel="Episode pairs", title="Compare shared prefix; no padding")
    cats = ("Both success", "Both failure", "CPU only", "GPU only")
    values = [sum(p["cpu_success"] == a and p["gpu_success"] == b for p in pairs) for a, b in ((1, 1), (0, 0), (1, 0), (0, 1))]
    axes[2].bar(cats, values, color=("#217e84", "#8795a1", "#ca8a32", "#7c64a4"))
    axes[2].tick_params(axis="x", labelrotation=15)
    axes[2].set(ylabel="Episode pairs", title="Outcome agreement; all 840 pairs")
    fig.suptitle("EXP_0002 — cross-device sensitivity is not an algorithm improvement")
    figure_save(fig, "EXP_0002_trajectory_divergence.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for index, (scenario, seed) in enumerate((("pulmonary_saddle", 900000), ("femoropopliteal_pad", 1020000))):
        traces = {d: np.load(RUN / f"formal/{d}/seed_42/traces/{scenario}_{seed}.npz") for d in ("cpu", "cuda_0")}
        try:
            for device, trace in traces.items():
                axes[index, 0].plot(trace["clot_masses"].sum(axis=1), label=device)
            count = min(len(t["positions"]) for t in traces.values())
            diff = np.linalg.norm(traces["cpu"]["positions"][:count] - traces["cuda_0"]["positions"][:count], axis=-1)
            for robot in range(diff.shape[1]):
                axes[index, 1].plot(diff[:, robot], label=f"R{robot}")
            axes[index, 0].set_title(f"{scenario} — total clot mass")
            axes[index, 1].set_title(f"seed 42 / episode {seed} — position difference")
            axes[index, 0].set_ylabel("Clot mass (simulation units)")
            axes[index, 1].set_ylabel("Robot position L2 (simulation units)")
        finally:
            for trace in traces.values():
                trace.close()
    for ax in axes.flat:
        ax.set_xlabel("Environment transition")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("EXP_0002 — the same fixed episode IDs as EXP_0001, no selection by outcome")
    figure_save(fig, "EXP_0002_fixed_episode_comparison.png")


def fmt(value, percent=False):
    factor = 100 if percent else 1
    return f"{value['mean'] * factor:.4f} ± {value['std'] * factor:.4f}" + ("%" if percent else "")


def main():
    study = read(ROOT / "configs/experiments/EXP_0002.json")
    assert read(RUN / "formal_gate.json")["passed"]
    seeds = study["seeds"]
    groups = {device: {} for device in study["devices"]}
    maps, costs, repeat_count = {}, [], 0
    for device in study["devices"]:
        for seed in seeds:
            directory = RUN / f"formal/{device.replace(':', '_')}/seed_{seed}"
            records = rows(directory / "episodes.jsonl")
            assert len(records) == 280
            assert len({(r["scenario"], r["episode_seed"]) for r in records}) == 280
            assert len({r["scenario"] for r in records}) == 14
            for scenario in {r["scenario"] for r in records}:
                assert sum(r["scenario"] == scenario for r in records) == 20
            summary = read(directory / "summary.json")
            groups[device][seed] = summary["macro"]
            for key in KEYS:
                assert abs(np.mean([r[key] for r in records]) - summary["macro"][key]) < 1e-7
            assert read(directory / "checks.json")["passed"]
            repeats = rows(directory / "repeats.jsonl")
            assert len(repeats) == 14
            repeat_count += len(repeats)
            maps[(device, seed)] = {(r["scenario"], r["episode_seed"]): r for r in records}
            costs.append({"device": device, "seed": seed, "seconds": read(directory / "attempt_1.json")["seconds"]})
    pairs = []
    for seed in seeds:
        assert maps[("cpu", seed)].keys() == maps[("cuda:0", seed)].keys()
        for key, cpu in maps[("cpu", seed)].items():
            gpu = maps[("cuda:0", seed)][key]
            assert cpu["initial_state_sha256"] == gpu["initial_state_sha256"]
            assert cpu["complete_geometry_sha256"] == gpu["complete_geometry_sha256"]
            paths = [RUN / f"formal/{device}/seed_{seed}/traces" / cpu["trace"] for device in ("cpu", "cuda_0")]
            pairs.append({"training_seed": seed, "scenario": key[0], "episode_seed": key[1],
                "cpu_success": cpu["success"], "gpu_success": gpu["success"],
                "success_flipped": cpu["success"] != gpu["success"],
                "trace_exact": cpu["trace_arrays_sha256"] == gpu["trace_arrays_sha256"],
                **{key + "_gpu_minus_cpu": gpu[key] - cpu[key] for key in KEYS}, **trajectories(*paths)})
    aggregate = {device: {key: stats([groups[device][s][key] for s in seeds]) for key in KEYS} for device in study["devices"]}
    deltas = {key: stats([groups["cuda:0"][s][key] - groups["cpu"][s][key] for s in seeds]) for key in KEYS}
    inventory = read(RUN / "splits/summary.json")
    assert inventory["passed"] and inventory["prospective_test_policy_evaluations"] == 0
    training = read(RUN / "splits/saved_training_geometries.json")
    coverage = {}
    for seed, children in training["coverage"].items():
        # _geometry_generation is initialized to 0; the first real tree is generation 1.
        # Preserve raw inventory's 0..max list, and derive actual tree coverage from observations.
        actual = []
        for child in children:
            observed = child["observed_generations"]
            missing = sorted(set(range(1, max(observed) + 1)) - set(observed))
            actual.append({"child": child["child"], "observed_generations": observed,
                           "missing_actual_generations": missing, "expected_actual_generations": max(observed)})
        coverage[seed] = actual
    commit = read(RUN / "provenance/snapshot.json")["git_commit"]
    result = {"experiment_id": "EXP_0002", "execution_commit": commit,
        "generated_at": datetime.now(timezone.utc).isoformat(), "analysis_script_sha256": sha(__file__),
        "aggregate": aggregate, "paired_delta_gpu_minus_cpu": deltas, "paired_episodes": len(pairs),
        "same_device_repeat_pairs": repeat_count, "exact_cross_device_traces": sum(p["trace_exact"] for p in pairs),
        "position_divergent_pairs": sum(p["first_position_divergence_transition"] is not None for p in pairs),
        "success_flips": sum(p["success_flipped"] for p in pairs),
        "cpu_only_success": sum(p["cpu_success"] == 1 and p["gpu_success"] == 0 for p in pairs),
        "gpu_only_success": sum(p["cpu_success"] == 0 and p["gpu_success"] == 1 for p in pairs),
        "maximum_initial_action_difference": max(p["initial_action_max_absolute_difference"] for p in pairs),
        "maximum_position_difference_common_prefix": max(p["max_robot_position_l2_common_prefix"] for p in pairs),
        "job_costs": costs, "geometry_inventory": inventory, "actual_training_generation_coverage": coverage,
        "gate": {"passed": True, "meaning": "protocol consistency and prospective split only; historical complete disjointness unproven"},
        "training_steps": 0, "world_model": "N/A", "unseen_topology": "NOT EVALUATED"}
    analysis = RUN / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (analysis / "success_flip_cases.json").write_text(json.dumps([p for p in pairs if p["success_flipped"]], indent=2, allow_nan=False))
    with (analysis / "paired_episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    with (analysis / "per_seed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["device", "training_seed", *KEYS])
        writer.writeheader()
        for device, values in groups.items():
            for seed, metrics in values.items():
                writer.writerow({"device": device, "training_seed": seed, **{k: metrics[k] for k in KEYS}})
    figures(groups, pairs, seeds)
    files = [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sorted(RUN.rglob("*"))
             if p.is_file() and "analysis" not in p.relative_to(RUN).parts and p.name != "results_archive.json"]
    (analysis / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    generate_reports(study, result, groups, pairs)
    print(json.dumps({k: result[k] for k in ("aggregate", "paired_delta_gpu_minus_cpu", "success_flips", "cpu_only_success", "gpu_only_success", "exact_cross_device_traces", "gate")}, indent=2))


def generate_reports(study, result, groups, pairs):
    a, d, inv = result["aggregate"], result["paired_delta_gpu_minus_cpu"], result["geometry_inventory"]
    table = ["| Device | Success | Mass removal | Episode length | Wall / robot-step | Collision / pair-step |",
             "|---|---:|---:|---:|---:|---:|"]
    for device in study["devices"]:
        values = a[device]
        table.append(f"| {device} | {fmt(values['success'], True)} | {fmt(values['removal_rate'], True)} | {fmt(values['steps'])} | {fmt(values['wall_contact_rate'], True)} | {fmt(values['collision_rate'], True)} |")
    success_pp = d["success"]["mean"] * 100
    relative = d["success"]["mean"] / a["cpu"]["success"]["mean"] * 100
    lines = ["# EXP_0002 — Evaluation protocol report", "", f"执行版本：`{result['execution_commit']}`。分析脚本hash：`{result['analysis_script_sha256']}`。", "",
        "本轮不训练，复用EXP_0001全部三个最终权重。唯一性能变量是CPU vs cuda:0；同一840回合配对，另做84次同设备重复。均值±样本标准差来自3个训练seed，不能把840回合解释为840独立模型。",
        "开发验证，不是封存test；GPU未来统一设备在看到结果前已固定。", "", *table, "",
        "## Per-seed paired results", "", "| Seed | CPU success | GPU success | GPU−CPU success (pp) | CPU removal | GPU removal |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for seed in study["seeds"]:
        cpu, gpu = groups["cpu"][seed], groups["cuda:0"][seed]
        lines.append(f"| {seed} | {cpu['success']:.4%} | {gpu['success']:.4%} | {(gpu['success']-cpu['success'])*100:+.4f} | {cpu['removal_rate']:.4%} | {gpu['removal_rate']:.4%} |")
    lines += ["", "## Actual result and comparison", "",
        f"成功率GPU−CPU={success_pp:+.4f}pp，相对{relative:+.4f}%；清除率差{d['removal_rate']['mean']*100:+.4f}pp；长度差{d['steps']['mean']:+.4f}步；wall差{d['wall_contact_rate']['mean']*100:+.4f}pp；pair collision差{d['collision_rate']['mean']*100:+.4f}pp。",
        "这些是测量设备效应，不称Algorithmic Improvement或临床差异。CPU与父实验同回合的非时间指标和全部trace数组精确一致，基线差0。",
        f"成功结局翻转{result['success_flips']}/840：仅CPU成功{result['cpu_only_success']}，仅GPU成功{result['gpu_only_success']}；跨设备全部trace数组精确一致{result['exact_cross_device_traces']}/840。",
        f"相同初始状态下首次执行动作的最大绝对差{result['maximum_initial_action_difference']:.9g}；共同时间前缀内最大机器人位置L2差{result['maximum_position_difference_common_prefix']:.9g}（模拟单位）。不补齐已结束轨迹，不把不同长度直接作整段相减。",
        "首次动作已出现差异能定位到policy/控制计算链的设备影响；尚未定位具体浮点算子或证明哪种设备更接近物理真值。放大机制仍为HYPOTHESIS — NOT VERIFIED。", "",
        "## Correctness, stability and costs", "",
        "完整回归177 passed、1 skipped。56回合smoke通过；正式CPU父实验840回合一致；84对同设备重复精确一致；42个历史GPU控制回合与旧接口的success/steps/wall一致且removal差≤1e-7。没有训练、梯度更新或checkpoint选择。",
        f"CPU完整策略接口推理：{fmt(a['cpu']['inference_ms_per_step'])}ms/step；GPU：{fmt(a['cuda:0']['inference_ms_per_step'])}ms/step。包含critic/动作转换与传输，在并发执行下测得，不是纯actor或受控硬件benchmark。",
        f"正式任务作业耗时总和CPU={sum(c['seconds'] for c in result['job_costs'] if c['device']=='cpu'):.2f}s、GPU={sum(c['seconds'] for c in result['job_costs'] if c['device']=='cuda:0'):.2f}s，包含载入/重复/trace记录，不等于并行总wall-clock。",
        "Training Stability、Training Cost、Sample Efficiency改善：N/A（Training Steps=0）。Completion Rate=Success Rate；任务质量/碰撞定义沿用EXP_0001；未引入新的unsafe流速阈值或物理单位。", "",
        "## Split audit", "",
        "内容hash包含点、半径、分支、连接、flow_fraction和派生几何数组；它检查精确有序内容身份，不是图同构或近似重复检测。初始状态另包含机器人/血栓/RNG身份。",
        f"未来train/validation/test分别{inv['counts']['train']}/{inv['counts']['validation']}/{inv['counts']['test']}个实例；各split内部及三者之间内容hash无重复。封存test策略执行数0。",
        "该清单是供后续训练消费的prospective split；尚未接入训练器，不可宣称旧策略没有训练泄漏。所有集合来自相同14类场景，不是unseen-topology test。",
        f"历史评估与选择验证的复用：{json.dumps(inv['historical_validation_reuse'], ensure_ascii=False)}。",
        f"已保存训练树共{inv['counts']['saved_historical_train']}条记录、{inv['unique_counts']['saved_historical_train']}个独立完整hash；checkpoint无法提供完整训练历史。",
        "实际generation从1开始。原清单的missing_generations_through_last_saved按0..max列举，包含不存在的generation0；本报告仅由observed_generations按1..max重新计算实际树覆盖。保留原始字段，不静默改写。",
        "该分母为创建过的树generation数量，可能包含初始化后重置、未用于采样的树，不能直接解释为全部实际训练转移覆盖。", ""]
    for seed, children in result["actual_training_generation_coverage"].items():
        observed = sum(len(c["observed_generations"]) for c in children)
        expected = sum(c["expected_actual_generations"] for c in children)
        lines.append(f"- seed{seed}：已保存的子环境generation {observed}/{expected}，未覆盖{expected-observed}；不足以认证历史训练与评估完整互斥。")
    lines += ["", "全部集合的交集计数如下；saved_historical_train仅代表有保存证据的子集。", "", "| Pair | Identical content hashes |", "|---|---:|"]
    lines.extend(f"| {key} | {count} |" for key, count in inv["overlap_counts"].items())
    lines += ["", "## All outcome flip cases", "",
              "全部10个翻转回合均列出；1表示全部血栓清零。", "",
              "| Training seed | Scenario | Episode seed | CPU success | GPU success | GPU−CPU removal (pp) |",
              "|---|---|---:|---:|---:|---:|"]
    for pair in pairs:
        if pair["success_flipped"]:
            lines.append(f"| {pair['training_seed']} | {pair['scenario']} | {pair['episode_seed']} | {int(pair['cpu_success'])} | {int(pair['gpu_success'])} | {pair['removal_rate_gpu_minus_cpu']*100:+.4f} |")
    lines += ["", "## Failure cases, observed behaviors and interpretation", "",
        "全部失败与翻转回合在paired_episodes.csv和各设备episodes.jsonl；没有按成功挑选或重跑。轨迹图沿用EXP_0001预固定的肺动脉与股腘动脉两个episode ID。",
        "What improved：测量来源、设备、配置、身份和未来划分现在可审计。Why：同设备和父接口控制提供直接证据。",
        "What worsened：新增检查和trace记录有执行成本；设备间性能指标的变化见上表，不能概括为统一改善。",
        "What did not change：权重、物理、奖励、观测、动作变换和开发验证回合。",
        "Hypothesis support：通过同设备/接口复现检查；跨设备是否精确一致由翻转与轨迹计数判断。H1–H5未测试。",
        "New question：如何定位最早浮点分歧及其接触/投影敏感性，怎样在未来训练中强制消费split清单？这些应独立实验，不在此轮改数值物理。",
        "Engineering Improvement：协议检查与数据身份。Algorithmic Improvement：无。Scientific Contribution：本机固定模拟器下的设备敏感性证据。Potential Novel Contribution：无。Needs Literature Verification：I1–I7。",
        "NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION。Literature verification required.", "",
        "## Figures and reproducibility", ""]
    for name in ("paired_device_metrics", "trajectory_divergence", "fixed_episode_comparison"):
        lines += [f"![EXP_0002 {name}](figures/EXP_0002_{name}.png)", ""]
    lines += ["原始数据、attempt、协议、snapshot、硬件依赖、父输入hash与split清单：`research/runs/EXP_0002/`。",
        "分析入口：`scripts/research_protocol_report.py`；结果版本独立归档。未来测试清单未获得性能分数。", "",
        "## Conclusion and next experiment", "",
        "EXP_0002协议gate通过；不认证历史完整训练划分或跨设备等价。允许EXP_0003完善科研指标及准备未来split消费接口，不直接进入世界模型规划。", ""]
    body = "\n".join(lines)
    (ROOT / "research/EVALUATION_PROTOCOL_REPORT.md").write_text(body)
    path = ROOT / "research/experiments/EXP_0002.md"
    prereg = path.read_text().split("## Actual Result")[0]
    path.write_text(prereg + "## Actual Result\n\n" + body.replace("(figures/", "(../figures/"))
    summary = ROOT / "research/RESULTS_SUMMARY.md"
    previous = summary.read_text().split("## EXP_0002")[0].rstrip()
    summary.write_text(previous + "\n\n## EXP_0002\n\n" + "\n".join(table) +
        f"\n\n设备配对success差{success_pp:+.4f}pp，翻转{result['success_flips']}/840。同设备重复与CPU父接口检查通过；没有算法变化。详见[EVALUATION_PROTOCOL_REPORT](EVALUATION_PROTOCOL_REPORT.md)。\n")
    registry = ROOT / "research/EXPERIMENT_REGISTRY.csv"
    with registry.open() as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    for row in records:
        if row["experiment_id"] == "EXP_0002":
            row.update(status="complete_protocol_verified", success_rate=a["cuda:0"]["success"]["mean"],
                clot_removal=a["cuda:0"]["removal_rate"]["mean"], episode_length=a["cuda:0"]["steps"]["mean"],
                collision_rate=a["cuda:0"]["collision_rate"]["mean"],
                main_result=f"canonical_gpu_dev;success_delta_pp={success_pp:+.4f};flips={result['success_flips']}/840;not_algorithm_gain")
    with registry.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
