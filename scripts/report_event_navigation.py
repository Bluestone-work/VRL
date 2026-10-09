"""Require complete paired development runs before reporting EXP0062 outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_navigation_interface import paired_audit
from scripts.iterate_hierarchical_navigation import summarize


def acceptance(metrics):
    return bool(metrics['cluster_safe_success']['delta'] > 0.
                and metrics['relaxed_safe_success']['delta'] >= 0.
                and metrics['wall_contact_s']['delta'] <= 0.
                and metrics['obstacle_events_static']['delta'] <= 0.
                and metrics['obstacle_events_dynamic']['delta'] <= 0.
                and metrics['task_success']['delta'] >= -.05)


def report(directory):
    directory = Path(directory)
    full, hybrid = directory/'full_v5.jsonl', directory/'full_v6.jsonl'
    groups = [(full, method) for method in ['switch', 'guard', 'guard_replan', 'semantic', 'semantic_replan']]
    groups += [(hybrid, method) for method in ['hybrid', 'hybrid_replan']]
    summary = {}
    comparisons = {}
    for path, method in groups:
        statistics = summarize(path)[method]
        if statistics['episodes'] != 84 or statistics['errors']:
            raise ValueError(f'Incomplete development run: {method} {statistics}')
        summary[method] = statistics
        if method != 'switch':
            comparison = paired_audit(full, path, 'switch', method)
            if comparison['paired'] != 84 or comparison['reference_only'] or comparison['candidate_only']:
                raise ValueError(f'Incomplete pairing: {method}')
            comparison['passes_descriptive_gate'] = acceptance(comparison['metrics'])
            comparisons[method] = comparison
    recovery_path = directory/'RUNTIME_RECOVERY.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    return dict(summary=summary, comparisons=comparisons, runtime_recovery=recovery,
                interpretation='Repeatedly used development scenes; no sealed or unseen-anatomy claim. '
                               'Gates are descriptive, not clinical safety certification. '
                               'Bootstrap resamples paired scenes, not independent anatomies.')


def markdown(result):
    lines = ['# EXP0062 完整开发配对结果', '',
             'N=1，14个已使用开发解剖×6场景，300 s，图像定位＋同一含噪检测，topo v3。',
             '全部方法相同 scene hash 与 initial-state hash；无密封测试访问。', '',
             '| 方法 | 严格 Safe | RSafe | 任务 | 平均壁接触 s | 平均最长连续接触 s | 静态事件/局 | 动态事件/局 | 重规划/局 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, row in result['summary'].items():
        counts = [f"{int(round(row[key]*84))}/84" for key in ['cluster_safe_success', 'relaxed_safe_success', 'task_success']]
        values = [f"{row[key]:.3f}" for key in ['wall_contact_s', 'max_continuous_wall_contact_s',
                                                'obstacle_events_static', 'obstacle_events_dynamic', 'replan_count']]
        lines.append('| '+' | '.join([method]+counts+values)+' |')
    lines += ['', '## 完整报告，不选择性报告', '',
              '严格 Safe 维持旧定义：清除、壁接触<1 s、零所有障碍事件、无流失/集群接触/间距违规。',
              'RSafe 保持之前登记定义：清除、壁接触<5 s、零静态事件、无流失；另保留集群安全条件。',
              'RSafe 允许动态事件只是一项放宽代理，不证明动态接触安全。', '',
              '| 候选 vs switch | 严格 Safe 差值 pp | 配对场景 bootstrap 95% pp | 描述性接纳门槛 |',
              '|---|---:|---:|---|']
    for method, comparison in result['comparisons'].items():
        metric = comparison['metrics']['cluster_safe_success']
        lower, upper = metric['paired_bootstrap_95']
        gate = '通过，仅列后续学习候选' if comparison['passes_descriptive_gate'] else '未通过，保留为权衡/负结果'
        lines.append(f"| {method} | {100*metric['delta']:+.2f} | [{100*lower:+.2f}, {100*upper:+.2f}] | {gate} |")
    lines += ['', '这些是经典控制/执行器改善或回归，不能称为 RL / DAgger 已经学会。',
              '开发反馈已经用于设计 v6；区间不证明泛化到未见解剖。',
              '未自动替换现有训练、检查点或部署默认。早停 AUC 在输出中补齐到300 s。', '',
              '## 复现与审计', '',
              '- `full_v5.jsonl`、`full_v6.jsonl`：逐场完整结果。',
              '- 对应 `.manifest.json`：冻结配置及源文件 SHA256。',
              '- `research/runs/EXP0062_EVENT_LOCAL_20261007/v1` 至 `v6`：原代码快照。',
              '- `oracle_interface_audit.json`：Oracle 原始成本与标签审计。',
              '- `RUNTIME_RECOVERY.json`：一例原进程 native futex 挂起；同环境/配置/种子重跑，任务失败与54次静态事件如实纳入。',
              '- 后续评估入口增加120 s无结果 watchdog，记录待重跑场景，不把运行时挂起计为模型成败。',
              '- `research/experiments/EXP0062_INTERFACE_FINDINGS_20261007.md`：接口和物理缺口。',
              '- 运行 `python -m scripts.report_event_navigation --directory research/validation/EXP0062_EVENT_LOCAL_20261007` 重建本报告。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    arguments = parser.parse_args()
    result = report(arguments.directory)
    (arguments.directory/'FINAL_REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    (arguments.directory/'FINAL_REPORT.md').write_text(markdown(result))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
