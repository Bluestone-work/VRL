# Research workspace

当前已完成 `EXP_0001`基线复现与`EXP_0002`评估协议核验，下一项为EXP_0003完整科研指标；不提前引入新的世界模型。

已完成：3seed×1M transitions、原协议840回合与额外开发验证840回合、5张结果图。工程复现gate通过；完整报告见下方入口。世界模型假设尚未测试。

EXP_0002另完成840对设备回合、84次同设备重复、42次旧接口控制和56次smoke回放；3张新图、560/140/280未来split清单。测试清单只登记，未运行policy。

## 记录入口

- [PROJECT_AUDIT](PROJECT_AUDIT.md)：实现、风险与研究空间。
- [ROADMAP](ROADMAP.md)：Phase 0–8顺序及gate。
- [EXP_0001](experiments/EXP_0001.md)：预登记、结果和下一问题。
- [BASELINE_REPORT](BASELINE_REPORT.md)、[RESULTS_SUMMARY](RESULTS_SUMMARY.md)：由原始结果计算。
- [REGISTRY](EXPERIMENT_REGISTRY.csv)、[RESEARCH_LOG](RESEARCH_LOG.md)、[FAILURE_LOG](FAILURE_LOG.md)、[DECISION_LOG](DECISION_LOG.md)：持续记录。
- [INNOVATION_TRACKER](INNOVATION_TRACKER.md)、[LITERATURE_GAPS](LITERATURE_GAPS.md)：证据与文献边界。
- [PHASE_0_REVIEW](reviews/PHASE_0_REVIEW.md)：阶段结论。
- [EXP_0002](experiments/EXP_0002.md)、[EVALUATION_PROTOCOL_REPORT](EVALUATION_PROTOCOL_REPORT.md)：设备配对与划分审计。
- [EVALUATION_PROTOCOL](EVALUATION_PROTOCOL.md)：后续实验需要显式采用的设备、指标和清单约定。

## 本次执行来源

冻结提交：`8c159ab153e2f651a5cbdfa7c22706c86dc203d8`。
冻结worktree：`/home/wj/桌面/vascular_marl_research_EXP_0001`。
配置：`configs/experiments/EXP_0001.json`。
数据：`research/runs/EXP_0001/`，不覆盖旧`experiments/ladder_stage1`。

Python为`/home/wj/miniconda3/envs/v/bin/python`。coordinator与子进程均需明确PYTHONPATH；否则包含自定义环境类的checkpoint可能无法反序列化，见FAILURE_LOG F001。

从项目根运行的正式命令为：

```bash
env PYTHONPATH=/home/wj/桌面/vascular_marl_research_EXP_0001 \
  taskset -c 0-5,8-23 /home/wj/miniconda3/envs/v/bin/python \
  /home/wj/桌面/vascular_marl_research_EXP_0001/scripts/research_baseline.py \
  train --project-root /home/wj/桌面/vascular_marl_local.tar.
```

其他stage为`prepare`、`checks`、`smoke`、`evaluate`，按序运行。`prepare`只执行一次，运行器拒绝覆盖已存在attempt。已完成的实验不要原地重复训练；新复现应注册新ID、配置独立输出路径和源码快照。

分析入口：`python scripts/research_report.py`，只在三个训练及开发验证全部完成后生成报告。当前报告器属于独立分析代码，不改冻结的训练/评估源码；其版本与最终记录单独归档。

已执行的只读完整性/权重比较见`checks/final_integrity.json`，对应工具为`scripts/research_verify_artifacts.py`。分析图表可以由原始记录重新生成；训练和验证attempt不覆盖。结果归档入口为`scripts/research_archive_results.py`，结果commit见`runs/EXP_0001/results_archive.json`。冻结执行commit负责定位运行源码；结果commit负责定位报告及精简数据。约211MB本地运行产物中的大checkpoint/trace/log未全部写入Git，只保存本地文件及hash清单。

## EXP_0002 execution and results

执行commit：`c7830951f2674aa7899b25da2ff0d49d8e81d8ef`；冻结worktree：`/home/wj/桌面/vascular_marl_research_EXP_0002`；配置`configs/experiments/EXP_0002.json`。数据在`research/runs/EXP_0002/`，约93MB；NPZ保留本地，compact JSON/JSONL、文档和图单独Git归档。

运行器`scripts/research_protocol_run.py`依次为prepare/checks/smoke/formal，已完成stage拒绝覆盖；verify只校验冻结源码、父输入、原branch/index。使用`env PYTHONPATH=/home/wj/桌面/vascular_marl_research_EXP_0002`和上述固定Python从冻结代码调用。分析入口`scripts/research_protocol_report.py`可由原始数据重建；归档入口`scripts/research_protocol_archive.py`，版本回执`runs/EXP_0002/results_archive.json`。

未来清单在`runs/EXP_0002/splits/`，test没有性能数据。旧trainer尚未接入manifest消费；不要把清单存在误解成旧训练已经严格隔离。

## 解释边界

原协议100000起点是validation。900000起点是未参与本轮训练选择的development validation，仍不是封存unseen-topology test。图表中的能量是动作代理量，flow为模拟速度，缺失shear/治疗强度/物理单位时标N/A。

H1–H5均为`HYPOTHESIS — NOT VERIFIED`。创新判断为`NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION`。
