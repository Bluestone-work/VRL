# EXP0067：短时预测局部控制器复核

2026-10-07，复用已有 `LocalNavigator` 和 `trajectory_cost`，不新增启发式触发条件，不修改训练权重或 benchmark 指标。该控制器使用统一的短时 wall、障碍、进展、切换代价和 SDF 几何预测，作为当前 `semantic_reference_guard` 的原则化对照。

## Full 84-scene result

| 方法 | Task | Strict Safe | RSafe | 壁接触 s/局 | 最长连续壁接触 s/局 |
|---|---:|---:|---:|---:|---:|
| `local` | 27/84 (32.14%) | 13/84 (15.48%) | 22/84 (26.19%) | 20.910 | 8.023 |
| `local_replan` | 27/84 (32.14%) | 13/84 (15.48%) | 22/84 (26.19%) | 17.414 | 6.970 |
| 当前 `semantic_reference_guard` | 69/84 (82.14%) | 24/84 (28.57%) | 49/84 (58.33%) | 6.728 | 0.796 |

## Failure analysis

- `local_replan` 在若干 MCA、肺动脉和 ICA 场景长时间选择 `wait`；例如部分 MCA episode 的 `wait` 决策超过 2,000 个控制步。
- 这不是通过增加一个 clearance 阈值就应修复的问题。当前等待动作的状态机、血流漂移估计和候选动作集之间存在耦合，需要单独做模型预测控制的可行性分析。
- 因此不把 `local` 或 `local_replan` 接入 `semantic_reference_guard`，也不宣称预测局部控制带来收益。

## 决策

- 保留该实验作为原则化方法的负结果；不覆盖既有输出。
- 当前部署候选仍是 `semantic_reference_guard`，而不是全程 `LocalNavigator`。
- 下一步若继续研究预测控制，应先修复“等待导致无进展”的可行性/终端代价定义，并用小规模 paired pilot 验证后再扩展；不能直接通过额外启发式规则补洞。

结果文件：

- `research/validation/EXP0067_PREDICTIVE_LOCAL_20261007/full.jsonl`
- `research/validation/EXP0067_PREDICTIVE_LOCAL_20261007/full.jsonl.summary.json`
- `research/validation/EXP0067_PREDICTIVE_LOCAL_20261007/audit/`
