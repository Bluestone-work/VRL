# EXP0065：进展绑定风险护盾与 clearance-gate 归因

2026-10-07，沿用 EXP0062/EXP0063 的图像感知、物理、执行器、奖励、指标和开发 seed 协议；不修改 PPO/DAgger 权重，不访问 sealed/test 数据。

## 动机

EXP0064 的 `risk_hysteresis` 能降低壁接触，但 Strict Safe 和任务完成出现回归。下一步把风险介入绑定到三个条件：

1. 风险残差达到高阈值；
2. 目标航段剩余距离在短窗口内没有改善；
3. 没有前方障碍需要语义绕行。

风险护盾最多承诺 `progress_commit_s=2.0` 秒，航段剩余距离改善 `progress_release_mm=0.3` mm 或承诺到期后释放。

实现：`ProgressBoundRiskNavigator`，位于 `marl/risk_navigation.py`。

## 重要归因控制

当前 `AllObstacleNavigator.act_all` 仍包含健康地图 clearance 硬门控。EXP0063 已经说明该门控可能把有效语义绕行切回 reference，导致任务失败。因此本实验新增：

- `semantic_no_gate`：保留同一个测量语义绕行和 wallguard，只移除 clearance mode switch；
- `progress_bound_risk`：在 `semantic_no_gate` 的前方障碍优先级基础上，再加入进展绑定风险护盾。

这样可以区分“移除错误硬门控”的收益和“新增风险护盾”的收益。

## Pilot

四个窄血管/回归解剖，24 个 paired episodes：

| 指标 | all_semantic | progress_bound_risk |
|---|---:|---:|
| Task | 11/24 (45.83%) | 17/24 (70.83%) |
| Strict Safe | 3/24 (12.50%) | 3/24 (12.50%) |
| RSafe | 6/24 (25.00%) | 6/24 (25.00%) |
| 壁接触 s/局 | 42.371 | 37.323 |
| 静态事件/局 | 3.833 | 2.625 |
| 动态事件/局 | 1.208 | 1.208 |

该 pilot 满足进入全量验证的条件；但它没有单独证明风险护盾收益，因为 progress-bound 前方障碍分支同时绕过了原 clearance gate。

## 全量 84 场景结果

### 当前实现 paired control

| 指标 | all_semantic | progress_bound_risk | delta |
|---|---:|---:|---:|
| Task | 60/84 (71.43%) | 68/84 (80.95%) | +8 |
| Strict Safe | 22/84 (26.19%) | 24/84 (28.57%) | +2 |
| RSafe | 46/84 (54.76%) | 47/84 (55.95%) | +1 |
| 壁接触 s/局 | 14.472 | 11.126 | -3.347 |
| 最长连续壁接触 s/局 | 4.862 | 4.637 | -0.225 |
| 静态事件/局 | 1.440 | 1.429 | -0.012 |
| 动态事件/局 | 0.643 | 0.619 | -0.024 |

paired 结果：Task 在 8 个场景提升、0 个场景下降；Strict Safe 4 个提升、2 个下降；RSafe 2 个提升、1 个下降；壁接触在 20/84 场景下降、17/84 上升，平均下降 3.347 s/局。

### 归因 control：semantic_no_gate

| 指标 | semantic_no_gate | progress_bound_risk |
|---|---:|---:|
| Task | 69/84 (82.14%) | 68/84 (80.95%) |
| Strict Safe | 25/84 (29.76%) | 24/84 (28.57%) |
| RSafe | 47/84 (55.95%) | 47/84 (55.95%) |
| 壁接触 s/局 | 10.864 | 11.126 |
| 静态事件/局 | 1.357 | 1.429 |
| 动态事件/局 | 0.631 | 0.619 |

结论非常明确：收益主要来自**移除不可靠的健康地图 clearance 硬门控**，不是来自进展绑定风险护盾。`progress_bound_risk` 相比 `semantic_no_gate` 反而少 1 个 Task、少 1 个 Strict Safe，并增加壁接触 0.262 s/局。

## 决策

- 当前最高 Task Success：`semantic_no_gate`，`69/84 = 82.14%`。
- 当前最高 Strict Safe：此前 `hybrid_replan` 的 `28/84 = 33.33%`；`semantic_no_gate` 为 `25/84 = 29.76%`。
- 不接纳 `progress_bound_risk` 为默认方案。
- 保留 `semantic_no_gate` 作为研究候选，但不直接修改历史 benchmark 默认方法；它需要与 frozen `rule_switch` 在预注册 acceptance gate 下重新做正式 paired 报告。
- 不再继续只调 risk threshold。下一阶段应转向 teacher/student primitive 接口和局部动作策略学习，避免把经典控制收益误报为 RL/DAgger 学习收益。

结果文件：

- `research/validation/EXP0065_PROGRESS_BOUND_RISK_20261007/full.jsonl`
- `research/validation/EXP0065_PROGRESS_BOUND_RISK_20261007/attribution_full.jsonl`
- `research/validation/EXP0065_PROGRESS_BOUND_RISK_20261007/full.jsonl.summary.json`
- `research/validation/EXP0065_PROGRESS_BOUND_RISK_20261007/attribution_full.jsonl.summary.json`
