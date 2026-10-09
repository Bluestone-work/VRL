# EXP0068：同观测、同指标的先进方法横向比较

日期：2026-10-07。目标是先比较已有方法，不继续修改当前导航策略。

## 公平协议

- 统一使用图像观测：`sensing_model=image`。
- 统一使用 84 个开发场景、6 个开发 seed：`2600000000–2600000005`。
- 通过 `scenario_hash` 逐场景配对；所有主表方法均为 84/84，无错误。
- 统一报告 Task、Strict Safe、RSafe、壁接触、最长连续壁接触、静态/动态障碍事件。
- 不把 `noise` 观测的旧 DAgger 结果放进主表；它们需要重新用 image 观测运行后才可公平比较。

## 主结果

| 方法路线 | Task | Strict Safe | RSafe | 壁接触 s/局 | 最长连续接触 s/局 | 静态事件/局 | 动态事件/局 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 `semantic_reference_guard` | 69/84 | 24/84 | 49/84 | 6.728 | 0.796 | 1.155 | 0.548 |
| `rule_noavoid` | 72/84 | 1/84 | 6/84 | 8.433 | 8.433 | 3.667 | 3.202 |
| APF | 27/84 | 14/84 | 23/84 | 9.522 | 2.763 | 0.012 | 0.095 |
| `rule_switch` | 56/84 | 15/84 | 52/84 | 12.540 | 9.153 | 4.905 | 1.286 |
| 视觉 BC/VP | 37/84 | 0/84 | 6/84 | 5.501 | 5.460 | 1.726 | 2.321 |
| 视觉 PPO + wall-aware action | 38/84 | 15/84 | 29/84 | 22.390 | 22.073 | 0.583 | 0.619 |
| 视觉 world-model PPO | 59/84 | 16/84 | 38/84 | 28.897 | 26.715 | 0.381 | 0.310 |
| 短时预测 `LocalNavigator` | 27/84 | 13/84 | 22/84 | 17.414 | 6.970 | 1.845 | 0.750 |

主结果原始 JSON：`research/validation/EXP0068_IMAGE_METHOD_COMPARISON_20261007.json`。

## 解读

1. `rule_noavoid` 的 Task 最高为 72/84，但 Strict Safe 只有 1/84，说明只看任务完成率会奖励危险贴壁/碰撞行为。
2. APF 的障碍事件最低，但 Task 只有 27/84，体现了避障与推进之间的经典折中。
3. 现有视觉 BC、视觉 PPO 和 world-model PPO 没有在这套 image 观测协议下超过当前候选；尤其 world-model PPO 的壁接触和最长连续接触明显偏高。
4. 现有短时预测控制器比部分学习方法更容易解释，但大量等待导致 Task 只有 27/84，因此当前不能作为主线替代方案。
5. 当前 `semantic_reference_guard` 是表中唯一同时保持较高 Task、较高 RSafe、较高 Strict Safe 且把最长连续壁接触压到 1 秒以内的方法；但它仍是经典控制候选，不是 RL/DAgger 学习收益。

## 可迁移的论文方法候选

### 1. CBF-QP safety filter：最适合下一步迁移

迁移方式：保留任意策略的 nominal command，用图像估计位置/速度和注册血管 SDF 构造局部安全约束，再求最小修改的安全动作。它比当前 wallguard 更规范，因为安全过滤器的目标是“在约束下尽量接近 nominal”，而不是手工切换模式。

当前限制：环境没有现成 QP 依赖，且需要先明确离散时间、血流漂移和估计延迟下的可行约束。下一步应先做离线 action-filter replay，再做 24 场景 pilot；不能直接宣称安全证书。

### 2. MPPI / sampling-based MPC：可迁移，但需要显式动力学模型

迁移方式：对图像观测得到的状态估计、血流漂移和障碍速度采样动作序列，按 wall/obstacle/progress 代价滚动优化。

当前限制：已有 `LocalNavigator` 证明“有短时代价”不等于“可用”。它在困难场景过度选择等待。因此需要先解决终端进展代价和可行性，再与 CBF filter 做组合比较。

### 3. Recurrent PPO / world-model PPO：已经完成同观测对比

本地已有视觉 PPO 和 world-model PPO 结果，已纳入主表。它们可以继续作为学习方法基线，但现阶段没有超过当前经典候选，不能用更复杂模型替代公平评估。

### 4. DAgger / policy distillation：暂不纳入主表

本地 DAgger 结果主要是 `noise` 观测或不同训练协议。若要比较，必须在当前 image 观测、同一 84 场景、同一指标下重新运行；否则会把观测差异误报成算法差异。

## 决策

- 当前候选保持不变，不做盲目改进。
- 主线对比结论：`semantic_reference_guard` 是当前综合表现最佳，但优势属于经典安全执行层。
- 下一项正式实验优先选择 CBF-QP safety filter；先做 action-filter replay 和 pilot，不直接改训练奖励或 PPO/DAgger 权重。
- MPPI 作为第二候选，前提是先修复已有 `LocalNavigator` 的等待/终端代价问题。

## 后续审计说明

后续 EXP0069 审计发现，本实验生成时 `Episode.observe()` 的目标选择路径直接读取了 `env.masses`。该路径在单 cluster 场景通常与图像可见的清除状态一致，但不满足严格的无特权输入协议。因此本文件和对应 84 场景结果保留为历史横向比较，不作为 clean deployable benchmark 的最终结论；后续正式比较必须使用 `PlanTargets.targets(..., clot_alive)` 的观测状态路径并重新生成结果。
