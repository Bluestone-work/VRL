# EXP0069：论文方法的可部署迁移验证

日期：2026-10-07。目标是验证两个论文中常见、可迁移到当前血管环境的方法，而不是继续调参或把失败结果接入主线：

- CBF-QP safety filter；
- MPPI / sampling-based MPC。

## 公平协议

- 使用与 EXP0068 相同的 `image` 观测链路、同一环境和同一指标定义。
- 使用 4 个开发解剖、6 个开发 seed：`2600000000–2600000005`，共 24 个 paired episodes。
- 仿真时长 300 s；不使用 sealed/test 场景，不训练新模型，不修改 PPO/DAgger 权重。
- 两个方法都接收相同的 nominal route command；差异只在 nominal command 后的 CBF-QP 过滤器或 MPPI 滚动采样控制器。
- `Strict Safe` 对应环境现有 `cluster_safe_success`；`RSafe` 对应现有 `relaxed_safe_success`，不重新定义指标。

## 可部署信息边界

控制器只使用：

- 图像估计的 cluster 位置、速度、活动状态和 map-matching edge；
- 预术血管中心线、健康管腔半径和注册 SDF；
- 图像检测器输出的障碍相对位置、半径和由连续观测估计的相对速度；
- 控制周期、机器人几何参数和 nominal route command。

控制器不使用：

- `env.positions_mm`、真实 `env.edges`、真实质量或真实 clot 状态；
- `ObstacleField` 的真实障碍位置、动态障碍内部状态或仿真器副本；
- flow field、求解后的 occluded lumen radius、真实壁接触或未来轨迹。

运行时输入边界由 `scripts/audit_paper_method_privilege.py` 静态审计，并在双方法单场景运行中复核。修订后的评估脚本只记录首帧 deployable observation hash，不再记录真值初始状态 hash。`scenario_hash` 仅用于 paired 场景匹配，是评估元数据，不传入控制器。

## Pilot 结果

| 方法 | Task | Strict Safe | RSafe | 壁接触 s/局 | 最长连续壁接触 s/局 | 静态事件/局 | 动态事件/局 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `semantic_reference_guard` | 17/24 (70.83%) | 3/24 (12.50%) | 8/24 (33.33%) | 27.746 | 1.924 | 2.042 | 0.708 |
| CBF-QP safety filter | 1/24 (4.17%) | 0/24 (0.00%) | 0/24 (0.00%) | 58.190 | 40.766 | 0.417 | 0.292 |
| MPPI / sampling-based MPC | 12/24 (50.00%) | 3/24 (12.50%) | 9/24 (37.50%) | 22.050 | 12.338 | 1.333 | 1.500 |

最终 clean pilot 原始结果：

- `research/validation/EXP0069_PAPER_METHODS_20261007_DEPLOYABLE_PILOT/clean_pilot.jsonl`
- `research/validation/EXP0069_PAPER_METHODS_20261007_DEPLOYABLE_PILOT/clean_pilot.jsonl.manifest.json`
- `research/validation/EXP0069_PAPER_METHODS_20261007_DEPLOYABLE_PILOT/audit/`
- `research/validation/EXP0069_PAPER_METHODS_20261007_DEPLOYABLE_PILOT/semantic_reference_guard_clean.jsonl`
- 对照的 clean 结果不再使用旧 EXP0066 pilot；旧结果只保留作历史记录。

早期 pilot 文件保留在 `research/validation/EXP0069_PAPER_METHODS_20261007/`，但不作为最终结果引用。审计时发现旧评估器的目标选择路径直接读取了 `env.masses`；虽然该状态在当前单 cluster image 链路中通常与可见清除状态一致，但仍属于不允许的仿真器读取。修复 `PlanTargets.targets(..., clot_alive)` 并用 image 估计驱动后重跑 clean pilot；最终汇总数值未变，但只有 clean pilot 满足本实验的输入边界。

## 结果解释

### CBF-QP

CBF-QP 在 24 个 pilot 场景中 Task 仅 1/24，Strict Safe 和 RSafe 均为 0/24，壁接触和最长连续接触均显著恶化。当前实现不能作为安全过滤器接入主线。

这不是“再调几个 margin”即可接受的结果。后续若继续研究，必须先审计离散时间 CBF 约束的可行性、障碍相对速度定义、约束不可行时的投影行为以及命令归一化是否破坏约束；在此之前不做大规模参数搜索。

### MPPI

MPPI 比 CBF-QP 可用，但相对 clean `semantic_reference_guard` 是明显的 trade-off：Task 为 12/24 对 17/24，Strict Safe 同为 3/24，RSafe 为 9/24 对 8/24，壁接触降低到 22.050 s/局，但最长连续壁接触从 1.924 s/局恶化到 12.338 s/局，动态障碍事件也上升。它没有显示出足以替换当前候选的综合收益，因此不扩展到 84 场景、不接入默认执行链。

当前实现还暴露出方法层面的限制：短预测 horizon、简化的局部运动模型、终端进展代价不足以及检测障碍物的短期预测误差，可能共同造成局部保守或绕行失败。这些是待验证的模型假设，不应直接通过额外启发式规则掩盖。

## 评估器验证

双方法同时运行曾暴露评估脚本复用 CBF 配置的 bug，导致 MPPI 未启动；现已修复为每个方法独立构造 typed configuration，并加入回归测试。另修复了目标选择读取 `env.masses` 的特权路径。修复后的单场景双方法验证：

- 两个方法均无运行错误；
- 两个方法使用相同首帧 image observation hash；
- 输出不含 `initial_state_hash`，只含 `initial_observation_hash`；
- manifest 明确记录 `privileged_inputs: false`；clean pilot 的 48 条记录均无 `initial_state_hash`，且每条都有首帧 `initial_observation_hash`。

针对性测试结果：`6 passed`；特权输入静态审计：`PASS`。

## 决策

- 不把 CBF-QP 或 MPPI 接入当前主线。
- 不把本次结果包装成 RL/DAgger 学习收益。
- 在本次无特权 24 场景 pilot 中，`semantic_reference_guard` 仍是综合任务/安全折中的参考候选；历史 84 场景结果保留在 EXP0068，但不把它当作本次 clean protocol 的最终 84 场景结论。
- 论文方法保留为负结果和可复现实验基线；若继续研究，先做单场景约束/模型一致性诊断，再决定是否重新实现，而不是继续盲目调参。
