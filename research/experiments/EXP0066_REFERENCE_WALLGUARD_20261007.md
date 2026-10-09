# EXP0066：全局 reference 的测量型 wallguard

2026-10-07，沿用 EXP0065 的图像感知、topo pursuit、物理执行器、奖励、指标和开发 seed 协议；只使用 `2600000000 <= seed < 2700000000`，不访问 sealed/test 数据，不修改 PPO/DAgger 权重。

## 动机

EXP0065 的日志显示，`semantic_no_gate` 的主要残余失败并不总发生在障碍绕行期间。多个高壁接触场景在没有前方检测障碍时长期执行 `reference`，导致持续贴壁。已有 `WallRecoveryExecutor` 只在语义障碍分支上执行，因此新增独立候选 `semantic_reference_guard`：

- 前方检测障碍：沿用 `semantic_no_gate` 的语义绕行和 wallguard；
- 无前方检测障碍：对全局 `reference` 也使用同一个测量型 wallguard；
- 输入只包括图像估计、地图匹配和当前命令，不读取仿真器真实 wall contact、真实 lumen 或 obstacle truth；
- 不使用健康地图 clearance 作为硬切换条件。

## Pilot 消融

四个高风险解剖、6 个开发 seed，共 24 个 paired episodes。

| 配置 | Task | Strict Safe | RSafe | 壁接触 s/局 | 最长连续壁接触 s/局 |
|---|---:|---:|---:|---:|---:|
| margin 0.35, gain 0.8：`semantic_no_gate` | 18/24 | 3/24 | 6/24 | 36.491 | 15.010 |
| margin 0.35, gain 0.8：`semantic_reference_guard` | 17/24 | 3/24 | 7/24 | 26.006 | 1.651 |
| margin 0.20, gain 0.8：`semantic_no_gate` | 19/24 | 3/24 | 8/24 | 31.018 | 12.932 |
| margin 0.20, gain 0.8：`semantic_reference_guard` | 18/24 | 3/24 | 9/24 | 20.716 | 1.594 |
| margin 0.20, gain 0.4：`semantic_no_gate` | 17/24 | 3/24 | 8/24 | 41.221 | 16.197 |
| margin 0.20, gain 0.4：`semantic_reference_guard` | 18/24 | 4/24 | 10/24 | 21.770 | 1.724 |

margin 0.20、gain 0.4 在 pilot 中同时改善任务和安全指标，因此进入完整开发集验证。所有 pilot 结果均保留；没有因为结果较差而删除或替换日志。

## Full 84-scene paired result

最终配置：`recovery_margin_mm=0.20`、`recovery_gain=0.4`。

| 指标 | `semantic_no_gate` | `semantic_reference_guard` | paired delta |
|---|---:|---:|---:|
| Task | 68/84 (80.95%) | 69/84 (82.14%) | +1/84 |
| Strict Safe | 22/84 (26.19%) | 24/84 (28.57%) | +2/84 |
| RSafe | 47/84 (55.95%) | 49/84 (58.33%) | +2/84 |
| 壁接触 s/局 | 12.354 | 6.728 | -5.626 |
| 最长连续壁接触 s/局 | 4.973 | 0.796 | -4.177 |
| 静态障碍事件/局 | 1.095 | 1.155 | +0.060 |
| 动态障碍事件/局 | 0.571 | 0.548 | -0.024 |

84 场景 paired bootstrap 95% 区间：

- Task delta `+0.0119`，区间 `[-0.0238, +0.0476]`；
- Strict Safe delta `+0.0238`，区间 `[-0.0238, +0.0714]`；
- RSafe delta `+0.0238`，区间 `[-0.0238, +0.0714]`；
- 壁接触 delta `-5.626 s/局`，区间 `[-11.716, -1.170]`；
- 最长连续壁接触 delta `-4.177 s/局`，区间 `[-9.226, -0.680]`。

## 决策

- 接纳 `semantic_reference_guard` 作为当前安全优先的研究候选；它在本轮同配置 paired 结果中同时提高 Task、Strict Safe 和 RSafe，并显著减少持续贴壁。
- 不把它称为 RL/DAgger 学习收益；收益来自部署侧的经典测量型 wallguard。
- 不覆盖既有 benchmark 默认方法；历史最高 Strict Safe 仍是 `hybrid_replan` 的 28/84，历史最高 Task 仍为 69/84，本候选的 Task 与其并列。
- 静态事件均值略升，后续应针对“wallguard 与静态障碍绕行的冲突”做单独消融，而不是继续盲调 risk threshold。

## 结果与审计文件

- `research/validation/EXP0066_REFERENCE_WALLGUARD_20261007/full_margin020_gain040.jsonl`
- `research/validation/EXP0066_REFERENCE_WALLGUARD_20261007/full_margin020_gain040_paired.json`
- `research/validation/EXP0066_REFERENCE_WALLGUARD_20261007/full_margin020_gain040.summary.json`
- `research/validation/EXP0066_REFERENCE_WALLGUARD_20261007/audit_full_margin020_gain040/`
- `research/validation/EXP0066_REFERENCE_WALLGUARD_20261007_config_margin020_gain040.json`
