# 修复后的配对重评报告

## 起始提交与身份校验

本轮工作分支从 `d6940d0f352381af774a1711374c03d33aca963f` 建立。远端目标分支在该提交后没有更新。当前配对比较要求以下身份键完全一致并检查唯一性，`split`、`scene_seed`、`scenario_hash`、`realized_scenario_hash`、`accepted_seed`、`navigation_targets`。全平局仅表示观测到零差异，`equivalence_proven` 保持为 false。

旧 chunked 输入的严格身份交集为训练 4、留出 2。输入文件和身份键已写入 `paired_comparisons.json`。

## 配对局部控制结果

| method_a | method_b | split | 训练种子 | 配对场景数 | 胜 | 平 | 负 | 差值方向 |
|---|---|---|---:|---:|---:|---:|---:|---|
| IPPO-S41 | pursuit | train | 41 | 180 | 2 | 177 | 1 | a_gt_b |
| MAPPO-S41 | pursuit | train | 41 | 180 | 0 | 179 | 1 | b_gt_a |
| IPPO-S41 | MAPPO-S41 | train | 41 | 180 | 2 | 178 | 0 | a_gt_b |
| IPPO-S41 | pursuit | holdout | 41 | 100 | 0 | 100 | 0 | observed_zero_difference |
| MAPPO-S41 | pursuit | holdout | 41 | 100 | 0 | 100 | 0 | observed_zero_difference |
| IPPO-S41 | MAPPO-S41 | holdout | 41 | 100 | 0 | 100 | 0 | observed_zero_difference |

## 回归数字

训练侧导航成功 pursuit 152/180、IPPO 153/180、MAPPO 151/180。IPPO 对 pursuit 为 2/177/1，IPPO 对 MAPPO 为 2/178/0。留出侧三种方法逐场成败相同，均为 80/100，严格成功均为 77/100。脚本从 JSON 自动生成这些数字，旧数字只作回归校验。

## TPG on/off

TPG-on 与 TPG-off 共用场景、起点、靶点和局部 pursuit。`tpg_gate_calls` 是 gate 调用次数，`tpg_trigger_count` 是 hold 边沿次数，`waiting_transitions` 和 `waiting_robot_s` 分别是等待转入次数和累计等待 robot·s，`spacing_violation_pair_s` 来自 SpacingTracker，实体接触单独记录。

| N | on 全覆盖 | off 全覆盖 | on 最后到达 s | off 最后到达 s | on 触发次数 | on 间距违规 pair·s | off 间距违规 pair·s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 103.283 | 103.283 | 0 | 0.000 | 0.000 |
| 2 | 0.667 | 0.500 | 78.800 | 98.967 | 4 | 3.948 | 3.845 |
| 3 | 0.667 | 0.667 | 64.575 | 61.175 | 5 | 3.826 | 6.101 |

## 结论边界

IPPO 在训练侧仅有 2 个导航胜场、1 个负场，留出侧没有逐场导航差异。全平局不能证明方法等效。TPG 的 N=3 六场景全覆盖均为 4/6，on 成功回合最后到达时间 64.575 s，off 为 61.175 s，间距违规均值分别为 3.826 与 6.101 pair·s。这里报告的是导航到达和安全计量，不等同于药物释放或磁场干扰终点。
