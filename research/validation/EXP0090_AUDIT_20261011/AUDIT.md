# EXP0090 审计


## 范围与限制
本审计只读核验代码、checkpoint/config、JSONL/CSV；未训练、未修改既有文件、未提交。项目路径实际以句点结尾。除本文件外无写入。

## 1. 训练实现与动作映射
- `scripts/train_lysis_nav.py:48-49,84-113`：GAE γ=.995、λ=.95；基础 token 39 维，ECG 加2，`observable_history` 加1。
- `:116-165` 保留 `command_legacy`，当前 `command` 把速度写成 `clip(prior + residual_scale*clip(a0,-1,1),0,1)`，横向使用 route frame 的法向/副法向偏移；不是纯RL。
- `:169-193` `PriorSpeed(adaptive=False)` 为固定半径 .3 mm 的 settling prior；默认训练 prior residual scale 在 `:320` 为1.0。
- **评估动作尺度不一致**：`NavController.__call__` 在 `:246` 用 `cfg.get('prior_residual_scale', .5)`；训练 rollout 在 `:320` 用默认1.0。旧 V5 checkpoint 缺字段，因而评估走 .5；含显式字段的新 checkpoint 走其保存值。
- `:322-332` reward 读取真值位置/去除/壁接触/损失：进度、+50 removal、−3 wall、−10 lost、动作变化与 live 惩罚，多机器人另 −.5 spacing。奖励真值不等于 actor observation 泄漏。
- `:351-380` time-limit truncation bootstrap；任务终止/机器人退出切 stream；held/TPG 行留在 GAE stream 但 `learn=False`，不进入 PPO 更新。

## 2. 执行链
`benchmark_lysis.py:46-47,127-140,203-242` 和 rollout `:247-258`：policy/低层 rule → WallGuard（EventReplanner 可替换 rule，WallRecoveryExecutor 可恢复/投影）→ `LysisEpisode.step` 先把 hold 行清零 `:130` → spacing Shield.filtered `:132-134` → local-to-world → actuation/variation `:135-139` → physics。故 learned action 可被 WallGuard、TPG hold、spacing shield 覆盖/投影，最终执行量不等于 policy 输出。

## 3. T50/T90/T100 口径
`benchmark_lysis.py:102,159-161,172-194` 用 None 表示未达阈值；默认 horizon 300 s（`:48,543-545`）。`summarize_lysis.py:27-31`、`report_hard_baselines.py:22,48-51`、`evaluate_hard_baselines.py:55` 将 None 回填300并另给 reached/conditional；但 `report_nav_decomposition.py:10,16-24` 直接跳过 None，表头仍写 T50/T90/T100，未标 conditional、未报 reached 数，且 `:66-69` no_learning 参考使用 T90_300，形成同报告混口径。

## 4. V5 与 hard matrix
V5 开发 seed 函数为 `benchmark_lysis.py:39-42`：base 2,600,000,000、按 anatomy stride 100,000；默认 horizon 300，legacy flow（未传 `flow_inlet_mm_s`）与默认 sensing latency=1。hard panels 在 `evaluate_hard_baselines.py:11-13`：low (.05,1,0)、high (.05,3,0)、strong (.1,2,.625)、variable (.05,2,1.25)，并从 EXP0073 manifest 读取 anatomy/N/seed；可见调 `flow_inlet_mm_s`、latency、variation。两者 flow 物理标定、latency、variation、seed 协议不同，不可把百分比直接合并或宣称算法退化/提升。

## 5. 消融结果的已核验问题
`EXP0080.../aggregate.csv` 每 cell n=140；例如 s10,N1,full strict=.8357，T90 列167.624 是 reached-only 的均值（JSONL 有18个 None），若统一 T90_300 应184.644。`report_nav_decomposition.py` 的 None 丢弃会选择性降低慢方法；同一报告 no_learning T90_300 不能与之横比。

## 6. 最新输出状态（初步）
`research/validation/SCHED_20261010/*/episodes.jsonl` 行数：baselines 1260，flowres_s5 252，flowres_cons_s6 252，switchres_s0 252，oracle 168；EXP0081 validation 目录当前仅 PROTOCOL.md，未见评估 JSONL，不能称已完成。`EXP0080.../REPORT.md` mtime 为 2026-10-10 22:55:02，aggregate 存在。

## 7. classical / residual / paper-baseline边界
- `benchmark_lysis.py:356-400`：Fixed `SettleGuard` 是 WallGuard 后在半径 .3 mm 内按 d/r 缩放；Adaptive 另维护响应 EMA、10点停滞检测和2 s release。
- `:404-420` SwitchSettle 每步同时推进 fixed/adaptive，按 frame age 选择输出；但该类本身未见 `@method` 注册（不能声称自动 METHODS 包含它）。
- `marl/sched_settle.py:26-180`：ScheduledSettle/SwitchResidual/FlowResidual 的动作是 speed/release/radius/lateral 或 speed/drift 等 residual；文档约定零动作回归 prior，不等于 `classical_settle` 的完整类身份。
- `train_lysis_nav.py:169-193` PriorSpeed(adaptive=False) 只生成 route-frame speed prior（a=(speed−1,0,0)），`command` 仍按 carrot 方向重新构造 lateral aim；WallGuard 后置，且 `prev_world` 由上一步执行 local 转 world 更新。故 NavController(a=0) 仅在无事件重规划、同一 target/状态、scale=1 等条件下接近 Fixed Settle；不能宣称与 `classical_settle` 逐步 exact equality。
- PAC-NMPC/STPG 代码说明见 `marl/lysis_baselines.py:1-20,43-153,199-229`：分别为 3D 一阶 cluster + 采样/PAC bounds、以及 N≤3 周期在线枚举 precedence flips 的仓库适配；不是作者代码复现。注册见 `benchmark_lysis.py:306-330`。Turbo direct 是 `train_lysis_local.py:102-133,231-261` 的 TemporalPolicy 直接输出，非外部实现复现；direct 默认无 WallGuard，仅保留 episode spacing shield。

## 8. 最新实验与数字
- EXP0080 final 的 `aggregate.csv` 共 5 variant × 3 strength × 3 N = 45 cells，每 cell 140 回合（6300 rows）。s10,N1,full：strict 83.57%，conditional T90=167.624 s；18/140 未达，统一 T90_300=184.644 s。该差异足以改变 guard/no_guard 的效果结论。
- EXP0081 `PROTOCOL.md` 定义 s7102/03 原始 v3 与 belief 辅助头（训练特权流速/响应标签，部署不使用）；validation 目录当前只有 `PROTOCOL.md`，无评估 JSONL。runs 下训练 log/checkpoint 存在，但 config `agent_steps` 字段为0、`updates=500`，不能仅凭 checkpoint 文件称评估完成；状态应记为评估未完成/未证实。
- SCHED validation 已有：baselines 1260 行，flowres_s5 252，flowres_cons_s6 252，switchres_s0 252，oracle 168；均是部分面板/不同 scope，不能互称完整 hard matrix。switchres 训练 `log.jsonl` 最后一行 it=2182、agent_steps=18,732,363、strict=.465、success=.505、strict_lat3=.4632、strict_lat1=.4925（训练日志指标，不是 validation strict success）。

## 9. 明确问题及受影响结果
1. **优先级最高：** `report_nav_decomposition.py:16-24` 把未达 T 值的 None 删除，而表头未声明 conditional；EXP0080所有消融 T50/T90/T100均受影响，尤其 s=1,N=1 的 T90。
2. **动作尺度兼容陷阱：** 旧 V5 缺 `prior_residual_scale`，评估 fallback=.5；新训练 rollout fallback=1.0（`train_lysis_nav.py:246,320`）。旧/新同名 nav_tf_v3 不能按同一动作 recipe pooling。
3. **协议不可比：** V5 legacy flow/latency与hard显式 flow/latency/variation、seed体系不同；EXP0073训练分布（multi-flow/latency）也不同于原V5 seed0（config.json）。
4. **安全层非共同项：** report中“所有方法共享 WallGuard”不适用于 Turbo direct；FlowOracle 另是 privileged upper bound（`benchmark_lysis.py:429-477`）。
5. `auxiliary.jsonl` 的 weighted_loss 硬编码 .01（`train_lysis_nav.py:512`），当 CLI aux_weight 非.01时日志值与真实 loss（`:498`）不一致。
6. SCHED/EXP0081 文件有训练或部分验证产物，但 scope、行数、panel不同；不得把 252/168 行结果与1260行全基线或 V5 每强度420回合合并。

## 10. checkpoint/config核验摘要
以下结论来自逐个目录的 `config.json`、`log.jsonl` 与 `policy.pt`/manifest；manifest 的 EXP0080 checkpoint sha256 为 `0f7b1677f687284a74c1e2ac43c00c0872e2ce15a4871a7d1f7852d68efcb283`，与 `research/runs/EXP0073_LONG1M_20261009/nav_tf_v3_s7101/policy.pt` 一致（EXP0080 manifest）。

|运行|recipe关键差异|训练/日志状态|
|---|---|---|
|V5 `research/runs/V5_20261008/nav_tf_v3_s0`|20 workers、120 min、seed0、layers3/window32、s_max1.25、speed_prior、beta .5→.02；无 prior_residual_scale、flow_levels/latency_max 字段|log 约2000行；policy mtime 2026-10-08；旧动作评估 fallback=.5|
|EXP0073 `.../nav_tf_v3_s7101`|4 workers、120 min、updates500、seed7101、flow [.025,.05,.1]、latency_max3、beta 1→.05、prior_residual_scale=1、scene base 2.1e9|log 500行；末行 it500，agent_steps=1,006,040，success=.30348，removal=.52676，wall=3.0697，t90=249.43|
|EXP0080 s7101 / final|final manifest引用上行 EXP0073 policy；不是新训练 recipe|final评估 6300 rows；policy 2,443,506 B，mtime 2026-10-09 19:49:04|
|EXP0081 nav s7102/s7103|4 workers、updates500、flow三档、latency3、aux=.01、matrix_arm=nav_tf_v3、seed7102/03|每个 log 500行；最终训练日志 success=.32367/.32178、removal=.52425/.55376、wall=.58957/.87389、t90=244.84/250.11|
|EXP0081 belief s7102/s7103|同上但 matrix_arm=belief；训练期特权辅助预测流速/增益，非无特权监督|最终 success=.33493/.32020、removal=.52925/.54461、wall=.48169/2.29208、t90=241.65/249.30|
|EXP0076 observable smoke/reference|`EXP0076_OBSERVABLE_SMOKE_s7600/config.json`：2 workers、updates2、observable_history=True、no_privileged_supervision=True、flow [.025,.05]、latency2、CPU；另有 `EXP0076_FAIR_REFERENCE_s7601`、`EXP0076_OBSERVABLE_HISTORY_s7601`，不可与V5/EXP0073当多seed|smoke/短预算，不是正式同配方|

EXP0073/EXP0081 都是显式多流量、多延迟的新训练分布；V5 原始 s0 是旧 legacy flow/默认 latency 与不同 beta/worker/budget。EXP0080 final 只是复用 EXP0073 s7101 policy，不能作为 V5 s0 的独立新 seed。belief 组含训练特权辅助损失；nav 组不能与 belief 直接视为同 recipe。上述 recipe、预算、动作尺度、推理安全层任一不同，均禁止 pooling 为“multi-seed”。

`research/figures/SCHED_AUTO_20261010/flowres_cons_s6/REPORT.md:5-54` 的 paired FlowResidual vs SwitchSettle（每 panel n=42，总 n=252）strict：high_delay 71.4 vs 69.0%（+2.4 pp），low 88.1 vs88.1，moderate 50.0 vs50.0，ood 35.7 vs35.7，strong 11.9 vs11.9，variable 31.0 vs31.0；overall 48.0159 vs47.6190%，仅 +0.3968 pp。该结果是 paired screening，不是 V5 全量或 EXP0081 多seed评估。
