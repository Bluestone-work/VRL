# PROJECT_AUDIT — Physics-Informed Graph World Model for Cooperative Thrombolysis

日期：2026-09-20。阶段：Phase 0，正式训练前静态审计完成；运行验证结果另见 BASELINE_REPORT 与 PHASE_0_REVIEW。

## 范围与证据规则

审计对象是当前工作目录，而不是仅有一个提交的 Git HEAD `1dafc45`。已有 13 个 tracked 文件修改，以及 contact geometry、research ladder 和相关测试等未跟踪文件；这些不是本轮引入的算法变更。必须将当前源码另存为独立实验提交后运行，不能只报告旧 HEAD。

阅读范围覆盖 README/历史说明、environment、observation/action/reward、geometry/flow/clot/collision、MARL、world model、allocation、训练/评估/实验管理脚本与回归测试。已有中文分析作为辅助索引，结论以代码和原始 JSON 为准。渲染代码按入口和职责审阅，不作为动力学真实性的证据。逐文件 SHA256 和定义目录将保存在实验 provenance 中，便于核实审计对象。

已证实实现与潜在问题分开记录；尚未做行为验证的因果解释统一为 `HYPOTHESIS — NOT VERIFIED`。本文不把历史预期性能或脚手架的占位返回值当实验结果。

## 当前项目结构与任务定义

| 路径 | 职责 |
|---|---|
| `environments/vessel_geometry.py` | 分支图、连续中心线查询、Dijkstra 路由、Frenet 框架、约束投影、解析流速 |
| `environments/vessel_anatomy.py`、`vessel_tree_generator.py` | 14 类解剖血管、狭窄/血栓候选点、程序化随机树 |
| `environments/vascular_3d_marl_env.py` | 单回合 Gymnasium 仿真、接触消融、奖励、安全事件 |
| `environments/vector_env.py`、`balanced_vector_env.py` | 共享树的批量仿真、按场景均衡分组、自动重置 |
| `marl/geometric_control.py` | world/local/guided/flow_guided/flow_spread 动作解释 |
| `marl/maddpg_policy.py`、`mappo_policy.py`、`mappo_advanced.py` | MADDPG、MAPPO、可替换图/注意力 actor |
| `marl/world_model.py`、`transition_dataset.py` | 已有图动力学 ensemble、转移分片、按 geometry 划分 |
| `marl/task_allocator.py`、`hierarchical_allocator.py` | 最近/均衡分配、离散高层分配策略 |
| `scripts/train_*.py`、`eval_*.py` | 多条历史训练路线与评估入口，协议并不完全统一 |
| `scripts/auto_research_ladder.py`、`auto_research*.py` | 对照实验、配置/恢复/重试管理；不自动视作新研究流程的可信入口 |
| `tests/` | 环境、几何、单/批量对齐、策略、世界模型、编排回归 |
| `experiments/ladder_stage1/` | 当前实际存在的 12 组 1M-step checkpoint 与原始验证结果 |

任务是在随机三维血管树中独立控制多个微型机器人，导航至若干静止血栓，在有限步数内通过接触效应消除全部质量。当前最佳路线是基于几何引导的残差 MARL；现有物理不包含外部磁场驱动耦合、真实药物动力学或碎片输运。

## Observation

默认 geometric 为每机器人 36 维：位置 0:3，局部速度 3:6，路由前视 6:15，管壁外法向 15:18，径向余量/管腔半径/阻塞后半径比 18:21，局部流速 21:24，近壁推进系数 24，目标位移 25:28，沿血管距离 28，欧氏距离 29，接触 30，目标剩余质量比 31，最近邻位移与距离 32:36。

`geometric_v2` 为 42 维，在原布局后追加流速对数、沿路线流速对数、控制余量、回合进度、整体剩余质量、拥挤度。`legacy` 为 20 维。当前 EXP_0001 固定 36 维。

另有 `adjacency[N,N]`，按欧氏机器人距离 ≤0.06 构图并包含自连接；`clot_state[(nominal_clots+1),6]` 包含血栓位置、质量比、活动标志和归一化弧长，供 critic 使用。目标和路由依赖全局真值地图，不能声称仅凭真实局部传感完成任务。几何观测不包含完整历史，接触积累、压力场等未表示。

## Action 与运动

每机器人三维连续向量，裁剪后再限长度；速度标度 0.018。`flow_guided` 把流速感知引导与 0.2 倍 actor 残差相加，再由局部坐标转为世界坐标。引导速度参数 0.65。

位置更新为主动推进×近壁系数 + 血流 + 随机扰动 + 重叠分离；随后投影回管腔并修正撞壁法向速度。没有显式时间/质量 SI 单位标定。近壁最低推进系数 0.35，随机扰动随机器人半径按平方根缩放。每个机器人拥有独立动作权限；这是假定，不是已实现的全局磁控系统。

## Reward

团队项：10×本步去除质量 + 一次性里程碑(50%/75%/90%/99%) + 每清除一血栓 3 + 0.2×当前处理的血栓数 − step_cost；全部成功额外 30。

个体项：10×分摊消融贡献 + 0.1×同目标测地接近量 − 0.1×撞壁 − 0.1×同伴碰撞，成功再分摊 30/N。PPO 训练奖励为个体项 + 团队项/N；环境记录的 scalar return 为团队项 + 个体项均值，两者不能混用。

当前 `reward_double_count=on` 保留团队和个体里的进度/成功重复激励，`step_cost=0`。逐步 coverage 奖励可能使低效停留有利，属于 `HYPOTHESIS — NOT VERIFIED`，应做独立轨迹诊断与奖励消融，不在基线复现中修正。

## Clot dissolution 与阻塞

血栓固定在采样位置，具有初始和剩余质量。名义 3 个不代表每回合恰好 3 个：存在 2–4 的采样及解剖可用位点限制。

有效接触需欧氏距离 d≤0.035；geodesic 模式还需连续修正后的沿血管距离≤0.035。权重 w=exp(-(d/0.035)^2)，不接触时为零。每个血栓本步消融为 min(剩余质量, 0.018×min(Σw,4))。饱和针对加权总贡献，不是第 5 个机器人必然无效。默认目标是沿血管最近的存活血栓，但消融检查所有机器人—血栓组合。

血栓同分支附近施加高斯形状半径缩减：r_eff≈r×(1−0.65×mass_fraction×bump)。这改变解析血流，并不建立血栓实体接触或可钻穿障碍。没有剪切应力、治疗强度、药物量、接触时间依赖的动力学，也没有碎裂/再栓塞。不能凭该规则提出真实血栓材料参数的结论。

## Blood flow

每分支按预分配 flow_fraction 分流；平均速度随 (r_ref/r_eff)^2 缩放，并限制到入口流速的 8 倍；横截面使用 2×mean×(1−(radial/r_eff)^2) 的非负剖面。入口标度 0.004，中心速度最高约 0.064。血栓消融改变局部半径和流速，但分支流量份额不会由全局压力网络重算，所以“flow redistribution 已被学习/实现”目前不成立。

36 维观测把流速/max_speed 裁剪到 [-1,1]，高流速信息可能丢失；42 维扩展提供额外表示，但本轮不引入。流速控制器反推中心速度并横向偏移，接近血栓后试图抵消血流以维持接触；其常数针对当前模拟器。

## Collision 与多智能体交互

机器人是有限半径粒子；血管投影产生 wall contact，机器人两两重叠产生分离位移和惩罚。它不是连续接触力/应力求解，离散大步长、分叉和狭窄处需特别验证。GAT 的机器人连接按空间距离构建，可能跨相邻但不连通的血管壁传递信息；通信物理没有建模。

`flow_guided` 的协作来自邻接消息、共享策略、目标观测和奖励。高层 allocator 与 `flow_spread` 是另外模块，未启用于复现。图结构本身不证明学到了有效分工或可扩展性。

## MARL 与现有算法

MADDPG 使用共享 actor/中心化 critic、replay buffer 和目标网络；MAPPO 用 tanh-Gaussian actor、GAE 与 PPO。高级实现支持 gat、edge_gat、edge_bias_gat、mlp、Transformer 等，但当前向量训练入口只开放 gat/edge_bias_gat/mlp。

当前 GAT 实现实际使用 masked scaled dot-product attention；名称和文件注释不足以证明与原始 additive GAT 完全相同。`critic_value_mode=v` 不输入动作；q 对照输入动作；它们都在 MAPPO 框架中。不能把 q 对照写成 MADDPG。架构工厂对 Transformer 也更换 critic，与“critic 保持一致”的旧注释存在冲突；跨架构研究必须核查变量。

`progressive_training.MultiStageTrainer._train_stage()` 存在占位返回，包括固定 success_rate=0.5；不能把这些占位结果用于论文或注册表。当前复现不调用它。

## 现有 World Model：已有基础及缺口

`GraphWorldModelEnsemble` 是机器人节点图的前馈动力学 ensemble；输入 observation、action、机器人位置/速度、扁平血栓状态、几何摘要及场景 embedding。输出节点变化、全局状态变化、位置变化、速度、reward 和 done 概率。血栓/血管不是独立异构图节点；没有 RSSM 或时序潜变量信念。

训练使用 bootstrap ensemble 与多头 SmoothL1/BCE。已有 1/3/5 步开环误差和 persistence/zero-reward 参考，但没有 10/20/50 步的专门 clot/flow/collision 误差、物理约束验证、uncertainty 校准或 OOD 安全评估。generic observation 中包含流速和质量，不等于已验证了物理一致耦合。

MVE 在当前实现中混合 imagined return 到 critic target，actor advantage仍来自真实 GAE；没有在线 MPC、候选分配搜索或重规划。它拒绝 `flow_guided`，因为旧模型假定 world-frame action；数据中 raw action 和 executed_action 已分开记录，后续研究必须明确动作语义。

世界模型预测统一把 obs/state 裁剪到 [-1,1]，不保证质量非负/单调、不保证位置与观测一致或守恒。done target 来自 `done`，包括时间截断；若用于 imagined termination 会混淆任务终止和观测窗口截断。这些问题必须在进入 Phase 3 前登记验证。

## 训练流程

`train_vector_mappo.py`：固定 NumPy/Torch seeds → 按 14 个 territory 均衡分配 64 个向量环境 → 每128步采样一次，共8192 transitions/常规update → 5个epoch PPO → 周期验证、checkpoint与日志。每900次环境步重采样子环境血管，未结束回合标 truncated，bootstrap使用 final_observation，防止跨回合取值。

保存初始、验证最优、周期和最终参数；支持保存环境/RNG/optimizer恢复。默认训练未强制 deterministic CUDA；硬件与库版本必须记录。原线程/CPU 亲和性脚本排除CPU6/7，依据是历史稳定性疑虑，尚不是本轮硬件故障诊断。

## Evaluation protocol 与成功条件

当前训练验证采用14类×5回合，起点100000且场景间隔10000。最终评估为14×20，仍从100000开始，因此包含验证回合，明确是validation。`best_score`属于小验证集最优模型；`final_evaluation`属于最终参数，不能把best score配到final checkpoint上。

成功=全部质量清零；清除率=1−remaining/initial；不能以90%质量清除冒充成功。wall_hits_total是各机器人跨整回合的接触总和，每步计数可超过1；应另提供按robot-step归一化的rate。first_contact_step在移动后、steps加1前记录，零表示第一次转移即接触，不代表reset时已经接触。

当前可核验参考：geodesic_v 42/43/44最终validation成功率72.1429%/75.0000%/71.4286%，均值72.8571%、样本标准差1.8898%；平均质量清除92.7773%。旧README约92.26%是旧实验训练后半段统计；SUCCESS_STUDY的70.83%是缺少当前原始目录的历史测试报告，均不能替代本轮基线。

## 科研风险与可能的 bug

| ID | 证据与影响 | 状态/处理 |
|---|---|---|
| A01 | 当前dirty源码与唯一Git提交不一致 | 已确认；实验冻结快照和逐文件hash |
| A02 | final验证含训练验证回合，旧脚本不同默认半径/池/接触模式 | 已确认；原协议保留为复现，独立开发验证另列，封存测试集不用于调参 |
| A03 | policy_loader恢复网络/控制模式但不完整恢复环境参数 | 已确认；外部诊断显式读取配置并核对meta |
| A04 | 训练episode的wall_collisions仅末步；另有wall_hits_total | 已确认；报告只用明确累计字段，外部评估累计pair collision |
| A05 | world model把truncation作为done；generic clamp不保证物理一致 | 静态问题已确认；性能影响待独立实验 |
| A06 | world model的gate指标计算于命名test split，并被下游用来决定模型使用 | 已确认；未来必须用validation做gate，test仅最终报告 |
| A07 | 多来源数据对geometry_id偏移，但sequence key仅episode_id/step | 潜在冲突；同seed多数据来源需检查全局episode唯一性，未运行验证 |
| A08 | geometry group ID不是全局内容hash，跨run物理相同树可能进入不同split | 潜在泄漏；未来按geometry内容hash固定split |
| A09 | reward重复项、coverage可能鼓励拖延，图通信可能穿壁 | 机制可见；实际reward hacking为HYPOTHESIS — NOT VERIFIED |
| A10 | 旧预训练脚手架含固定成功率占位 | 已确认；禁止作为实验记录 |
| A11 | 相同物理的单/批量初始化与终止标志边界并非完全相同 | 部分静态差异：单环境horizon步成功可同时term/trunc；需回归覆盖，不改基线 |
| A12 | 历史native crash、resume的库/硬件依赖 | 历史记录，当前待测试；失败原样留存、有限重试，不按性能重跑挑seed |

## 当前创新空间与真实控制差距

可检验方向：显式机器人/血栓/血管异构图；潜在时序状态；质量/接触/有效半径的物理一致预测；不确定性约束的多步规划；面向不同机器人数量的分配规划。均为 `HYPOTHESIS — NOT VERIFIED`。

所有“首次/新颖/优于已有方法”判断：`NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION`。已有图世界模型和MVE意味着简单再加入GNN或ensemble不能当作本项目的新贡献。

真实控制尚缺：共同磁场耦合、执行器带宽/延迟/饱和标定、真实观测噪声与失联、患者几何、搏动和压力边界、流固耦合、血栓材料/反应、剪切/热/组织损伤、碎片安全与实验验证。当前可研究“该仿真中的耦合控制”，不能外推临床疗效。

## Phase 0 行动

不改物理、reward、observation、actor/critic或控制器。冻结当前源码；记录依赖/设备；运行回归测试和短训练；以EXP_0001从头训练固定3 seeds各1M transitions；沿用历史validation复现，并用预登记的独立开发验证补充诊断。测试集留到协议审查后再建立并封存。任何失败进入FAILURE_LOG；Phase 0 gate通过前，不开展主要创新。

## Runtime evidence after EXP_0001 / EXP_0002

本节为静态审计后的追加证据，不回写原始审计判断。EXP_0001三seed原协议指标与actor/critic张量精确复现。EXP_0002确认A02：原final validation有70/280（25%）回合曾用于模型选择；它不是独立test。A03通过新增显式配置/meta/default校验防止本研究评估漂移，但旧入口没有被全局修改。

840对CPU/GPU同权重/初始状态回合中10次success翻转，未来比较固定cuda:0。同设备84对重复及CPU840个父实验回放精确一致，证据限本机，不保证跨机器等价。A12历史native问题本轮未重现，不能宣称彻底解决。

完整geometry hash加入连接/flow_fraction，未来560/140/280实例清单互斥；测试仅登记、不评分。历史训练树仅部分可恢复，不能排除全部训练泄漏；新清单还需后续训练器显式消费。具体见[EVALUATION_PROTOCOL_REPORT](EVALUATION_PROTOCOL_REPORT.md)。H1–H5及I1–I7依然未验证。
