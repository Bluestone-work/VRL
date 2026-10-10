# EXP0090 协议（2026-10-11 00:55 登记，任何控制结果之前）

## 目的
检验"动作条件化的短时运动预测（学习的 belief / world model）参与指令选择"是否给 nav_tf_v3 学习导航带来**可归因于学习组件**的收益，与同框架 learner-off、Fixed Settle、SwitchSettle、STPG（本地适配）、PAC-NMPC（本地适配）配对比较。

## 审计结论（详见 `../EXP0090_AUDIT_20261011/AUDIT.md`）
- nav_tf_v3 s7101（EXP0073_LONG1M）与 s7102/s7103（EXP0081）配置只差 scene_seed_base（2.3e9 vs 2.1e9）与训练时代码版本；同为 500 updates、约 1.0 M agent steps、flow {.025,.05,.1}、latency ≤3、speed prior、prior_residual_scale=1.0。本实验把三者作为"近同配方 3 种子"，并在报告中注明差异。V5 nav_tf_v3_s0 是不同配方（prior_residual_scale 缺省→评估用 .5），不并入。
- belief_s7102/s7103 = arm C（辅助损失：训练期用仿真真值流速/响应标签，部署不读），只有 2 个种子。
- 旧 `report_nav_decomposition.py` 把未达 T90 删除（conditional）却与 T90_300 混用；本实验所有表同时给 T90_300、达到比例、conditional T90。
- 执行链：动作 → command → WallGuard（重规划 + 近壁投影）→ TPG hold 清零 → spacing shield → 执行器。learner 的输出可被覆盖，本实验逐步记录覆盖比例和幅度。

## 实验臂（第一轮）
| 臂 | 内容 | 学习组件 |
|---|---|---|
| A nav_off | 同一 NavController（历史、prior、动作映射、WallGuard、TPG、shield），网络输出置 0 | 无 |
| B nav | nav_tf_v3，3 种子（s7101/s7102/s7103，~1 M agent steps，已有 checkpoint） | PPO 残差 |
| C belief | nav_tf_v3 + 训练期辅助流/响应预测头（s7102/s7103） | PPO 残差 + 辅助损失 |
| D sel_learned | 学习预测器（3 个种子的集成）给候选指令打分并选择 | 预测器 |
| E sel_learned_fb | D + 不确定性回退到 SwitchSettle | 预测器 + 回退 |
| F sel_online | 同一选择器，预测器换成可部署在线仿射估计器（gain·cmd + drift） | 无 |
| 规则 | fixed_settle、adaptive_settle、switch_settle、stpg（本地适配）、pac_nmpc（本地适配） | 无 |

## 预测器
- 数据：`scripts/collect_dyn_data.py`，训练解剖，种子 2121000000+（训练 360 回合）、2122000000+（val_train 60）、2123000000+（留出解剖 val_heldout 60）；域：入口流速 {.025,.05,.1}、延迟 U{1,2,3}、v5 强度 0（p .3）或 U[0,1.25]，行为策略为规则与随机候选混合。
- 输入只含可部署量；未来位置、瞬时流速、响应增益只作监督标签。

## 事先固定的选择规则
1. **回退阈值**：`FALLBACK_STD` = val_train 上集成认知不确定性（1.2 s 处）的第 95 百分位 = **0.0445 mm**（已由 `prediction/prediction_eval.json` 计算，评估前固定）。
2. **选择器权重**：在调参场景（`eval_exp0090.py --protocol tune`：训练解剖、种子 2124000000+，与 EXP0073 评估场景、V5 开发场景、预测器数据均不相交；面板 low/high/strong/variable）上比较 3 组权重 W0/W1/W2，取 Strict 均值最高者，平分时取壁接触更低者。D、E、F 共用选出的权重。
3. **checkpoint**：nav/belief 一律用各 run 的最终 `policy.pt`，不逐场景挑选。
4. 评估场景：EXP0073 注册的困难矩阵（每面板 42 = 训练解剖 27 + 留出解剖 15），面板 low_delay、moderate_delay、high_delay、strong_flow、variable_response、ood_flow_delay；正式阶段再加 V5（s=0/1/1.5，N=1/2/3，14 解剖）。密封测试集不访问。
5. 第一轮若 D/E 不优于 A 与 SwitchSettle，最多再做两轮有依据的调整，如实报告。
