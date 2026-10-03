# Innovation tracker

EXP_0015 已注册但尚未形成性能结论：`geometric_dynamic` 只提供当前最近动态障碍物的局部状态，
`risk_aware_connectivity` 只改变血栓目标分配并记录当前路线风险；动作仍是 Direct Local
Frenet policy 输出。路径长度采用所有机器人实际位移之和，同时报告每单位清除质量的路径代价。
85% 是预注册验收门槛，不满足时必须报告失败，不能通过测试集选参或修改障碍物轨迹来宣称达到。

EXP_0015 已完成 3 seed × 1M transitions、420 回合独立动态验证：成功率
53.33% ± 19.68pp，质量清除 87.27% ± 7.51pp，路径长度 7.6163 ± 0.8020，
路径/清除质量 3.2220 ± 0.2637。85% gate 失败；seed 44 的 73.6% 不能替代三 seed
均值。该结果保留为真实的研究候选，不声称最优，也不跳过后续 observation/scheduler
消融或动态基线对照。

EXP_0016 已注册并运行：在 EXP_0015 基础上加入当前状态的 0.45 步常速度预测、
TTC/预测间隙/碰撞概率/漂移不确定性特征，以及 `predictive_risk_connectivity`
调度。训练早期没有数值异常；最终性能尚未验证，85% gate 仍然有效。

研究假设：模型显式学习 robot–clot–flow coupled dynamics，并用预测支持分配/规划。`NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION`。Literature verification required.

这里的current VRL指本仓库已有血管强化学习代码，不假定它对应某篇未提供的论文。

| ID | Motivation | Difference from current VRL | Difference from existing RL / World Models | Implementation | Experiment Evidence / Performance Contribution / Ablation Evidence | Current Confidence | Novelty Risk | Remaining Validation |
|---|---|---|---|---|---|---|---|---|
| I1 Graph state | 表达跨实体交互 | 当前主要只有机器人图+扁平血栓 | 异构物理图是已有研究方向，不能仅凭图称新 | 旧robot GAT已存在；新异构图未实现 | 均未验证 | 低 | 高 | 文献与非图、同构图消融，robot数量变化 |
| I2 Latent coupled dynamics | 预测长期联合状态 | 当前前馈ensemble，无时序latent | Dreamer/TD-MPC/RSSM等需正式比较 | 未实现 | 均未验证 | 低 | 高 | 基础transition先通过预测gate |
| I3 Physics lysis | 保持质量和接触一致 | 当前固定消融规则；旧WM未约束 | PINN/physics-informed GNN已有广泛工作 | 未实现 | 均未验证 | 低 | 高 | 可依据文献的约束，w/o physics |
| I4 Hemodynamics | 消融后可控性预测 | 当前解析局部半径/流速，无压力重分流 | 需与流场代理/流固学习模型比较 | 局部真实状态可读；新预测未验证 | 均未验证 | 低 | 高 | flow/occlusion误差，多步和分布外 |
| I5 Model allocation | 比较未来资源分配 | 最近/均衡/学习allocator已有，WM规划未实现 | model-based multi-agent planning已有 | 未实现 | 均未验证 | 低 | 高 | greedy/RL/model planning公平对照 |
| I6 Hierarchical planner | 降低规划动作维度 | 独立高层allocator已有，无联合WM规划 | HRL/MPC组合不能直接称创新 | 部分旧模块存在 | 新方案均未验证 | 低 | 高 | w/o hierarchy与推理成本 |
| I7 Safe imagination | 减少碰撞/不可信预测 | MVE有ensemble阈值，无安全规划 | uncertainty/risk-sensitive MPC已有 | 旧不确定性拒绝存在 | 安全收益与校准未验证 | 低 | 高 | calibration、风险指标、w/o safety |

H1样本效率、H2数量泛化、H3分配、H4长期收益、H5复杂度：全部 `HYPOTHESIS — NOT VERIFIED`。

EXP_0014：实现并完成三 seed × 1M transitions 的 `adaptive_edge_gat` 对照。每个
通信边的相对位置/速度特征同时产生每头 attention bias 和 sigmoid value-message
gate；Direct Local 动作、观测、奖励与物理保持不变。prospective validation 相对
EXP_0005 的成功率 +3.33pp、质量清除 +3.32pp、wall contact -11.79pp、pair collision
-3.84pp。该结果支持“自适应通信门控值得继续研究”的工程/实验假设，但不等于已完成
顶会文献新颖性验证；仍需多次独立复现、gate/edge-feature 消融和文献核查。

EXP_0001已完成：3seed×1M transitions；原协议成功72.8571%±1.8898pp、清除92.7773%±0.6408pp，与归档指标和actor/critic张量一致；额外840回合开发验证成功74.6429%±0.3571pp。详见[BASELINE_REPORT](BASELINE_REPORT.md)。

EXP_0001分类：Engineering Improvement（可追溯基线与只读测量）。Algorithmic Improvement：无。Scientific Contribution：固定模拟器和本机条件下的描述性复现证据、跨设备及场景限制。Potential Novel Contribution：无新增主张。Needs Literature Verification：I1–I7全部。

F004揭示部分场景消融停滞与高wall contact并存，值得后续检验可控性/分配/接触因素；这不支持I1–I7的性能贡献，也不证明world model能修复。H1–H5的证据状态没有改变。

EXP_0002已完成：同权重840对CPU/GPU回合出现10个success翻转，GPU−CPU成功率+0.9524pp但清除−0.0414pp；84对同设备重复一致。此结果约束未来算法对照必须固定设备。新增完整geometry身份与未来互斥清单属于Engineering Improvement；Scientific Contribution限本机测量敏感性证据；Algorithmic Improvement无，Potential Novel Contribution无。I1–I7无新增性能或消融证据，H1–H5仍未验证。Literature verification required.

EXP16–20 v2（2026-09-27 15:07 完成）：arm 17（六帧历史 GRU 位移预测）在 2M 达 76.7%±4.4、3M 73.8%±3.3，为批次最佳；arm 16（CV 预测+adaptive-edge）67–71%；arm 18（八候选动作+解析风险评分）与 arm 20（+世界模型规划）崩溃至 0.7–4%，失败根源定位在 18 的控制器架构（每步 ~1500 次干预、coronary 场景 wall_rate 0.84），世界模型门槛三项全过且 20 的 model_trusted ~1499/1500 步仍与 18 同水平，排除世界模型为失败原因。85% 开发验证阈值未达到，负结果按协议保留。尾迹取证（同日）：61% 失败=最后一栓拖尾，机器人距存活栓 0.05–0.12 原地漂 227 步——收尾行为没学会，是训练密度与终止信号问题，非观测/物理问题。

EXP_0010–0013（2026-09-27 17:47 prospective validation 完成）：variable-N 训练 +5.24pp（方差减半）；粒子训练+评估 +7.4~+12.0pp（16 最佳 81.0%，训练/评估贡献未分离，confounded）；separated-init −18.10pp、每步重规划 connectivity allocator −9.76pp 为保留负结果。全部为 validation 证据，非 test 结论。

2026-10-03 EXP_TEACHER_BC: topology-aware Graph Transformer student distilled from the route+avoid+wait teacher, no route controller at inference. Diagnostic: 97.8% train / 91.0% held-out anatomies vs teacher 98.9 / 96.0 and pure RL 21–33 / 16–34. Category: algorithmic (imitation with topology graph); not yet validated on sealed test, single seed. DAgger round 1: null on completion.
