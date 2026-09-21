# Innovation tracker

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

EXP_0001已完成：3seed×1M transitions；原协议成功72.8571%±1.8898pp、清除92.7773%±0.6408pp，与归档指标和actor/critic张量一致；额外840回合开发验证成功74.6429%±0.3571pp。详见[BASELINE_REPORT](BASELINE_REPORT.md)。

EXP_0001分类：Engineering Improvement（可追溯基线与只读测量）。Algorithmic Improvement：无。Scientific Contribution：固定模拟器和本机条件下的描述性复现证据、跨设备及场景限制。Potential Novel Contribution：无新增主张。Needs Literature Verification：I1–I7全部。

F004揭示部分场景消融停滞与高wall contact并存，值得后续检验可控性/分配/接触因素；这不支持I1–I7的性能贡献，也不证明world model能修复。H1–H5的证据状态没有改变。

EXP_0002已完成：同权重840对CPU/GPU回合出现10个success翻转，GPU−CPU成功率+0.9524pp但清除−0.0414pp；84对同设备重复一致。此结果约束未来算法对照必须固定设备。新增完整geometry身份与未来互斥清单属于Engineering Improvement；Scientific Contribution限本机测量敏感性证据；Algorithmic Improvement无，Potential Novel Contribution无。I1–I7无新增性能或消融证据，H1–H5仍未验证。Literature verification required.
