# Research roadmap

日期：2026-09-20。核心问题：图结构潜在世界模型能否学习机器人—血栓—血流联合动态，并通过未来轨迹规划改善协同溶栓？

当前进度：PROJECT_AUDIT、EXP_0001基线复现及EXP_0002评估协议核验已完成。证据见[BASELINE_REPORT](BASELINE_REPORT.md)与[EVALUATION_PROTOCOL_REPORT](EVALUATION_PROTOCOL_REPORT.md)。Phase 0审查通过，下一项为EXP_0003完整科研指标；尚未进入主要创新阶段。

## 范围与假设

初始工作PROJECT_AUDIT → baseline复现 → EXP_0001已完成；用户“继续”后推进EXP_0002并更新Phase 0 Review，按证据选择下一实验，不一次性跳过中间gate。H1样本效率、H2跨机器人数量泛化、H3 clot-flow分配、H4长期规划、H5分层降复杂度均为 `HYPOTHESIS — NOT VERIFIED`。

| Phase | 目标/实验 | 进入下一阶段的最低证据 |
|---|---|---|
| 0 | 审计、EXP_0001原基线复现；EXP_0002评估协议验证 | 代码/配置/日志可追溯，短训练与正式训练无NaN，checkpoint可载入，复算指标一致；未解决问题显式保留 |
| 1 | EXP_0003完整科研仪表；EXP_0004正式多seed统一对比；EXP_0005泛化基线 | 指标定义和trace一致，train/val/test清晰，统计单位为训练seed |
| 2 | 有意义的现有model-free对照 | MADDPG/MLP-MAPPO/GNN-MAPPO预算公平；SAC只有必要且可公平实现时加入 |
| 3 | EXP_0006简单transition model；0007一步；0008多步 | 独立预测motion/mass/flow/collision/reward，1/5/10/20/50步与persistence对照，validation gate通过 |
| 4 | 简单模型对照潜在动力学 | 证明潜变量的任务适用性、预测/控制收益及成本，不能只换模型名 |
| 5 | 显式机器人/血栓/血管图 | 同预算非图对照；不同机器人数量与未见几何测试 |
| 6 | 物理约束clot-flow预测 | 质量非负/单调、局部流速/半径一致；缺失shear/treatment只定义接口，不伪造标签 |
| 7 | EXP_0009/0010的早期model-assisted对照，最终imagined allocation/MPC | 与无规划、greedy、RL assignment同预算对比；模型预测gate先于性能宣称 |
| 8 | 高层世界模型规划 + 低层MARL | 高低层消融，长期分配、安全性、扩展性与成本分析 |

编号表示问题登记顺序；Phase 4–8不预先制造结果。已有world model仅作为需审计的旧实现，先做简单模型再决定是否复用。

## 固定科研规则

每实验先登记问题、唯一主要变量、config、seeds、训练预算、Git版本和gate；smoke先于正式训练。固定代码运行期间不改；失败/重试保留attempt。至少3训练seeds作为正式性能证据，报告mean±sample std，不以840回合替代3次训练的独立性。

每次只改变一个核心模块；配置修复、评估修复、物理修复分别登记，不与新算法合并。全部消融在相应模块真正实现后运行。每个阶段保留Review；负结果同样进入注册表。

## 数据划分计划

EXP_0001沿用100000开始的历史validation用于复现；900000开始的额外评估为development validation。EXP_0002建立了560/140/280个same-family训练/验证/测试实例及完整内容hash；测试未运行policy，清单尚未接入训练器。未来训练须显式消费/约束生成范围；EXP_0005另建未见拓扑集合。封存测试不参与阈值、模型或seed选择。不同flow/clot/robot泛化须单独标记环境改变。

## 仪表与消融清单

Phase 1应完整覆盖success、mass removal、首次接触/完成时间、episode length、robot-clot距离、assignment分布、pair collision、wall contact、flow exposure、flow velocity、总/逐clot质量、每clot活跃机器人、action magnitude、train/eval reward、时间/显存。EXP_0001只做复现所需与只读轨迹诊断，不能宣称完成所有安全指标。

模型实现后依次注册 w/o graph、w/o latent、w/o physics、w/o planning、w/o hierarchical；只有在其余变量相同时才归因。预测误差图未有模型实验前标N/A，不生成虚构曲线。
