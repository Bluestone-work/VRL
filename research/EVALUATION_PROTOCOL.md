# Evaluation protocol after EXP_0002

来源：[EXP_0002报告](EVALUATION_PROTOCOL_REPORT.md)。执行版本`c7830951f2674aa7899b25da2ff0d49d8e81d8ef`。本协议是后续实验必须显式采用的约定；没有静默修改旧训练/评估入口。

## Fixed evaluation conditions

- 未来正式算法比较固定`cuda:0`，在EXP_0002结果产生前已登记。设备与驱动/库版本同属实验配置；变更设备需单独对照，不把0.95pp设备差当算法增益。
- 同一对照使用相同episode清单、环境物理、observation/action/reward和checkpoint选择规则。EXP_0001/2均使用final_policy，不混入best_score或best checkpoint。
- 从训练原始config显式创建环境；检查checkpoint meta与observation/state维数、网络、机器人数量、接触/控制/奖励参数。另核对原始config、checkpoint和执行源码hash，以覆盖metadata未保存的参数。
- 明确default：robot_radius=0.0011、geometric36、geodesic contact、flow_speed=0.004、tube_radius=0.055、max_speed=0.018、contact_radius=0.035、legacy initialization、randomized clots、无curriculum。改变任一项必须新实验配置。
- 固定seeds与随机源、确定性policy mean、actor/critic eval模式及线程数；同设备重复与旧接口控制属于测量验收，不把有限回合相同外推为跨机器逐位确定性保证。

## Metrics and pairing

主指标为全部质量清零Success Rate，以及每回合1−remaining/initial。完成步数仅在成功回合统计；未接触/未完成为null，不填0。wall事件/(robots×steps)，pair collision事件/(pairs×steps)，先按回合计算，再按场景等权。保留原始事件数。

正式报告至少3训练seed，mean±sample SD(ddof=1)以训练seed为单位。配对差按相同training seed/scenario/episode计算。保存全部失败、成功翻转和每seed结果；推理时间包含载入策略接口中的critic/控制变换/传输，不能称纯actor成本。

内容身份分两层：geometry含有序点/半径/branch/连接/flow fractions及派生数组；initial state再含机器人、血栓和RNG。精确hash既不检查图同构，也不排除近似重复。轨迹不同长度只比较共同前缀，单独记录终止长度，不padding伪造尾部。

## Prospective split reservation

固定清单位置：`research/runs/EXP_0002/splits/`。

| Split | Instances | Policy evaluation in EXP_0002 | SHA256 of manifest |
|---|---:|---:|---|
| train.json | 560 | 0 | `8cf6b70203dda0f1197989d628a326e576ec7fc197d961715988f5b6dca75421` |
| validation.json | 140 | 0 | `f39d3a0e7bcbe4006bf4c1463692311e1a7ce74eec124049bfd62d9d44b50351` |
| test.json | 280 | 0 | `f5a044c57f6c256b0b696a1dab22e851a7fa17430c49f897baa8cdd6c7aab232` |

三者精确内容互斥，均属相同14类解剖场景。测试集仅构造初始状态/登记hash，没有策略结果，不用于选择模型或调参。EXP_0005再另立未见拓扑协议，不把本清单包装成拓扑外推。

未来训练器需要新增显式manifest消费与生成器范围约束，并在实验预登记中确认；当前旧trainer仍随机生成血管，尚不保证消费这些清单。接入前不能宣称严格train/validation/test训练隔离已完成。EXP_0001旧最终验证仍为validation，其70/280回合曾用于模型选择；不得改名test。

## Known limits

EXP_0002同设备84对重复一致、CPU父实验840回合一致、42个旧GPU接口控制通过，但跨CPU/GPU有10/840个success翻转。未来结果小幅变化必须在一致设备/协议下复核。历史保存训练树只是部分generation，完整历史训练与验证互斥性未获认证。

F005的generation0仅为未使用计数；保留原始清单，分析按正generation推导创建树覆盖。今后修改该工具必须记录新版本，不回写EXP_0002执行源码。H1–H5仍为`HYPOTHESIS — NOT VERIFIED`。
