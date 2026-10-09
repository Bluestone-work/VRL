
# EXP0061 连续科研进展


更新时间：2026-10-06T09:28:09


本队列持续执行：实际训练 → 同场景评估 → 失败分型 → 登记下一轮。负结果不会被改写为成功。所有比较均为开发集探索，原密封测试不访问。


当前阶段：registered_24_round_budget_completed；已登记轮次上限 24。09:00 生成阶段快照，不因单个实验结束而停止队列。


## 已完成的上一轮诊断


EXP0060 seed0 的 480 更新模型对传统追踪：Safe 同为 5/12；清除 97.92% vs 100%；壁接触 22.54 s vs .502 s。该模型未被接受为优于传统方法。


## 实际学习方法


GRU 历史编码 + 三成员动作条件观测/风险预测器 + PPO。预测器学习下一次测量变化和训练回报/接触事件；actor 只见测量历史和网络预测，不读取候选动作的真实仿真后果。


每轮两个训练臂共享父检查点、追加预算、场景流和奖励。continued-PPO 是相同预算的 DRL 对照；world-PPO 加预测辅助和预测门。不是 DreamerV3 复现，也不是临床已验证的世界模型。


初始配置同时增加两臂的安全训练代价，因此相对旧模型提升不能全部归因于预测模型；只能用两臂差异检验其增量贡献。


## 每轮结果

| 轮次 | 方法 | 场景 | Safe % | 清除 % | AUC % | 壁接触 s | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | continued_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.637 | not_accepted_continue_research |
| 0 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.590 | not_accepted_continue_research |




轮 0 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon




| 1 | continued_ppo | 12 | 41.7 | 97.9 | 89.78 | 22.585 | not_accepted_continue_research |
| 1 | world_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.637 | not_accepted_continue_research |




轮 1 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; compare another initialization after repeated failure; both arms share it




| 2 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 2 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 2 下一轮依据：predictive MSE worse than persistence: reduce auxiliary interference and increase warmup




| 3 | continued_ppo | 12 | 58.3 | 97.9 | 87.77 | 21.772 | not_accepted_continue_research |
| 3 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 3 下一轮依据：no robust gain: broaden policy exploration




| 4 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 4 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 4 下一轮依据：no robust gain: broaden policy exploration; compare another initialization after repeated failure; both arms share it




| 5 | continued_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.598 | not_accepted_continue_research |
| 5 | world_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.598 | not_accepted_continue_research |




轮 5 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon




| 6 | continued_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.588 | not_accepted_continue_research |
| 6 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.584 | not_accepted_continue_research |




轮 6 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; reduce update size after bounded safety coefficients stop changing




| 7 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.609 | not_accepted_continue_research |
| 7 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |




轮 7 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; compare another initialization after repeated failure; both arms share it




| 8 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 8 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 8 下一轮依据：no robust gain: broaden policy exploration




| 9 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 9 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 9 下一轮依据：predictive MSE worse than persistence: reduce auxiliary interference and increase warmup




| 10 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 10 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 10 下一轮依据：wall contact regression: increase safety cost equally in both arms; compare another initialization after repeated failure; both arms share it




| 11 | continued_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.609 | not_accepted_continue_research |
| 11 | world_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.598 | not_accepted_continue_research |




轮 11 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; reduce update size after bounded safety coefficients stop changing




| 12 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |
| 12 | world_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.609 | not_accepted_continue_research |




轮 12 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; reduce update size after bounded safety coefficients stop changing




| 13 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.608 | not_accepted_continue_research |
| 13 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |




轮 13 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; compare another initialization after repeated failure; both arms share it




| 14 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 14 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 14 下一轮依据：wall contact regression: increase safety cost equally in both arms; reduce update size after bounded safety coefficients stop changing




| 15 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 15 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 15 下一轮依据：wall contact regression: increase safety cost equally in both arms; reduce update size after bounded safety coefficients stop changing; bounded-parameter replication on a new registered training stream; not a new mechanism




| 16 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 16 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 16 下一轮依据：wall contact regression: increase safety cost equally in both arms; compare another initialization after repeated failure; both arms share it




| 17 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |
| 17 | world_ppo | 12 | 41.7 | 97.9 | 89.79 | 22.608 | not_accepted_continue_research |




轮 17 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; reduce update size after bounded safety coefficients stop changing; bounded-parameter replication on a new registered training stream; not a new mechanism




| 18 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |
| 18 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |




轮 18 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; reduce update size after bounded safety coefficients stop changing; bounded-parameter replication on a new registered training stream; not a new mechanism




| 19 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |
| 19 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |




轮 19 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; compare another initialization after repeated failure; both arms share it




| 20 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 20 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 20 下一轮依据：wall contact regression: increase safety cost equally in both arms; reduce update size after bounded safety coefficients stop changing; bounded-parameter replication on a new registered training stream; not a new mechanism




| 21 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 21 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 21 下一轮依据：wall contact regression: increase safety cost equally in both arms; reduce update size after bounded safety coefficients stop changing; bounded-parameter replication on a new registered training stream; not a new mechanism




| 22 | continued_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |
| 22 | world_ppo | 12 | 41.7 | 100.0 | 91.57 | 0.502 | not_accepted_continue_research |




轮 22 下一轮依据：wall contact regression: increase safety cost equally in both arms; compare another initialization after repeated failure; both arms share it




| 23 | continued_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |
| 23 | world_ppo | 12 | 41.7 | 97.9 | 89.80 | 22.588 | not_accepted_continue_research |




轮 23 下一轮依据：wall contact regression: increase safety cost equally in both arms; particle events: strengthen contact cost in both arms; incomplete clearance: longer reward horizon; reduce update size after bounded safety coefficients stop changing; bounded-parameter replication on a new registered training stream; not a new mechanism








## 解释边界


promising 只表示值得扩展验证，不能称稳定超过启发式。需要强固定选项对照、多种子与额外场景复核。单次 seed0 筛查、重复使用开发集和多轮选择存在选择偏差。


额外场景也标注为开发复核，不冒充密封测试。移除预测门的评估属于依赖性干预，不能代替同预算重训练消融。


真实磁场尚未校准，2 mm 仅为间距代理。体外能破栓与学习导航可迁移是两个需要分别验证的结论。


失败/预算未完成的作业留在 jobs 与 STATUS.json；三个连续运行基础设施失败会停在明确失败状态，避免无意义重试。





状态与全部证据：/home/wj/桌面/vascular_marl_local.tar./research/validation/EXP0061_ADAPTIVE_WORLD_RL_20261006

