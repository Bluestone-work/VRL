
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



## 实现结构与验收资料

![实际实现的预测辅助 DRL 网络](/home/wj/桌面/vascular_marl_local.tar./research/figures/EXP0061_20261006/predictive_network.png)

原始训练与评估保留在各 round 目录；模型门的预测来自训练好的网络，不是查询仿真真值。

## 固定行为轨迹上的预测审计

额外开发场景采用固定行为生成轨迹，各模型观察相同初始化；这些审计不进入自动调参规则。风险分数尚未校准，必须同时看阳性样本数量和零风险预测基线。

- round_00：下一测量 MSE 0.2786，持久性基线 0.2799；奖励 MSE 0.0145，零奖励基线 0.0123；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.015881935438851608, 0.005831019067950296, 0.0004355652994415903]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_01：下一测量 MSE 0.2789，持久性基线 0.2799；奖励 MSE 0.0201，零奖励基线 0.0202；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01605548814196226, 0.006110782432286372, 0.0007536016085491048]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_02：下一测量 MSE 0.2839，持久性基线 0.2799；奖励 MSE 0.0338，零奖励基线 0.0322；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.016413394299891308, 0.005540034561344117, 1.4520486017828666e-05]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_03：下一测量 MSE 0.2799，持久性基线 0.2799；奖励 MSE 0.0325，零奖励基线 0.0322；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.016399098102071415, 0.0055619559924807395, 0.0020646124734304836]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_04：下一测量 MSE 0.2800，持久性基线 0.2799；奖励 MSE 0.0324，零奖励基线 0.0322；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01638298566415524, 0.0055259839013310896, 5.765825083363927e-05]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_05：下一测量 MSE 0.2783，持久性基线 0.2799；奖励 MSE 0.0322，零奖励基线 0.0322；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01627411291525067, 0.005523889962491071, 3.222613914896926e-05]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_06：下一测量 MSE 0.2783，持久性基线 0.2799；奖励 MSE 0.0440，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.016150108826394897, 0.0055394765700437255, 3.316004101455073e-05]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_07：下一测量 MSE 0.2789，持久性基线 0.2799；奖励 MSE 0.0446，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01600442192180473, 0.005669542481469171, 0.0002939271947871931]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_08：下一测量 MSE 0.2798，持久性基线 0.2799；奖励 MSE 0.0445，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.016382703019827402, 0.005547876808326903, 4.728762245286689e-05]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_09：下一测量 MSE 0.2803，持久性基线 0.2799；奖励 MSE 0.0446，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01639180081932605, 0.005534859498406184, 0.0008002842578144636]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_10：下一测量 MSE 0.2797，持久性基线 0.2799；奖励 MSE 0.0446，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.016371172084002326, 0.0055525409804294355, 4.0616672915472766e-05]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_11：下一测量 MSE 0.2788，持久性基线 0.2799；奖励 MSE 0.0445，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.016086059102445854, 0.005704545920780323, 0.00038438905784490027]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_12：下一测量 MSE 0.2791，持久性基线 0.2799；奖励 MSE 0.0428，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01753638950146488, 0.008880037434946362, 0.0037939363423242964]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_13：下一测量 MSE 0.2794，持久性基线 0.2799；奖励 MSE 0.0450，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.03047430553312594, 0.026417912134183673, 0.019819512863016267]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_14：下一测量 MSE 0.2793，持久性基线 0.2799；奖励 MSE 0.0448，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.01950987505320356, 0.010521725728928136, 0.005750660809630805]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_15：下一测量 MSE 0.2789，持久性基线 0.2799；奖励 MSE 0.0446，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.02534445611309187, 0.016872453847144437, 0.01261606599602956]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_16：下一测量 MSE 0.2790，持久性基线 0.2799；奖励 MSE 0.0448，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.024942780113629787, 0.01669052077459155, 0.012770758284633428]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_17：下一测量 MSE 0.2794，持久性基线 0.2799；奖励 MSE 0.0438，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.04462320956313341, 0.04749277907897239, 0.03721799984453082]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_18：下一测量 MSE 0.2795，持久性基线 0.2799；奖励 MSE 0.0452，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.043475440912793886, 0.045471922229094144, 0.03600613112038806]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_19：下一测量 MSE 0.2795，持久性基线 0.2799；奖励 MSE 0.0449，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.04130371530426442, 0.04205770930407833, 0.03312996956474688]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_20：下一测量 MSE 0.2789，持久性基线 0.2799；奖励 MSE 0.0443，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.023474739562398286, 0.014851326203960779, 0.010638154263893454]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_21：下一测量 MSE 0.2790，持久性基线 0.2799；奖励 MSE 0.0443，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.026640183838813117, 0.019066048462356603, 0.014982060353220484]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_22：下一测量 MSE 0.2792，持久性基线 0.2799；奖励 MSE 0.0444，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.0215182005648844, 0.013396370900951303, 0.009198256752456349]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。
- round_23：下一测量 MSE 0.2794，持久性基线 0.2799；奖励 MSE 0.0447，零奖励基线 0.0450；壁/粒子/间距阳性样本 [12.0, 4.0, 0.0]；Brier [0.03947859467177988, 0.039228376556817164, 0.03070972445061605]，零风险基线 [0.016666666666666666, 0.005555555555555556, 0.0]。

![在线模型诊断，不是任务成功率](/home/wj/桌面/vascular_marl_local.tar./research/figures/EXP0061_20261006/model_diagnostics.png)

![最近候选策略实际 GUI 回放，场景固定](/home/wj/桌面/vascular_marl_local.tar./research/figures/EXP0061_20261006/round_00/gui_10s.png)
