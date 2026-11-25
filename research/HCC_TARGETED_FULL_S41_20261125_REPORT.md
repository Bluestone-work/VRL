# HCC 固定失败场景定向训练结果

## 训练设置

- 六个固定的 v5 失败场景，保持原始解剖、目标分配、动态障碍、入口工程流速和场景哈希。
- IPPO 从随机初始化训练 20,000 个物理步，60,672 个智能体步。
- 18 个完整语义动作，包含逆流、横向偏移和上游回切候选。
- 启用无效动作掩码、出口动作掩码、SettleGuard 安全对齐、中心线匹配和在线流速特征。
- 训练自评和冻结确定性复测均使用同一六个场景，结果只代表开发场景记忆能力。

## 结果

冻结策略在六个固定场景上导航成功率为 **0/6**。

| seed | 到达集群数 | 壁面接触 s | 动态颗粒碰撞事件 | 终止 |
|---:|---:|---:|---:|---|
| 2944100000 | 2/3 | 0.0 | 0 | time_limit |
| 2944100001 | 2/3 | 0.0 | 0 | time_limit |
| 2944100003 | 2/3 | 0.0 | 1 | time_limit |
| 2944100011 | 2/3 | 0.0 | 0 | time_limit |
| 2944100020 | 2/3 | 0.23 | 0 | time_limit |
| 2944100021 | 1/3 | 0.0 | 0 | time_limit |

训练日志中的单个成功批次不能替代冻结复测。缩减动作集合的同场景微调同样为 0/6。

## 解释

在旧工程流速下，参考连续轨迹证明六例可达。IPPO 仍没有稳定学习到“横向进入低速区、逆流、上游回切”这一连续动作序列。扩展动作集合改善了探索，但离稳定策略还差一步。

按 638.6 mL/min 健康门静脉入口流量的压力测试中，六例首先发生远端出口越界，尚无集群到靶。这是合成血管树的出口边界和分支流量假设暴露出的物理建模问题，不能归因于 IPPO。

## 可复现文件

- [入口与分支流场校准](/home/wj/桌面/vascular_marl_local.tar./research/runs/HCC_CALIBRATION_20261124.json)
- [主动速度可识别性扫描](/home/wj/桌面/vascular_marl_local.tar./research/runs/HCC_ACTIVE_SPEED_IDENTIFIABILITY_20261124.json)
- [冻结六例结果](/home/wj/桌面/vascular_marl_local.tar./research/runs/HCC_TARGETED_FULL_S41_20261125/RUN_COMPLETE.json)
- [训练配置](/home/wj/桌面/vascular_marl_local.tar./research/runs/HCC_TARGETED_FULL_S41_20261125/config.json)
