# EXP_0022B — 新MCA环境数值修复与纯RL接口复验

日期：2026-09-28。主目录 `vascular_marl_local.tar.`，分支 `research/direct-local-scaling`。
本轮为环境工程修复，沿用纯 RL direct-local；没有正式训练、旧权重迁移或新成功率结论。

## 修复内容与证据边界

1. 接手时的 `_project` 使用裁剪后的轴向坐标判断跨段，分支永远无法触发；修复前直管解析、世界坐标保持和高速出口测试均失败。跨段改用未裁剪坐标，排除当前管段并检查相邻两侧开口是否容得下机器人。
2. 弯管入口端帽中的个体向内运动时不再被误判为返回上游。判断同时使用运动方向；顺/逆流通行均有回归测试。
3. 积分采用管段内投影中点法，并以端点预计到达时刻限制子步。跨段采用单侧入射方向，避免把子管段流速提前用于母管段，也避免任意步长相位改变支路或出口。
4. 出入口判定改为基于壁面投影后的实际接受路径，修复擦壁个体越过出口后漏记离场、停在虚假端帽的错误。入口/出口 × 四档步长的8项回归全部通过。
5. 验证器比较所有控制时刻的位置、活跃掩码和出口编号，增加独立完整1秒固定流场输运审计及更细参考步长，保留全部尝试。

接手的工作目录与原失败 `attempt1` 的输运源码 hash 不同；不能把归档28–29 mm误差全部归因于接手时的单个 bug。修复前/后源码快照与hash均已保存。

## 最终验证

阈值始终为 **0.05 mm**。物理配置字段未改变，仅更新 `numerical_method` 描述；血流、机器人速度/尺寸、接触/溶解、默认空间分辨率0.1和成功定义不变。

|seed|0.1秒全动力学：0.05 vs 0.025最大误差 mm|1秒输运：三档 vs 0.0125最大误差 mm|1秒输运：0.025 vs 0.0125误差 mm|结果|
|---:|---:|---:|---:|---|
|42|0.000139250|0.000276632|0.000039367|PASS|
|43|0.000755281|0.002475805|0.000356708|PASS|
|44|0.000117916|0.000445182|0.000060833|PASS|

- 0.1秒全动力学复验：3/3通过；最大误差 0.000755281 mm，误差随细分下降，采样/活跃掩码/出口一致。
- 请求1秒的Gym回合检查通过，但回合会在约0.15秒因机器人全部离场终止；不将它冒称为完整1秒动态回合。
- 独立固定血栓/固定流场输运：3 seeds × 4分辨率，全部时间轴推进到1秒；0.1/0.05/0.025三档相对0.0125参考均通过，最大误差 0.002475805 mm；所有活跃状态与出口编号一致。
- 全套回归：**329 passed，1 skipped**；物理输运/接口专门测试43项通过。
- 纯RL桥接：新schema、新权重、6条transition、1次优化器更新的冒烟通过；不使用旧checkpoint，不保存策略，不据此报告成功率。

## 最终证据

- 短时全动力学：`research/validation/EXP0022B_DYNAMICS_20260928_attempt8/`
- Gym回合边界：`research/validation/EXP0022B_LONG_20260928_attempt4/`
- 完整1秒输运：`research/validation/EXP0022B_TRANSPORT_1S_20260928_attempt3/`
- 测试：`research/validation/EXP0022B_NUMERICAL_REPAIR_20260928/tests_full_final2.xml`
- 机械核验和源码：`research/validation/EXP0022B_NUMERICAL_REPAIR_20260928/FINAL_SUMMARY.json`、`source_before/`、`source_after/`

中间失败均保留：较早的中点法试验产生支路分歧；早期混合积分仍有短时不单调误差；首次完整1秒审计失败；第二次完整1秒审计在0.05分辨率暴露出口端帽漏记（约0.467 mm），不能因0.025已通过就宣称全面通过。第6次短时审计通过时仍有上游单元测试失败，也不作为最终版本。只以上述最终源码和全套检查作为本轮完成证据。

## 当前结论和下一步

**新环境在已验证的名义MCA、动作与时间窗内通过数值gate，纯RL接口可用；正式训练尚未启动。**
三seed共享名义几何，只随机粒子/动作，不代表跨解剖泛化。完整1秒审计固定血栓质量，不覆盖长时间溶解-流场反馈。
现有 `formal_training_ready=false` 保留。接下来应预登记新物理下的纯RL任务分布、初始化/时限/数据划分和可达性条件，继续验证长时间接触溶解反馈，并明确未标定机器人速度/尺寸、出口阻抗与独立执行器假设；不能混用EXP21旧物理成功率或直接启用旧权重。

## 复验命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 /home/wj/miniconda3/envs/v/bin/python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 /home/wj/miniconda3/envs/v/bin/python -m scripts.validate_mca_dynamics --out research/validation/EXP0022B_new_short_attempt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 /home/wj/miniconda3/envs/v/bin/python -m scripts.audit_mca_transport --out research/validation/EXP0022B_new_transport_attempt
```

输出目录必须不存在；禁止覆盖先前attempt。前两项物理验证和最新来源hash均已复核。
