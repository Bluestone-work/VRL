# Results summary

Frozen code: `8c159ab153e2f651a5cbdfa7c22706c86dc203d8`.

结果由 scripts/research_report.py 从不可替换的原始记录计算；三训练种子，标准差ddof=1。
主比较是相同历史validation协议，额外development validation不是封存test，也不代表未见拓扑。

| 数据集/运行 | Success mean ± SD | Clot removal mean ± SD | Episode length mean ± SD | Wall contact/robot-step mean ± SD |
|---|---:|---:|---:|---:|
| archived_validation | 72.8571 ± 1.8898% | 92.7773 ± 0.6408% | 155.7952 ± 5.7866 | 34.9418 ± 1.8215% |
| reproduced_validation | 72.8571 ± 1.8898% | 92.7773 ± 0.6408% | 155.7952 ± 5.7866 | 34.9418 ± 1.8215% |
| development_validation | 74.6429 ± 0.3571% | 94.9299 ± 0.3601% | 152.3857 ± 2.4573 | 34.7250 ± 0.8269% |


EXP_0001工程复现gate：PASS。详见[BASELINE_REPORT](BASELINE_REPORT.md)。

世界模型、planning、unseen-topology测试均未开展，H1–H5仍为HYPOTHESIS — NOT VERIFIED。

## EXP_0002

| Device | Success | Mass removal | Episode length | Wall / robot-step | Collision / pair-step |
|---|---:|---:|---:|---:|---:|
| cpu | 74.6429 ± 0.3571% | 94.9299 ± 0.3601% | 152.3857 ± 2.4573 | 34.7250 ± 0.8269% | 8.1387 ± 0.1522% |
| cuda:0 | 75.5952 ± 0.7435% | 94.8884 ± 0.2695% | 150.5321 ± 2.7320 | 34.3926 ± 0.7891% | 7.9576 ± 0.1663% |

设备配对success差+0.9524pp，翻转10/840。同设备重复与CPU父接口检查通过；没有算法变化。详见[EVALUATION_PROTOCOL_REPORT](EVALUATION_PROTOCOL_REPORT.md)。

## EXP_0010–0013（2026-09-27 prospective validation 完成）

协议：EXP_0005 协议、EXP_0002 封存 validation manifest（140 records）、确定性策略、每臂 3 训练 seeds × 140 episodes；test split 未打开。粒子/separated 臂评估条件与其训练条件一致（identity 断言按预登记放宽）。

|臂|Success|Removal|Steps|Wall|Pair|
|---|---:|---:|---:|---:|---:|
|0010 control|69.0%±11.8|88.2%|156|0.549|0.107|
|0010 variable-N|74.3%±6.3|90.6%|147|0.537|0.108|
|0011 separated-init|51.0%±9.3|78.8%|219|0.730|0.047|
|0012 particles-8|76.4%±10.1|91.5%|141|0.561|0.090|
|0012 particles-16|81.0%±5.1|92.7%|133|0.501|0.078|
|0012 particles-32|78.3%±8.1|92.3%|135|0.585|0.106|
|0013 connectivity alloc|59.3%±8.1|89.4%|194|0.591|0.089|

配对翻转（vs control，420 回合）：variable-N +50/−28；separated-init +27/−103；particles 8/16/32 = +56/−25、+66/−16、+68/−29；connectivity +46/−87。
正结果（variable-N、粒子训练）为 validation 证据且粒子增益的训练/评估贡献未分离；separated-init 与每步重规划分配为保留负结果。详见 RESEARCH_LOG。

## EXP16–20 v2（2026-09-27 15:07 完成，dev validation v2 分布：全部含 24 粒子）

|arm|1M|2M|3M|
|---|---:|---:|---:|
|16 adaptive-edge+CV 预测|70.0%±10.6|71.0%±10.6|66.9%±14.2|
|17 GRU 位移预测|72.6%±6.6|**76.7%±4.4**|73.8%±3.3|
|18 8 候选动作+解析风险|0.7%±1.2|2.1%±3.7|4.0%±7.0|
|20 18+世界模型规划|0.7%±1.2|1.0%±1.6|1.2%±1.5|

arm 17 最佳 76.7%（2M），arm 18/20 架构性崩溃（~1%），世界模型门槛全过但未救回 18。85% 开发验证阈值无任何 arm 达到，负结果按协议保留。两批（EXP_0010–13 vs EXP16–20 v2）评估分布不同，数字不可直接比较。

## EXP_0012 去混淆（2026-09-27 18:42 完成）

同一封存 manifest、同一无粒子 control checkpoint，仅评估时注入粒子：

|条件|Success|配对翻转(420)|
|---|---:|---|
|仅评估带粒子-8|69.8%±12.2|+4/−1|
|仅评估带粒子-16|69.3%±11.1|+4/−3|
|仅评估带粒子-32|69.5%±10.7|+8/−6|

按 seed 配对分解：评估效应 +0.2~+0.7pp（噪声内）；**训练效应 +6.7/+11.7/+8.8pp**——粒子臂增益是训练时域随机化的作用，非评估伪影。混淆解除。详见 RESEARCH_LOG 与 `research/runs/EXP_0012_PARTICLES_deconfound/deconfound_aggregate.json`。

## 跨批次对照汇总（2026-09-27）

两批（EXP_0010–0013 封存 manifest vs EXP16–20 v2 dev validation）的对照报告：`research/CROSS_BATCH_REPORT.md`。绝对数字不可直比（评估分布不同）；可比的是配对结论与场景难度结构（femoropopliteal_pad 两批皆最难）。EXP_0021 收尾修补预登记：`research/experiments/EXP_0021_TAIL_REPAIR.md`（三选一主变量，未启动，须单独授权）。

## EXP_0021 收尾修补（2026-09-28 13:14 完成，dev validation v2，含修复后重评估）

主变量：no-progress 早截断 K=50（用户授权选项 b）。底座两臂共用：variable-N 8 槽/5 实际 + 粒子-16 训练 + GRU 六帧预测。

|ckpt|base（底座自身）|repair（底座+K=50）|
|---|---:|---:|
|1M|68.3%±4.7|66.9%±8.0|
|2M|70.7%±6.8|**77.1%±2.6**|
|3M|**73.1%±4.4**|76.9%±2.7|

配对 repair−base：3M **+3.81pp**（逐 seed +5.7/+3.6/+2.1 全同向，t=3.67，单侧 p≈0.034）；2M +6.43pp（t=2.16，p≈0.082）。拖尾指标（失败回合尾部死时间）**189 步 → 49 步（−74%）**，回合长度 152.5 → 97.9（−36%）。场景增益：coronary_lm 83→100%、rca 43→73%、femoropopliteal 17→30%。判定（按预登记）：修补有效，**正结果**。85% 参考阈值未达（非本实验目标）。

注：首轮 18 个评估因 exp21_worker 评估路径动作语义 bug 全部无效，已隔离（`eval_invalid_action_semantics_20260928/`），上表为修复后重评估（checkpoint 未动，训练路径始终正确）。底座 73.1% vs arm 17（76.7%，不同底座组合）为待查项，不影响配对结论。数据：`research/runs/EXP_0021_TAIL_REPAIR_20260927a/aggregate.json`。

## EXP_0022 原尺寸MCA生理单位与可达性工程验证（2026-09-28）

**不属于RL成功率对比：未训练，成功率未评估，训练门槛尚未通过。**

补齐真实毫米缩放系数、压力驱动树流场及单位校验；保留旧物理协议。
1,296个流场质量守恒、健康流量及闭塞/再通检查通过，最大守恒残差4.55e-13 mm³/s。
全套286 passed、1 skipped；新增MCA测试42项。

名义3 mm MCA、146 mL/min对应截面平均344.25 mm/s。参考训练速度上端1 mm/s、
假设半径.08 mm时，乐观近壁逆流余量−70.48 mm/s；提示再通后逆流返回不能默认可行，
不能靠降低血流cap或继续堆训练步数宣称达到85%。所有尺寸/速度组合和不利结果留档。

下一阶段缺项：物理积分/出口/粒子接入，接触与溶解每秒速率、出口阻抗及硬件执行约束标定。
详情：`research/experiments/EXP_0022_MCA_PHYSIOLOGY.md`；
数据：`research/validation/EXP0022_MCA_20260928_final/summary.json`。

## EXP_0022B — 2026-09-28 数值修复

新MCA环境数值gate通过，纯RL direct-local接口冒烟通过。全套329 passed / 1 skipped；短时最大误差0.000755281 mm、完整1秒固定流场三档步长最大误差0.002475805 mm（阈值0.05 mm）。无正式训练或新成功率；不与旧环境成绩混用。详情见[修复报告](experiments/EXP_0022B_NUMERICAL_REPAIR.md)。
