# Research log

## 2026-09-20 — EXP_0001 — preregistration

### 今天解决的问题
确定本地源码与历史实验的关系，分清当前残差GAT-MAPPO与旧图世界模型/MVE；完成静态PROJECT_AUDIT。
### 修改内容
新增research记录系统和独立实验配置；尚未修改核心算法、环境、reward或observation。
### 为什么这样修改
原目录是dirty worktree，历史验证包含重复回合；先建立可追溯基线再研究新方法。
### 实验结果
尚未运行正式EXP_0001。参考geodesic_v：72.8571%±1.8898%成功率，92.7773%清除率，来自3个已存在最终validation摘要。
### 相对 baseline 的变化
计划主要变量：独立重新运行；算法与任务不变。尚无新性能变化可报告。
### 出现的问题
见PROJECT_AUDIT A01–A12；旧world model test用于gate、动作语义、占位预训练等需后续单独处理。
### 当前解释
HYPOTHESIS — NOT VERIFIED：当前完整源码和固定依赖能重现同量级的基线结果。
### 创新性影响
Engineering Improvement：记录与复现。没有新的Algorithmic Improvement或Scientific Contribution。
### 下一步
冻结独立Git工作树、依赖快照，运行全测试和短训练；通过后执行3 seeds×1M，不按结果调参或丢seed。

## 2026-09-20 — EXP_0001 — frozen source and checks

### 今天解决的问题
将132个源码/配置/记录文件冻结到独立提交 `8c159ab153e2f651a5cbdfa7c22706c86dc203d8`；原用户分支及index未被切换。
### 修改内容
新增执行器、只读评估器、梯度健康观察器和两个协议回归测试。梯度观察器只读取原裁剪函数返回的norm，不更换更新规则。
### 为什么这样修改
原训练日志未记录梯度健康，评估未完整继承环境配置；复现需在不改变核心计算的条件下核查。
### 实验结果
冻结源码完整测试：171 passed、1 skipped，8.82秒。stdout/stderr及JUnit保存在runs/EXP_0001/checks；16,384-transition smoke正在执行。
### 相对 baseline 的变化
性能尚未测得；仅新增工程审计和记录。核心算法、环境和reward保持审计时版本。
### 出现的问题
测试未出现本轮native crash；这不能证明历史稳定性问题永久消失。
### 当前解释
静态与单元级正确性初步通过；仍需要短训练和全预算验证。
### 创新性影响
Engineering Improvement；无算法或新颖性结论。
### 下一步
检查smoke的loss、梯度、参数变化、checkpoint载入和指标；通过后启动正式三seed。

## 2026-09-20 — EXP_0001 — smoke passed and formal launch

### 今天解决的问题
完成16,384-transition短训练及checkpoint健康检查，纠正外层coordinator的模块搜索路径。
### 修改内容
没有改冻结源码；coordinator启动时显式设置PYTHONPATH。分析脚本在运行前修正语法，见F002。
### 为什么这样修改
原checkpoint含环境对象，反序列化需要能导入对应模块；详见FAILURE_LOG F001。
### 实验结果
smoke训练38.2094s，两次PPO update、420次梯度裁剪观察均有限；actor实际变化、checkpoint可载入，14回合最终validation成功42.8571%、清除68.8687%，pre-update ratio最大误差1.19e-6。此短训练不用于正式性能结论。
### 相对 baseline 的变化
正式结果尚未产生；不拿短训练与1M基线比较。42/43分别在GPU0/1启动，44在GPU0固定队列等待42完成。
### 出现的问题
保留F001载入失败及恢复记录；没有重新训练smoke或更换seed。
### 当前解释
smoke gate通过，允许按原预登记预算正式运行。
### 创新性影响
Engineering Improvement；无新算法或世界模型证据。
### 下一步
监测三seed训练，完成后运行固定900000起点development validation、原始指标复算与图表。

## 2026-09-20 — EXP_0001 — replay verification during training

### 今天解决的问题
核查新增只读诊断器是否改变episode结果，并检查checkpoint跨设备回放。
### 修改内容
新增独立分析用回放脚本，不改冻结训练或评估实现。
### 为什么这样修改
CPU回放与原GPU轨迹存在差异，需要先核实，不能忽略异常。
### 实验结果
同设备cuda:0回放14回合：success/steps/wall total/removal全部精确一致。CPU回放success仍全部一致，但个别轨迹不同，最大removal差2.9065pp。原始对照见checks/diagnostic_replay*.json。
### 相对 baseline 的变化
这是smoke权重的测量一致性检查，不是算法增益；正式42/43均已超过300k，当前不提前判定最终效果。
### 出现的问题
跨CPU/GPU逐位重现不保证，记录为F003。
### 当前解释
同设备复现支持诊断器不改变所检验回合的动力学；浮点差异导致轨迹分叉的具体机制仍是HYPOTHESIS — NOT VERIFIED。
### 创新性影响
Engineering Improvement和复现限制证据，无新颖性主张。
### 下一步
固定正式协议和设备，完成既定训练与开发验证，不因中途结果调参。

## 2026-09-20 — EXP_0001 — seeds 42 and 43 complete

### 今天解决的问题
完成前两次1M-transition从头训练与各280回合最终验证；第三seed44按固定队列启动。
### 修改内容
只更新科研记录，冻结执行代码未改变。
### 为什么这样修改
持续公开全部seed状态，避免仅在最终结果后补记录。
### 实验结果
seed42成功72.142857%、清除92.218303%，训练921.2944s；seed43成功75.000000%、清除93.476733%，训练917.2398s。每个123次update、25,640次梯度检查均有限，actor变化且checkpoint可载入。
### 相对 baseline 的变化
两seed的success/removal与各自历史final validation精确一致，变化0pp；暂不形成三seed均值结论。训练时间不可仅凭不同并发与观察器成本归因。
### 出现的问题
无新的训练错误；跨设备回放差异仍保留为F003，未被精确GPU复现结果掩盖。
### 当前解释
结果支持本机冻结代码的复现能力，不支持世界模型收益或算法提升。
### 创新性影响
Engineering Improvement：可审计基线；无新算法创新。
### 下一步
完成44及三seed开发验证，自动生成图表、原始文件hash和Phase0 Review。

## 2026-09-20 — EXP_0001 — all training complete; development evaluation started

### 今天解决的问题
完成全部3个预登记训练seed；开始固定的840回合额外开发验证。
### 修改内容
新增独立分析用artifact核查脚本，对冻结代码hash和历史/本轮actor、critic张量作只读比较；未改训练或评估源码。
### 为什么这样修改
精确复现指标需要可核查证据，后续分析代码与冻结执行代码分别归档。
### 实验结果
seed44最终成功71.428571%、清除92.637008%；3seed原协议成功均值72.857143%、样本标准差1.889822个百分点，平均清除92.777348%。三个训练均无失败重试；每seed123次update、25,640次梯度检查。
### 相对 baseline 的变化
三个seed的success、removal与历史对应结果一致，主成功率差0pp；未改变算法，不能归因为算法改进。
### 出现的问题
开发验证使用CPU；与原GPU验证的差异不能只归因于新场景，设备影响F003仍需保留。
### 当前解释
本机原协议复现支持工程基线可靠性；开发验证和最后报告尚待完成。
### 创新性影响
Engineering Improvement；H1–H5仍为HYPOTHESIS — NOT VERIFIED。
### 下一步
完成840回合，按预登记指标复算，生成5张图和Phase0 Review，归档结果版本。

## 2026-09-20 — EXP_0001 — completed analysis and Phase 0 gate

### 今天解决的问题
完成原基线3seed从头训练、840回合原协议最终验证和另840回合开发验证；生成报告、原始指标表、5张图及Phase0 Review。
### 修改内容
仅新增执行/测量/分析脚本、两项协议测试和研究记录。冻结132文件完整性检查通过，原目录审计时86个Python文件未变；全部旧checkpoint原文件hash保持一致。

图表审阅后将安全箱线图的离群点与均值显示出来，并明确流速曲线采样在机器人位置、逐血栓曲线的编号；这些为呈现修正，不改原始数据、指标或实验执行。
### 为什么这样修改
先建立可信基线和可追溯证据，防止后续把配置差异、设备差异或遗漏seed误当算法贡献。
### 实验结果
原协议成功72.8571%±1.8898pp、清除92.7773%±0.6408pp、长度155.7952±5.7866步、wall contact/robot-step 34.9418%±1.8215pp。开发验证成功74.6429%±0.3571pp、清除94.9299%±0.3601pp、长度152.3857±2.4573步、wall contact 34.7250%±0.8269pp、pair collision/pair-step 8.1387%±0.1522pp。均为三个训练seed的均值±样本标准差。

全测试171 passed、1 skipped；smoke通过；3个正式训练均完成1M transitions、123次update、25,640次有限梯度检查，无训练重试。全部actor/critic张量与各自历史模型精确相等。CPU完整推理接口0.6631±0.0069ms/step；不是纯actor时间。
### 相对 baseline 的变化
同协议success、removal、长度和wall变化均为0，绝对成功率差0pp、相对变化0%。开发集分数高于原验证集，但场景seed和设备不同，不能计作改进。训练时间总和从2603.43s至2740.71s，增加137.28s、5.27%；并发/负载/梯度观察未单独控制，不能归因。三次耗时总和不是并行wall-clock。没有新算法对照，sample efficiency收益N/A。
### 出现的问题
F001–F003工程/跨设备限制保留。开发验证213/840回合失败；股腘动脉成功11.67%，wall contact80.26%。固定失败回合清除98.37%但最后50步消融0，登记F004。原协议仍复用训练validation；额外开发集不是封存unseen test。
### 当前解释
工程复现假设达到预登记验收标准；没有超出variance的算法增益主张。原物理、观测、奖励、网络和控制器未变。F004提示应独立检查可控性、分配和接触，而非直接断言图世界模型有效。相关因果解释为HYPOTHESIS — NOT VERIFIED。
### 创新性影响
Engineering Improvement：记录、复现、只读仪表。Algorithmic Improvement：无。Scientific Contribution：限定于当前仿真的描述性复现与限制证据。Potential Novel Contribution：无新增主张。Needs Literature Verification：I1–I7；H1–H5未验证。
### 下一步
EXP_0002核验评估配置、设备一致性及geometry内容hash划分，之后EXP_0003完善科研指标。通过协议与预测gate前不进入主要WM规划创新；全部失败和负结果留存，结果版本单独归档。

## 2026-09-20 — EXP_0002 — preregistration

### 今天解决的问题
根据用户“继续”，启动下一项评估协议核验；已读Phase0 Review和环境/训练/载入/诊断/数据划分实现。
### 修改内容
先登记EXP_0002与独立JSON配置。唯一性能比较变量为CPU/GPU推理设备；不重新训练、不改策略物理。
### 为什么这样修改
EXP_0001 F003显示跨设备轨迹可能分歧；旧checkpoint metadata未完整覆盖环境参数，geometry hash未包含拓扑连接与flow_fraction，历史训练数据清单也不完整。
### 实验结果
尚未运行EXP_0002。只读检查确认训练checkpoint保存了当前树及generation，可恢复部分历史几何；未保存的generation不能据此重建。
### 相对 baseline 的变化
待测。同权重同回合配对1680回合，另做固定重复、smoke和历史接口控制；零调参budget。
### 出现的问题
不能将未来清单的互斥性反推成历史训练无泄漏。封存test只登记hash，不运行policy。
### 当前解释
HYPOTHESIS — NOT VERIFIED：同设备接口可复现；跨设备影响大小待测。
### 创新性影响
Engineering Improvement；H1–H5未验证，无算法/新颖性主张。
### 下一步
实现最小协议工具与回归测试，冻结版本，先测试和smoke，再正式配对评估。

## 2026-09-20 — EXP_0002 — source frozen

### 今天解决的问题
完成最小协议工具：完整geometry身份、初始状态身份、配置一致性、同设备重复、CPU父实验回放及训练快照覆盖检查。
### 修改内容
新增scripts/research_protocol.py、research_protocol_run.py和6项针对性回归测试。冻结143文件至`c7830951f2674aa7899b25da2ff0d49d8e81d8ef`；父实验54个输入文件hash保留。未改环境、策略或原执行脚本。
### 为什么这样修改
让设备成为唯一性能变量，同时通过独立控制检查测量代码本身是否改变结果。
### 实验结果
冻结前协议开发检查8 passed（6新增+2原协议）；正式冻结完整回归正在运行。
### 相对 baseline 的变化
尚无EXP_0002性能结果。
### 出现的问题
无新增执行错误；完整历史训练几何覆盖仍需实测盘点。
### 当前解释
先通过正确性gate，再运行smoke与正式回放；不因性能结果改代码。
### 创新性影响
Engineering Improvement，无H1–H5证据。
### 下一步
完整回归、56回合smoke及固定正式评估；分析代码与最终记录在独立结果commit归档。

## 2026-09-20 — EXP_0002 — checks and smoke passed; formal launched

### 今天解决的问题
冻结完整回归177 passed、1 skipped（8.77s）；完成56个smoke回放（14场景×2设备×原回合/重复）。
### 修改内容
仅更新记录；执行源码保持c783095不变。
### 为什么这样修改
先确认重复/父实验一致性，再允许正式评估。
### 实验结果
两设备各14对smoke重复轨迹精确一致；CPU的14回合与EXP_0001父实验非时间指标和trace数组精确相同。CPU任务9.90s、GPU任务11.38s，含载入/记录等，不用短任务比较性能。
### 相对 baseline 的变化
当前只证明接口在所检验smoke回合中未改变行为；正式840个设备配对待测。
### 出现的问题
源码复核发现geometry_generation从1开始；清单工具的missing_generations_through_last_saved字段按0..max枚举，因此0是未使用的计数值，不能当作遗失的实际树。分析将从原始observed_generations按1..max重新计算真实覆盖，并明确该字段边界，不修改冻结执行代码或原始清单。
### 当前解释
测量gate通过。完整历史几何覆盖仍不可预先认定。
### 创新性影响
Engineering Improvement；无算法或世界模型证据。
### 下一步
固定预算执行两个设备各840回合、84次重复、42个历史控制，再生成未来split与结果分析。

## 2026-09-20 — EXP_0002 — completed paired evaluation and split audit

### 今天解决的问题
完成1680正式回合（840个CPU/GPU配对）、84次同设备重复、42次旧GPU接口控制以及未来geometry清单；含先前56回合smoke总计1862次回放，无重试。
### 修改内容
新增独立分析/归档工具与3张图；完成EVALUATION_PROTOCOL及报告，更新注册表、创新跟踪与Phase0 Review。执行源码仍为c783095，核心环境/策略与EXP_0001一致。
### 为什么这样修改
将设备差异和评估来源与算法差异分离，并给未来train/validation/test隔离提供可执行的实例身份约定。
### 实验结果
CPU成功74.6429%±0.3571pp、清除94.9299%±0.3601pp、长度152.3857±2.4573步、wall34.7250%±0.8269pp、pair collision8.1387%±0.1522pp。GPU成功75.5952%±0.7435pp、清除94.8884%±0.2695pp、长度150.5321±2.7320步、wall34.3926%±0.7891pp、pair collision7.9576%±0.1663pp。均为3训练seed样本标准差。

CPU父实验840回合指标和trace数组一致；84对同设备重复一致；42个历史接口控制通过。CPU/GPU全部trace数组完全一致0/840，机器人位置出现差异838/840；成功翻转10（仅CPU成功1、仅GPU成功9）。首次执行动作最大绝对差2.38418579e-7，共同前缀机器人位置最大L2差0.903776646模拟单位。

未来split为560/140/280实例，内部及互相完整hash均无重复；test策略执行0。旧final validation的70/280回合（25%）曾用于模型选择。保存checkpoint提供462条树记录、420个独立hash，与所检验验证集合无交集；完整历史不可认证。
### 相对 baseline 的变化
同设备CPU相对EXP_0001非时间指标变化0。GPU−CPU成功+0.9524pp（相对+1.2759%）、清除−0.0414pp、长度−1.8536步、wall−0.3324pp、pair collision−0.1811pp。它们是设备效应，不是算法增益。

完整策略接口CPU推理0.6668±0.0032ms/step，GPU1.0964±0.0025ms/step；本任务下GPU接口更慢，含传输/critic/控制成本且有并发，不作独立硬件因果结论。正式作业耗时总和CPU254.29s、GPU308.27s；不是并行wall-clock。Training Steps=0，训练成本/稳定性/样本效率改善N/A。
### 出现的问题
F005覆盖字段编号边界保留，分析以正generation重算。F006跨设备结局翻转；F007历史训练身份不完整。清单尚未接入训练器，不能把互斥清单等同于已强制隔离的训练流程。
### 当前解释
同设备测量可靠性得到支持；跨设备轨迹等价被本轮否定。首次动作不同可定位到设备相关policy/控制计算链，但具体浮点算子和接触/投影放大机制仍为HYPOTHESIS — NOT VERIFIED。新问题是控制数值敏感性与训练清单约束；未测world model收益。
### 创新性影响
Engineering Improvement：协议和身份校验。Algorithmic Improvement：无。Scientific Contribution：固定模拟器下的设备敏感性及历史证据边界。Potential Novel Contribution：无。Needs Literature Verification：I1–I7；H1–H5未验证。
### 下一步
Phase0 gate通过，进入EXP_0003的完整科研指标与测量正确性检查；后续独立接入split消费并注册训练对照。继续固定cuda:0，不按分数择设备，不运行封存test、不跳到WM规划。

## 2026-09-23 — EXP_0006–EXP_0009 — Direct Local scaling capabilities (preregistration and implementation)

### 今天解决的问题
在 Direct Local GAT-MAPPO（EXP_0005，不接 World Model、不回到 flow-guided residual controller）之上实现四项能力：可变数量智能体、分散初始化、connectivity-aware 任务分配、动态障碍物。全部默认关闭/兼容旧路径，不改变 reward、MAPPO 核心超参数与 direct-Frenet 动作语义。
### 修改内容
- `marl/gat_policy.py`、`marl/mappo_advanced.py`：GATLayer/GATEncoder/GATCritic 接受 `agent_mask`（padding 同时从 key 和 query 两侧被屏蔽，全 -inf 行经 nan_to_num 归零）；`MAPPOAdvanced(max_agents=...)` 记入 checkpoint meta；rollout buffer 存 bool mask；update 内对 rewards/dones/terminals/values/bootstrap 全部按 mask 置零，actor/critic/entropy 损失只对真实 slot 求均值，advantage 统计只在真实 slot 上计算，team 项除以 per-sample 真实数量（支持同一 batch 混合 N）。
- `environments/vector_env.py`：`active_robots <= num_robots` 填充；动作、pair 项、lysis 接触、wall 统计、reward 分摊、peer/crowding 特征、观测行与 adjacency 全部按 mask 屏蔽；Brownian 噪声只为 active slot 抽取，RNG 流不变（同种子与未填充 env bit 级一致）。另加 `initialization_mode="separated"`（血管图 geodesic FPS + 双距离约束 + 有界松弛阶梯）与 `dynamic_intravascular_particles` 开关。
- `environments/vascular_3d_marl_env.py`：separated 初始化模式（含 reset info 报告松弛等级与实际最小距离）；动态障碍物钩子与 per-step 记录。
- `environments/dynamic_particles.py`：blood-cell-inspired 障碍物——用与机器人相同的 tree.flow（含 occluded 半径覆盖）平流 + 少量随机漂移；每步投影回管腔；尺寸 = radius_ratio × robot_radius（无量纲）；专用 RNG。
- `marl/connectivity_allocator.py`：connectivity_aware 分配器（geodesic 距离、路径重叠、边拥堵、逆向流代价、目标切换惩罚、clot 容量软约束+有替代时硬约束），greedy 顺序求解；`nearest` 与 `flow_spread` baseline 同模块保留；`env.set_task_assignments` 只改目标不改动作。
- `scripts/train_vector_mappo.py`：`--active-robots/--max-agents/--initialization-mode/--dynamic-particles` 等新旗标；`scripts/run_scaling_smokes.sh` 六组 feature+control 冒烟。
- 新测试四份：test_variable_agents / test_separated_initialization / test_connectivity_allocator / test_dynamic_particles。
### 为什么这样修改
用户授权的四项改造全部落在"能力/机制"层，最小侵入：所有默认路径与 EXP_0001–EXP_0005 bit 级一致（专用 RNG、mask 旗标默认关），因此旧实验可复现；四项各自单独预注册为 EXP_0006–0009，不混合比较。
### 实验结果
- 全量 pytest：220 passed / 1 skipped（taskset 0-5,8-23；此前一次 rc139 为已知 core 6/7 不稳定，换核后通过）。
- 六组冒烟（448 transitions，CPU，均完成 PPO update 且有限）：variable_n(8 槽/5 实际/capacity10)、separated_init、dynamic_obstacles(16 粒子) 及各自对照，对照组 update 指标与历史 legacy 路径一致。
- variable-N 等价性：padding 前向对 3/5/8 真实 agent 与未填充输出一致（float32 舍入级）；env 级 20 步 bit 一致；混合 N batch (3/6/5/2) 训练有限。
- separated spawn：14 解剖场景 × 8 episode × 5 robots，112/112 reset 松弛等级 0；最小欧氏 0.0715（要求 0.0176）、最小测地占比 0.196（要求 0.12）。
- connectivity 分配（静态场景 8 robots）：mean 路径重叠 nearest 3.433 / flow_spread 2.856 / connectivity_aware 2.787；对 nearest 14/14 场景占优、对 flow_spread 8/14。
- 动态障碍物：位移与局部流方向平均 cos 0.994（孤立树测试 >0.8 断言）；开启后 env RNG 流不变；关闭时与 legacy bit 一致。
### 相对 baseline 的变化
无性能主张。机制层新增能力，所有数字都是等价性/约束满足/静态场景度量，不是训练增益。EXP_0005 的 direct-local 语义、reward、超参数完全未动。
### 出现的问题
- 测试驱动发现并修复两个真 bug：advantage 归一化曾把 padding 零计入统计（更新尺度被 padding 比例耦合）；vectorized reshape 分支曾丢失 ctx_all 中的 mask。
- `exactly-on-particle` 重合时分离脉冲方向未定义，已用确定性轴向兜底。
- 本机 core 6/7 不稳定（已知 F-类问题），全程 taskset 0-5,8-23。
### 当前解释
四项能力在机制层可信：等价性测试、约束满足测试、静态场景度量和冒烟都通过。跨 N 迁移、separated spawn 对成功率的影响、allocator 闭环效果、障碍物对碰撞率的影响都需要新的 1M-transition 级训练实验，本轮未获授权（长跑须单独确认）。
### 创新性影响
Engineering Improvement：mask 基础设施、FPS 初始化、allocator、障碍物模型。Algorithmic Improvement：无（无训练结果）。Scientific Contribution：静态场景下 corridor-overlap 度量的对比证据。Potential Novel Contribution：connectivity-aware 血管任务分配需文献核验（C²-Explorer 思想改编，非实现）。H1–H5 未验证状态不变。
### 下一步
如需性能结论，分别预注册 1M-transition 级训练实验（每项单独、seed 42/43/44、prospective 协议），先与用户确认授权边界。

## 2026-09-27 — EXP_0010–0013 — prospective validation complete

### 今天解决的问题
在 tmux vrl-exp0010-0013 于 2026-09-24 03:44 完成全部 26 个 1M-transition 训练后，用 EXP_0005 协议（EXP_0002 封存 validation manifest、确定性策略、逐回合 identity 断言、test split 未打开，policy_evaluations 保持 0）补完四个 scaling 能力的 prospective validation（每臂 3 seeds × 140 回合 = 420 episodes）。评估在 cuda:0 单卡、绑核 0-5,8-23 下断点续跑完成（17:12–17:47，进程 PID 471070 一次跑完，无 native crash）。
### 修改内容
- `scripts/prospective_eval_0010_0013.py`：EXP_0005 协议脚本扩展为 10 个臂；separated-init 与 particles 臂的 initial_state 断言按预登记理由放宽（spawn 规则/障碍状态改变初始状态 hash，geometry hash 与 active_clots/stations/branches 仍严格断言），其余臂严格断言全部通过。中断的 attempt1（basilar_vertebral/3090000 断言处按当时严格模式退出）已由续跑以正确放宽逻辑覆盖，旧 stdout 留在 driver log 中。
- 三个 control 臂共享同一组 checkpoint（SHA256 相同：b7eae8e7c201 / 9c26d989868e / 0b02ec4bbc5e），控制臂结果 69.0%±11.8 三次一致，证明评估器确定性。
### 实验结果
|臂|success（3-seed mean±SD）|清除|步数|wall|pair|
|---|---|---|---|---|---|
|0010 control（5 机器人）|69.0%±11.8|0.882|156|0.549|0.107|
|0010 variable-N（8 槽/5 实际）|74.3%±6.3|0.905|147|0.537|0.108|
|0011 separated-init|51.0%±9.3|0.788|219|0.730|0.047|
|0012 particles-8（训练+评估均含粒子）|76.4%±10.1|0.915|141|0.561|0.090|
|0012 particles-16|**81.0%±5.1**|0.927|133|0.501|0.078|
|0012 particles-32|78.3%±8.1|0.923|135|0.585|0.106|
|0013 connectivity_aware 每步重规划|59.3%±8.1|0.894|194|0.591|0.089|

逐回合配对翻转（vs 各自 control，420 回合）：variable-N +50/−28；separated-init +27/−103；particles-8 +56/−25、16 +66/−16、32 +68/−29；connectivity +46/−87。
### 相对 baseline 的变化
- variable-N：+5.24pp，且 seed 方差减半（11.8→6.3）。机制解释候选：8 槽 padding 训练的正则化效应，HYPOTHESIS — NOT VERIFIED。
- separated-init：−18.10pp 负结果。出生分散导致 5 机器人任务覆盖变差（coronary_lm 5→1、mca_m1 4→0、sma 3→0、femoropopliteal 5→1），步数 156→219。不是实现 bug：约束满足在 EXP_0007 已验证；是策略在分散开局下没学会协调收拢。
- particles 训练+评估：+7.4~+12.0pp，以 16 粒子最佳。注意混淆因素：粒子臂评估时也带粒子（initial state hash 不同属预期），且粒子使环境部分随机化（类似域随机化正则）。不能把该增益同时归因"训练时障碍"或"评估时障碍"——两者未分离，标注 confounded。
- connectivity_aware 每步重规划：−9.76pp 负结果。静态场景的 corridor-overlap 优势（EXP_0008）未转化为闭环收益；每步重规划的目标切换可能破坏策略学到的一致性（步数 156→194）。
### 出现的问题
- 臂间评估条件不完全一致（粒子臂带障碍评估、separated 臂初态不同），预登记时已声明并保留，不作为跨臂排名依据，只作各臂 vs 自身 control 的配对结论。
- EXP_0012 三档粒子臂的 control 均为同组无粒子 checkpoint，+pp 数字依赖 control 69.0% 的 11.8pp 离散度，n=3 不宣称显著。
### 当前解释
四能力中 variable-N 与动态粒子训练两项在 prospective validation 下有正向配对证据；separated-init 与每步重规划 allocator 两项负结果。全部为 validation 证据，非封存 test 结论。sealed manifest policy_evaluations=0 保持。
### 创新性影响
Engineering Improvement（评估断点续跑、identity 分级断言）。Algorithmic Improvement：variable-N padding 训练与粒子训练为候选，需消融确认（单独"仅评估带粒子"对照缺失）。Scientific Contribution：两项负结果（separated spawn、每步重规划分配）+ 一项混淆未分离的正结果。Potential Novel Contribution：无新增主张。
### 下一步
1. 粒子增益去混淆：补"无粒子训练 + 带粒子评估"臂（模型已有：0012 control checkpoint 直接在粒子评估条件下跑 420 回合即可，无需新训练）。
2. 结合拖尾取证（arm 17@2M 尾巴状态占比低、无终止信号）：预登记 EXP21 尾巴过采样/早截断/potential shaping 三选一。
3. EXP_0010–0013 与 EXP16–20 v2 的对照汇总报告（注明两批评估分布不同，不可直接比数字）。

## 2026-09-27 — EXP_0012 去混淆 — 完成

### 今天解决的问题
分离"训练时带粒子"与"评估时带粒子"对 EXP_0012 粒子臂 +7.4~+12.0pp 增益的贡献。用户授权后执行：`scripts/prospective_eval_0012_deconfound.py`，同一 EXP_0002 封存 manifest、同一确定性协议、同一 control checkpoint（无粒子训练），仅在评估时注入 8/16/32 粒子（与粒子臂评估条件一致，含相同 particle_seed 派生）。3 档 × 3 seeds × 140 回合 = 1,260 episodes，18:35–18:42 完成，无崩溃。
### 修改内容
仅新增去混淆评估脚本与结果目录 `research/runs/EXP_0012_PARTICLES_deconfound/`（含 deconfound_aggregate.json）；不重训、不改环境代码、test split 未打开（policy_evaluations 保持 0）。
### 实验结果
|条件|success|翻转(vs control, 420)|
|---|---|---|
|control（无粒子）|69.0%±11.8|—|
|仅评估带粒子-8|69.8%±12.2|+4/−1|
|训练+评估 粒子-8|76.4%±10.1|（+56/−25）|
|仅评估带粒子-16|69.3%±11.1|+4/−3|
|训练+评估 粒子-16|81.0%±5.1|（+66/−16）|
|仅评估带粒子-32|69.5%±10.7|+8/−6|
|训练+评估 粒子-32|78.3%±8.1|（+68/−29）|

按 seed 配对分解：评估效应 +0.7/+0.2/+0.5pp（三档都在噪声内）；训练效应 +6.7/+11.7/+8.8pp。
### 相对 baseline 的变化
粒子臂的增益几乎全部来自训练时与粒子共存（约 +7~12pp），评估条件本身贡献 ≈0。之前的 confounded 标注现在解除：结论改为"动态粒子训练是一种有效的域随机化正则"，非评估伪影。
### 出现的问题
无。评估器与 EXP_0010–0013 轮次相同（control 69.0% 复现一致）。
### 当前解释
训练时粒子扰动迫使策略学会对动态障碍的鲁棒控制，迁移回无粒子/带粒子评估都更稳。16 粒子最优（+11.7pp），32 略降（+8.8pp，过强扰动）。仍为 validation 证据、n=3，非 test 结论。
### 创新性影响
Scientific Contribution：粒子训练增益的因果分解（评估贡献排除）。Algorithmic Improvement：粒子训练正则为候选，与 variable-N padding 正则独立。Potential Novel Contribution：无新增主张（域随机化是已知技术，本结果是任务特定证据）。
### 下一步
预登记 EXP21 收尾行为修补（依据拖尾取证）；跨批次对照汇总报告。

## 2026-09-27 — 跨批次对照汇总报告 — 完成

### 今天解决的问题
把 EXP_0010–0013（含去混淆）与 EXP16–20 v2 两个批次的证据合成为一份对照报告，明确两批评估分布不同、绝对数字不可直比，可比的是各自配对结论与逐场景难度结构。
### 修改内容
新增 `research/CROSS_BATCH_REPORT.md`：批次概览、两批结果表、去混淆结论、逐场景难度对照（femoropopliteal_pad 两批皆最难、coronary_rca 在粒子分布下显著变难）、五条合成结论、EXP21 候选配方（variable-N + 粒子-16 + GRU 预测为底座；尾巴过采样/早截断/potential shaping 三选一主变量；避开三个已证伪方向）。
### 为什么这样修改
用户要求"自动补齐2、3"：任务 2 是预登记 EXP21，任务 3 是本报告。报告先行，为 EXP21 预登记提供证据引用。
### 实验结果
无新实验（汇总性质）。报告中所有数字均出自已归档原始文件。
### 相对 baseline 的变化
N/A（汇总）。
### 出现的问题
批次 B 的 eval config.json 为空（worker 未写评估侧配置），评估条件从协议文档与 episodes.jsonl 的 particle_seed 字段交叉确认。
### 当前解释
见报告第 5 节五条结论。
### 创新性影响
Engineering Improvement（汇总与可追溯性）。无新增性能主张。
### 下一步
EXP21 预登记（见下一条目）。

## 2026-09-27 — EXP_0021 预登记 — 完成（未启动）

### 今天解决的问题
按用户"自动补齐 2、3"指令，完成 EXP21 收尾行为修补的预登记文档：`research/experiments/EXP_0021_TAIL_REPAIR.md`。
### 修改内容
仅新增预登记文档。三个候选主变量（尾巴过采样课程 / no-progress 早截断 K=50 / potential-based shaping Φ=−剩余质量）三选一，启动前由用户选定，不合并。推荐 (b)：改动最小、直接消灭"白耗 227 步"、与 GAE 兼容。
### 为什么这样修改
拖尾取证判定收尾失败是训练密度+终止信号问题（非观测/物理）；两批对照报告确认该瓶颈跨分布存在（femoropopliteal 两批皆最难）。底座组合三项（variable-N + 粒子-16 + GRU 预测）均为已验证独立正则。
### 实验结果
未运行。预登记不构成训练授权——3M×3 seeds×2 臂正式训练须用户单独确认。
### 相对 baseline 的变化
待测。主对照臂 = 底座自身（验证三项正则叠加），主变量臂 = 底座 + 修补。
### 出现的问题
(c) 与现有 progress 10×/milestone 奖励部分重叠，边际收益存疑——已在文档中标注。(a) 的状态注入器是新代码面，实现风险高于 (b)。
### 当前解释
HYPOTHESIS — NOT VERIFIED：收尾修补能把 arm 17 的 61% 拖尾失败转成成功。判定指标含拖尾直接度量（质量停滞步后的剩余步数）与机制验证（尾巴状态访问频率）。
### 创新性影响
预登记本身为 Engineering Improvement；若 (b) 有效，为任务特定的训练技巧证据（早截断是通用技术，不宣称新颖性）。
### 下一步
等用户选定主变量 (a)/(b)/(c) 并授权启动。未获授权前不动 3M 训练。

## 2026-09-27 — EXP_0021 — 启动（用户授权：主变量 (b)，启动确认）

### 今天解决的问题
用户选定主变量 (b) no-progress 早截断并确认启动。实现、测试、冒烟、正式批次全部完成。
### 修改内容
- `environments/exp21_tail_env.py`：TailTruncationVector/Single（K=50 连续零清除 → truncated 非 terminated，value 继续 bootstrap；cut 后即时 _reset_envs，语义与 horizon 截断一致）；make_training_env（56 envs、14 场景均衡、variable-N 8 槽/5 实际、粒子 16、GRU 预测观测栈）；make_eval_env（dev val v2 分布：episode_seed 72M 起、粒子 24、5 真实机器人）。
- `marl/gnn_advanced.py`：EdgeFeatureGAT/EdgeBiasGATLayer/AdaptiveEdgeGATLayer 增加 agent_mask 支持（pad keys+queries 双侧屏蔽 + nan_to_num，镜像 gat_policy 的已验证处理）；AdaptiveEdgeGATActor/EdgeBiasGATActor encode 传入 mask。此前这些架构不支持 padding，variable-N 只在 'gat' 架构上验证过。
- `scripts/exp21_worker.py`：train/evaluate 两臂共用，唯一差异 stall_limit（base=∞，repair=50）。
- `tests/test_exp21_tail.py`：6 项新测试（cut 语义、计数器重置、有进展不切、单环境、shell 聚合、adaptive_edge_gat padding 等价 2.4e-7）。
- `research/scripts/run_exp0021.sh`：driver，断点续跑（summary.json 存在即跳过）。
### 实验结果
- 冒烟（1792 transitions，CPU）：repair 臂 9/9 回合被 stall cut（steps=50），base 臂无 cut；PPO 更新前 ratio 最大误差 2.9e-6；评估链路两臂各 28 回合跑通。
- 全量 pytest：243 passed / 1 skipped（taskset 0-5,8-23）。
- 正式批次 19:51 于 tmux `vrl-exp0021` 启动：repair/base × seeds 42/43/44 × 3M transitions（~747 fps，预计 6×~67min ≈ 6.7h + 18 评估）。GPU0 训练，评估固定 cuda:0。
### 相对 baseline 的变化
待测。base 臂=底座自身（三项已验证正则叠加：variable-N + 粒子-16 + GRU 预测），repair 臂=底座+早截断。这是收尾修补的配对对照。
### 出现的问题
- 实现中发现 adaptive_edge_gat 系列架构原不支持 agent_mask（EXP_0006 的 variable-N 验证用的是 'gat' 架构）；已补齐并加等价测试。此改动影响三个 edge 架构的无 mask 路径吗？不影响——mask=None 时行为与原来逐位一致（新增分支仅在 mask 非 None 时进入），全量测试通过佐证。
- 冒烟中 stall cut 后立即有新回合记录，episodes.jsonl 正常滚动。
### 当前解释
HYPOTHESIS — NOT VERIFIED：早截断把拖尾死时间换成有效探索，提高 success 与拖尾指标。判定按预登记（repair−base ≥ 0 且拖尾指标下降）。
### 创新性影响
Engineering Improvement（mask 补齐、cut 语义实现）。算法有效性待实验。
### 下一步
批次跑完后：按预登记计算配对结论（success、拖尾指标、尾巴状态访问频率），更新 registry 与报告。

## 2026-09-27 — EXP_0021 — 启动纠正（GRU predictor 接入）

### 今天解决的问题
首次启动后发现 worker 未传 `--predictor`（Features 退化为 CV 预测），偏离预登记底座规格（GRU 预测）。在 580k transitions 处停止（tmux kill + 进程终止），中断目录保留为 `repair_s42_interrupted_attempt1`（无 summary.json，不进统计）。
### 修改内容
`research/scripts/run_exp0021.sh`：train/evaluate 均传入 `--predictor research/runs/EXP16_20_FORMAL_20260927a/jobs/motion_fit_s4x_attempt1/model.pt`（该 seed 的 gate-passed GRU，来自 EXP16-20 正式批次）；带 predictor 的 CPU 冒烟 1792 transitions 复验通过（ratio 误差 2.6e-6）。19:57 重新启动，checkpoint meta 会记录 predictor_sha256 以便核验。
### 出现的问题
中断 attempt 保留；断点续跑逻辑以 summary.json 为准，中断目录不会误跳过。
### 当前解释
同上条目，HYPOTHESIS — NOT VERIFIED 不变。
### 下一步
等 1M checkpoint 出现后核对 meta.predictor_sha256 与 GRU 文件 hash 一致，再继续监控。

## 2026-09-28 — EXP_0021 — 评估 bug 发现与修复（重评估进行中）

### 今天解决的问题
首轮 18 个评估目录全部无效：训练侧成功 70%+ 但 dev 验证只有 7–17%（success mean±SD 9.76/7.38/10.24 base、13.33/15.71/14.52 repair），removal 只有 0.34–0.52，疑似系统性错误而非负结果。逐层取证定位根因。
### 取证过程
1. predictor_sha256 逐 seed 核验通过（34efef24…/589a3299…/768a1362…，GRU 确实接上了）；
2. padded act_batch vs 直接 act() 路径 A/B：同 checkpoint 同 episode 完全一致 → 不是 padding/agent_mask 问题；
3. 同 episode_seed 配对：arm 17 在 80 个 episode 上成功而 exp21 base 失败、反向仅 2，且失败模式不是收尾而是导航全程（mean removal 0.399）→ 与"收尾没学会"假设矛盾；
4. 用 arm 17 checkpoint 在 exp21 评估环境里复跑 pulmonary_saddle：0/6（其自家评估 100%）→ 差异在评估管线而非策略；
5. 因子分离：arm17 env + arm17 ckpt = 6/6，exp21 env + arm17 ckpt = 6/6，但 exp21 worker 的动作执行路径 env.step(action[0,:n]) 喂的是**原始 Frenet 帧提案**——env.step 把动作读作世界系速度指令，控制语义完全不同。
### 根因与修复
`scripts/exp21_worker.py` evaluate() 少了 `bounded` + `direct_local_action` 变换（训练路径 line 78 与 v2 评估都有）。修复后同 checkpoint pulmonary_saddle 0/6→4/6；整 140 回合重跑 s42@2M：base 17.1%→64.3%（removal 0.516→0.923）、repair 18.6%→76.4%（removal 0.530→0.955、steps 152→108、stall_cuts 32）。**训练完全没受影响**（训练路径一直正确），只是评估读数无效。新增回归测试 `test_worker_eval_applies_frenet_transform`（检查 evaluate 源码里 env.step 必须过 direct_local_action），7 passed。
### 处置
18 个无效评估目录移入 `eval_invalid_action_semantics_20260928/`（保留取证证据，不进统计）；修复后重评估在 tmux `vrl-exp0021-eval` 跑 `research/scripts/rerun_exp0021_eval.sh`（仅评估，不动 checkpoint；单 eval ~70–90s，共 18 个）。`research/scripts/analyze_exp0021.py` 已写好，跑完后出预登记配对结论（success、拖尾指标、尾巴状态密度、按场景表）。
### 当前解释
首轮数据不可用于结论（负或正都不算数）。修复后 s42 初步读数与底座预期一致：repair 2M 76.4% 已超 arm 17 的 72.9%（同 seed 同协议），且 steps 减半——早截断在生产侧的行为改善信号初现，但**须等 3 seeds 全部完成再下结论**。
### 下一步
重评估跑完后运行 analyze_exp0021.py，按预登记判定写入 registry/RESULTS_SUMMARY/CROSS_BATCH_REPORT，并更新记忆。教训入档：**每个新 worker 的评估路径必须有动作语义回归测试**（此前 16-20 worker 直接复制了带变换的路径所以没暴露）。

## 2026-09-28 — EXP_0021 — 修复后重评估完成，预登记判定：修补有效（正结果）

### 实验结果（18 个修复后评估，3 seeds × 3 checkpoints × 140 回合，dev validation v2）
成功率 mean±SD（3-seed）：

| ckpt | base（底座自身） | repair（底座+K=50 早截断） |
|---|---|---|
| 1M | 68.33 ± 4.65 | 66.90 ± 8.03 |
| 2M | 70.71 ± 6.81 | **77.14 ± 2.58** |
| 3M | **73.10 ± 4.36** | 76.90 ± 2.70 |

配对 repair−base（逐 seed）：1M −1.43pp（−8.6/+1.4/+2.9，噪声）；2M **+6.43pp ± 5.15**（+12.1/+5.0/+2.1，t=2.16，单侧 p≈0.082）；3M **+3.81pp ± 1.80**（+5.7/+3.6/+2.1，全 seed 同向，t=3.67，单侧 p≈0.034）。

- 次指标：removal 3M 0.958 vs 0.953（持平）；**steps 152.5 → 97.9**（−36%）；失败回合尾部死时间 mean 189 步 → 49 步（**−74%**，即预登记的拖尾指标，直接命中）。
- 尾巴状态密度：repair 训练流 24–34% episode 以 stall cut 收束（把死时间换成新回合探索），base 为 0（无此机制）。
- 场景面：coronary_lm 83→100%、coronary_rca 43→73%、femoropopliteal 17→30%；ica_siphon/carotid/mca/sma 两臂同低（非拖尾瓶颈，属刮壁/几何）。
- 基线对照：底座自身 73.1% vs EXP16-20 arm 17（同 GRU 底座、无 variable-N/粒子-16 训练）76.7%——叠加项在本配对协议内未复现独立增益，可能与粒子训练 16 vs 24、评估 K 语义等差异有关，标记为待查项，不影响本实验配对结论。
- 2 个 cut-失败回合 removal 0.9991/0.9995（最后一栓残 0.05–0.09% 质量时被 K=50 截断）——截断代价存在但量级 ~1.4% of episodes，远小于收益。
- 判定（按预登记）：repair−base ≥ 0 ✅（2M/3M 全 seed 正）；拖尾指标显著下降 ✅ → **收尾修补有效，正结果**。85% 参考阈值仍未达（本实验不为测它）。

### 相对 baseline 的变化
+3.8pp（3M 配对）/ +6.4pp（2M 配对），方差同步减半（SD 4.36→2.70）。

### 出现的问题
analyze 脚本两个小 bug 已修（隔离目录名误入解析、paired 段 guard 写错）；修复后输出与手算一致。

### 下一步
(a) 尾巴过采样课程 / K 扫描（30/40/50）是下一个候选主变量——3M 时 1M→2M 后训练曲线平台 + KL 升高表明"更多同课程步数"边际收益低（用户问过欠拟合，答案：不是欠拟合，是训练分布）。写入 EXP_0021 结果至 registry / RESULTS_SUMMARY / CROSS_BATCH_REPORT。

## 2026-09-28 — 弱场景取证：失败分两类（不可逆流陷阱 + 终局定位），及控制架构确认

### 用户问题
为什么有的场景成功率那么差？现在是残差 RL 还是纯 RL？
### 控制架构答案
**纯 RL（direct local）**：`control_mode='local'`，策略输出 Frenet 帧单位向量（有界归一），`direct_local_action` 只做坐标变换到世界系，**无任何引导控制器、无残差项**（residual/guided/flow_guided/flow_spread 模式在代码里存在但本线从未启用；arm 18 的解析控制器干预是已证伪的另一条路）。观测侧同样"零先验注入"：贪婪 5 步调度器只给分配（route 索引进观测），不给动作。
### 弱场景取证（repair@2M 失败回合现场复跑 + 几何/流场分析）
两类失败模式，均与"拖尾"（已修）不同源：

**A. 不可逆流陷阱（sma_embolism 26.7% 的主因）**：clot@44 位于 branch 1 中段，栓塞使局部半径 0.019→0.0067，连续性方程把流场加速到 8×inlet 上限（均值 0.032、中心 0.064），而机器人最大推力 0.018。机器人清完 clot@4 后被**冲过 clot@44 进女儿支**（branch 3，流 0.032 > 0.018），从此物理上回不去——调度器一直正确分配 clot@44，但分配不可达。失败回合终点 min_geo≈0.5（远在接触半径 0.035 外）。判定：**环境物理不可达，不是策略缺陷**。
**B. 终局定位失败（femoropopliteal 30%、ica_siphon/carotid 63%、mca 63%）**：rem 常达 0.92–1.00（质量几乎清完或贴脸），终点 min_geo 0.06–0.5 不等；平均径向占比 0.75–0.84（贴壁）+ 每回合数百次 wall_hits——与拖尾取证"nav 桶=刮壁+几何卡点"一致，是壁面润滑阻力（lubrication）+ 分叉几何下的最后一段接近问题。
### 含义
- A 类修法是**环境/物理侧**（如流场 cap 8×→3×、或给 occluded 段降 profile、或允许贴壁低速逆流），不属于策略训练问题——改这个要按"环境改动"单独预登记。
- B 类修法才是策略侧（wall penalty 0.1→0.5、窄场景过采样课程）。
- coronary_rca 修复后 73.3%（+30pp）说明 K=50 之外剩余失败里 B 类占主导。

## 2026-09-28 — EXP_0022 — 原尺寸MCA单位/血流工程验证完成，训练门槛未通过

### 本轮解决的问题

继续此前中断的MCA生理参数改进，完成显式单位、压力驱动树流量、远端阻力敏感性、
有限尺寸逆流界和完整报告。未对旧`flow_speed`做降低难度的改写，旧worker不接入新物理。

新增发现：`build_territory`的旧`mm_per_unit`按名义最大管径反推比例，
把扰动后的物理尺度重新归一到名义直径。现记录归一化前后的真实比例
`source_units_per_unit`/`physical_mm_per_unit`，用于EXP22；不改旧字段、几何坐标或半径。

### 参数来源

- PC-MRI MCA 146±31 mL/min来自DOI `10.1038/jcbfm.2014.241` Table 1。
- An 2026补充表4的0.25–1.0 mm/s只是训练随机化群体速度；采用1.0作乐观工程参考，
  不声称是本设备、单机器人或血液中的实测上限。
- 50 ms周期是工程假设；文献给的总延迟<30 ms不能直接当控制周期。
- 半径0.025/.05/.08 mm、阻力比0/1/9、阻塞宽度2.5 mm均标明未标定。
- 论文共享Helmholtz磁场与现有MARL独立多机器人指令之间有执行器差距。

### 实际验证

42项MCA测试通过；全套286 passed、1 skipped。6种几何配置、1,296个流场、
11,664个尺寸/速度组合、3,888个积分周期组合均已输出。
健康流量恢复、分叉质量守恒、再通入口流量单调、完全闭塞站点零流量通过；
最大守恒残差4.55e-13 mm³/s。报告保存所有组合，不按导航难易挑条件。

名义直径3 mm、146 mL/min对应平均344.25 mm/s；半径.08 mm、推进1 mm/s的
最乐观近壁逆流余量−70.48 mm/s。健康目标站点1,944/1,944组合在局部稳态圆管模型
中排除逆流；这是参数网格计数，不能解释为策略失败率，也不能排除顺流到达。

### 异常与处理

第一次报告因NumPy bool_序列化失败，失败目录保留，增加原生布尔转换和回归测试后
重跑通过。第二次输出之后核对原文，将流量引用定位Table 2纠正为Table 1，
最终再次生成可核验hash的报告。中间目录不是独立科研重复。

### 判定与下一步

**环境工程验证通过，策略成功率未评估，未证明85%。**
新物理尚未接入RL worker；必须进一步实现物理子步/出口离场/同流场粒子，
并明确接触及每秒溶解速率、机器人尺寸速度、出口阻抗和共享场执行约束。
旧300步若按50 ms解释只有15秒，旧0.018步长相当25.67 mm/s，不能直接沿用。

本轮类型：Engineering Improvement，不列算法增益、不与旧物理成功率混池。
协议：`research/experiments/EXP_0022_MCA_PHYSIOLOGY.md`。
最终报告：`research/validation/EXP0022_MCA_20260928_final/REPORT.md`；
完整性：同目录`COMPLETE.json`；测试：`research/validation/EXP0022_FULL_TESTS_20260928.xml`。

## 2026-09-28 — EXP_0022B — 新环境数值修复与纯RL接口复验完成

修复未裁剪轴向跨段判断、弯管端帽入射方向、两侧开口检查、端点限制的中点积分和壁面投影后出入口事件。未改变血流/尺寸/推进/溶解/成功定义；只修复数值实现，低层仍为纯RL direct-local。

最终0.1秒三seed全动力学最大误差0.000755281 mm；独立完整1秒固定流场输运在0.1/0.05/0.025相对0.0125参考全部通过，最大误差0.002475805 mm，阈值保持0.05 mm，掩码/出口身份一致。请求1秒的Gym轨迹因全体机器人离场在约0.15秒提前结束，因此另做完整时间窗输运。

全套329 passed、1 skipped；专门输运/接口43项通过；纯RL新schema六条transition、一次优化更新冒烟通过，无新正式训练/成功率。所有中间失败保留；接手源码与最早失败产物的输运源码hash不同，不能混淆因果。

`formal_training_ready=false`保留；下一步是新物理纯RL任务协议与可达性、初始化/时限及长时间接触反馈验证。最终报告：`research/experiments/EXP_0022B_NUMERICAL_REPAIR.md`。

## 2026-09-29 — EXP_0023 新MCA纯RL正式训练启动

用户明确授权启动。按现有物理任务、新权重、42/43/44三seed各300万环境步启动独立后台进程；通过43项回归与GPU断点续训一致性检查。保持1秒时限及全部物理输入，记录完整清除不可达的任务上限，不声称成功率改善。协议见 `research/experiments/EXP_0023_MCA_PURE_RL.md`，输出见 `research/runs/EXP_0023_MCA_PURE_RL_20260929a/`。

## 2026-09-29 — EXP23 编译并行续训与三维渲染

三seed从8192步检查点迁移到EXP_0023_MCA_FAST_20260929b，物理参数不变，1×128改为8×16采样；首批吞吐93–104步/秒，原约1.8。342项测试通过、1项跳过，9组轨迹配对通过。恢复watch_gui.py对应VTK渲染，保留PyBullet入口。详见 `research/experiments/EXP_0023_ACCELERATION_AND_VIEW.md`。


## 2026-09-30 EXP26：先修正体外任务可完成性，再训练

> 最新状态（2026-09-30）：用户确认“暂无实测，先做体外仿真基线”。EXP26 可完成性验证 20/20、参考版 3/3；GPU 短跑 16384 步通过，已启动三个种子各 300 万步纯 RL（两张 RTX4090），保留 VTK 回放及监控。此为体外低流量仿真，不是人体生理条件或 RL 达到 80%。详见 research/experiments/EXP_0026_IN_VITRO_FEASIBLE.md。EXP24 保持暂停。

新运行 research/runs/EXP_0026_MCA_IN_VITRO_20260930a。低压参考流量 0.01 mL/min、180 秒、有限机器人与狭窄表面反应间隙 0.04 mm；保持原尺寸 MCA、分散起点和全清除标准。诊断控制器 20/20 在16.1–49.0秒完成，训练完全从零且不使用诊断器。新证书门禁绑定源代码，修改后必须重新验证。独立策略评估在2万步起记录。


## 2026-09-30 EXP26 接触范围复核及暂停

> 最新纠正（2026-09-30）：用户质疑金色血栓范围过大。核对确认 EXP26 的表面接触范围名义长 15 mm（sigma=2.5 mm，两侧各3sigma），前两处相隔7.2 mm，有重叠；这不只是渲染变化。其尺寸和整颗血栓的统一扣质量方式未经验证。EXP26 已保存并暂停，不把83.3%的新任务成绩当作原任务恢复或真实血栓模型正确的证据。后续需先明确局部实体血栓的几何边界与接触定义。

审计数据：research/validation/EXP0026_CONTACT_EXTENT_REVIEW_20260930/summary.json。中心截面及上游6.5 mm表面接触，在首10 ms均去除0.0036归一化质量，证实外围也可按相同速率清除整颗血栓。可完成性证书仅证明此简化模型可解，不证明其空间形状/动力学经过校准。


## 2026-09-30 EXP27：分散点目标及更多动态粒子

> 最新状态（2026-09-30）：按用户要求恢复四处分散点状血栓（局部0.12 mm接触）、动态粒子8→32并加入接触惩罚。EXP27可完成性20/20、参考版3/3、387项测试通过，GPU预检16384步通过；已从零启动三种子各300万步训练及VTK回放。EXP26保持暂停，其83.3%成绩不沿用。详见 research/experiments/EXP_0027_DISTRIBUTED_POINTS.md。

动态粒子模型：随流运动与重叠代价，没有刚体反弹。高斯场只作水力阻力近似，不定义反应范围或血栓实体。0.13 mm径向、原环状内壁、6.5 mm远处均不再清除。新schema mca_point_36_v3，观察及奖励指向中心点。可完成性轨迹24.2–55.3秒，纯RL成绩另记。运行 research/runs/EXP_0027_MCA_POINTS_20260930a；训练进程3412595/3412596/3412597，cuda:0/cuda:1/cuda:0。

## 2026-09-30 — EXP28 随机初始化与更快流动

> 最新状态（2026-09-30）：EXP28 已接续 EXP27 纯RL权重长跑。每回合随机机器人起点、三种障碍物疏密分布、15–25 μL/min体外参考流量；VTK 10倍回放且重启换种子。100组随机化检查通过，诊断30/30+参考5/5可完成，392测试通过，16384步GPU预检完成。EXP27已保存暂停，其50万步60/60成绩仅属于旧工况。详见 research/experiments/EXP_0028_RANDOMIZED_POINTS.md。

EXP27 安全暂停于 828192 / 610080 / 550176 步。新协议/源代码/父权重已冻结；三个 GPU 训练进程均已产生新更新。新环境的策略成功率需独立报告，不能套用诊断或旧任务成绩。

## 2026-09-30 — EXP28 实际速度要求确认

用户明确回复“实际障碍物移动更快，并纳入训练”，与已运行EXP28的每回合1.5–2.5倍共享流动一致。保持冻结训练连续运行；10倍VTK回放只影响显示。三种子各50万步独立评估均19/20，合计57/60=95%；seed42的100万步单独评估20/20，其余种子同阶段评估尚未齐全。验证集仍是各策略相同的20种布局，不代表完整鲁棒性或人体工况验证。

## 2026-09-30 — EXP29 全对象随机与避障修正

> 最新状态（2026-09-30）：EXP29已启动三种子接续训练与VTK回放。血栓、五个机器人、32个粒子每回合全部随机；增加四粒子预测观测、碰撞事件/接近惩罚，清除与无碰撞清除分别评估（100种布局）。100组三类初始化检查、400测试及16384步GPU预检通过。EXP28保存暂停；其100万步清除56/60但无粒子碰撞清除27/60，不能再称其100%可靠避障。详见 research/experiments/EXP_0029_ALL_RANDOM_AVOIDANCE.md。

论文Medany等2025年MBRL明确输入目标位置；Pygame图像导航与实物PZT控制的成功率不可直接对比本项目理想精确状态任务。源码、父权重和失败诊断尝试均保留。新长跑无oracle示范，未引入世界模型。

## 2026-09-30 15:21 EXP29随机布局成功率核查与世界模型预研

> 2026-09-30 15:21（Asia/Shanghai）EXP29进度审计：三seed新增50万步，各100个同布局开发验证，完整清栓33%/15%/27%（平均25%），零粒子接触且五机器人保留的清栓8%/2%/5%（平均5%）。尚未达到80%；纯RL训练/VTK继续。用户接受暂保留独立理想速度执行器；世界模型仅完成代码兼容性审计和接入方案，未启动或接入。详见 `research/validation/EXP0029_PROGRESS_20260930_1521/REPORT.md` 与 `research/experiments/EXP_0029_WORLD_MODEL_READINESS.md`。

## 2026-09-30 EXP30纯RL修复与配对筛选

> 2026-09-30T15:56:04+08:00 EXP30纯RL对照已启动：修复PPO的mask小批次损失分母；A原actor学习率3e-4、B降至1e-4，各三seed从同一EXP29的50万步权重起跑，各新增65,536步、20布局配对筛选。401测试通过/1跳过，两组16,384步GPU预检通过。EXP29已保存暂停，VTK/监控跟随新组。世界模型后置；尚无80%或改善声明。详见 `research/experiments/EXP_0030_PPO_REPAIR.md`。

## 2026-09-30 EXP30短跑结果与长跑延长

> 2026-09-30T16:08:06+08:00 EXP30短跑两组已完成：A三seed平均完整清栓21.7%、零粒子接触且全机器人保留的清栓6.7%；B分别28.3%、10%。每seed仅20种配对开发布局，未达到80%，不宣称显著或稳定提升。已启动等预算延长：保留完整状态，各seed/组累计500,000步，先B后A，最终各100布局复核；世界模型仍禁用。代码已首次推送8baed82，后续记录将随本轮补充提交。

## 2026-10-01 EXP31–37 汇总（RESEARCH_LOG 补记）

> 2026-10-01T17:00+08:00 各实验均为3个训练seed×每组新增500K步，在100个共享开发布局上用确定性动作评估完整清栓率（均值）。详细报告见各 `research/validation/EXP00xx_RESULTS_*/REPORT.md`。

- EXP31 探索噪声退火：control 24.7% → annealed 34.7%（seed44 下降），不推广。
- EXP32 原始轨迹观测：masked 45.0% → trajectory 12.0%，负结果。
- EXP33 有界/锚定轨迹观测：证书因EXP34改代码作废，未长跑。
- EXP34 三组拆分：masked 32.7% / masked_potential 27.7% / routed 47.3%。路由观测效应 +19.7 个百分点，三seed都为正；单调势函数自身没有收益。
- EXP35 恒定满速执行器（`command_speed='unit'`）：continue 38.3% → unit_speed 63.7%（52/79/60），三seed都为正，是目前最好的模型。
- EXP36 势函数系数×4：control 37.0% → shaping 21.3%，负结果。同时发现续训退化：父权重复评64.7%，原配置续训后37.0%。
- EXP37（进行中）：actor lr 3e-5 vs 1e-4，每10万步存checkpoint，在独立选择集（seed base 950000000）上选模型后再报告开发布局成绩。
- 诊断（`EXP0034_FAILURE_HANDOFF_DIAG_20260930`、`EXP0035_FAILURE_DIAG_20261001`）：失败几乎全是超时，多数剩一个目标；t=120 s 改由已知地图跟随器接管后基本都能清完；末段策略方向与路线方位余弦约0.1。80%目标未达到，世界模型未启用。

## 2026-10-01 EXP35 失败归因、EXP37 结果、EXP38 势函数诊断、EXP39 就绪

> 2026-10-01T18:30+08:00 只读归因（`research/validation/EXP0035_FAILURE_ATTRIBUTION_20261001/FAILURE_ANALYSIS_EXP35.md`）：在 300 个 EXP35 评估回合上逐步重放（与记录逐回合一致率 91–95%）。106 个失败全部是超时，72% 剩最后一个目标；但残留目标的测地最近距离中位为 6.6 mm，旧的"贴着血栓拖尾"结论在 EXP35 不成立。仅评估的替换实验：距自己目标 >0.7 mm 时改用已知地图跟随器，成功率 87/96/95%；≤0.7 mm 时替换只有 54/84/66%（父权重 51/78/65%）。瓶颈是中远距离的路由跟随（导航余弦 0.14–0.46），不是收尾/接触、观测缺失或分支规划。

- 根因之一：mass_weighted 势函数对所有剩余目标取平均，32% 的机器人朝自己目标走反而得负塑形，最后一个目标阶段梯度只有 0.075/mm。新增 `progress_potential='assigned'`：到自己分配目标的距离；分配变化那一步塑形置 0。原有选项算法不变。测试 `tests/test_mca_assigned_potential.py`。
- EXP37（`research/validation/EXP0037_RESULTS_20261001/REPORT.md`）：两组续训都退化，500K 时 low_lr 54.0%、control 40.0%，父权重 63.7%。
- EXP38 50K 诊断（`research/validation/EXP0038_DIAGNOSTIC_20261001/REPORT.md`）：assigned 53/71/59%，control 32/53/58%，逐场配对 +13.3 个百分点；但还没超过父权重，导航余弦 0.252（父权重 0.236）。
- EXP39 正式配对（assigned vs control，500K × 3 seed，10 万步里程碑 + 选择集选模）：证书和 16K 预检都通过，驱动为 `scripts/run_mca_assigned_formal.py`。**未启动，等待用户批准。** 80% 目标未达到，世界模型未启用。

## 2026-10-01 21:50 EXP39 sealed-test result (first formal result on the fixed splits)

> Checkpoint selected on the validation split (200 layouts), one evaluation per selected checkpoint on the sealed test (500 layouts). assigned 49.4/78.2/70.6 (66.1%), control 55.8/78.2/60.6 (64.9%), EXP35 sealed baseline 61.9%. Paired +1.2 pp, seeds not in the same direction: no reliable gain. Training stability clearly improved (milestone-mean validation 61.7% vs 43.4%, at 500K 66.5% vs 40.0%): the own-target potential removes the continued-training degradation. 80% not reached. Report: `research/validation/EXP_0039_SEALED_RESULTS/REPORT.md`.

## 2026-10-02 00:40 — EXP40 / EXP41 sealed results; all 14 anatomies registered

Sealed test, MCA, 500 layouts, selection on the 200-layout validation split:

| method | seed 42 / 43 / 44 | mean | collision-free (mean) |
|---|---|---:|---:|
| EXP35 pure RL baseline | 47.0 / 78.2 / 60.6 | 61.9% | — |
| EXP39 assigned potential (pure RL) | 49.4 / 78.2 / 70.6 | 66.1% | — |
| EXP40 own bearing in fixed obs columns (pure RL) | 59.2 / 75.2 / 70.2 | 68.2% | 22.2% |
| route prior alone (no learning) | — | 88.4% | 22.8% |
| EXP41 route prior + RL residual 0.5 | 88.0 / 89.2 / 94.2 | 90.5% | 24.0% |
| route + avoid prior alone, gain 6 (no learning) | — | 94.4% | 84.8% |

- EXP40 vs EXP39 paired: +9.8 / −3.0 / −0.4 pp. Exposing the own bearing helps pure RL only slightly and inconsistently; pure RL still does not learn to follow it.
- EXP41 vs EXP40 paired: +28.8 / +14.0 / +24.0 pp. The structural prior, not the network, carries the gain. EXP41 vs the prior alone: +2.1 pp mean, but collision-free rate is unchanged (~23%), so the RL residual did not learn particle avoidance.
- The hand-written avoid term is still the strongest result. EXP42 (RL residual on top of it) is training.

Multi-anatomy: `DynamicsConfig.anatomy` (default mca_m1_lvo) selects the territory; all 14 territories now have fixed validation/diagnostic/test splits in configs/evaluation_splits.json (diagnostic rule moved to 970M+k·1M after a collision with MCA validation at k≥10). Prior-only probe on 30 diagnostic layouts each: 13 of 14 anatomies at 97–100% complete, sma_embolism 90%, MCA 97%; MCA is the hardest territory for this controller. Zero-shot sealed tests of the gain-6 prior on the 13 non-MCA anatomies are running (study PRIOR_ZERO_SHOT_ANATOMIES, 13 declared).

## 2026-10-02 01:20 — EXP42 and zero-shot multi-anatomy sealed tests

- EXP42 (RL residual 0.5 on route+avoid prior, 500K × 3): sealed 95.4 / 94.8 / 92.8, mean 94.3%, collision-free ~83.8%. Paired vs prior alone: +1.0 / +0.4 / −1.6 pp. No gain from RL over the prior.
- Route+avoid prior, zero-shot on the 13 other anatomies (sealed, 500 layouts each): 11 at 100% complete, sma_embolism 93.2%, cerebral_venous_sinus 88.0%. See research/validation/MULTI_ANATOMY_PRIOR_SEALED_20261002/REPORT.md.
- Open: cerebral_venous_sinus (retrograde flow) is the weakest anatomy; per-anatomy failure analysis on its diagnostic split is the next step.

## 2026-10-02 22:00 — EXP43 / EXP44 sealed

- Diagnosis behind EXP43: on the unit-speed actuator the EXP42 residual (0.5) could never cancel the unit prior, so RL could not wait. A hand-written wait rule (sealed baseline BASELINE_WAIT_PRIOR_ONLY) gives 92.6% / 91.4% collision-free, the strongest traditional result.
- EXP43 (residual 1.0 + stop deadzone, 1M × 3, selection on collision-free): sealed complete 98.4 / 98.6 / 99.6 (98.9%), collision-free 88.4 / 88.2 / 90.0 (88.9%). Paired vs prior alone +4.5 / +4.0 pp; vs wait rule +6.3 pp complete, −2.5 pp collision-free.
- EXP44 (residual RL over the wait-rule prior): worse; complete 95.2 / 90.0 / 87.0, collision-free 89.8 / 88.6 / 82.6. Two seeds selected at 100K, i.e. RL degraded the stronger prior. Negative result.

## 2026-10-02 23:10 — EXP43 + shield; EXP45

- EXP43 selected checkpoints + post-policy shield (stop on forecast clearance < 0.3 within 0.5 s), no retraining: sealed complete 98.4 / 97.8 / 98.8 (98.3%), collision-free 96.4 / 96.4 / 95.0 (95.9%). Paired vs the wait rule, the strongest traditional controller: +5.8 / +5.2 / +6.2 and +5.0 / +5.0 / +3.6 pp. First method that beats every traditional controller on both metrics on all seeds. Report: research/validation/EXP0045_SHIELD_PROBE_20261002/REPORT.md.
- EXP45 (continue with shield in loop, collision penalty 20, 500K): 97.9% / 95.6%; paired vs EXP43 + shield −0.4 / −0.8 / 0.0 pp. Null result.

## 2026-10-03 — Cross-anatomy sealed comparison (14 anatomies × 6 methods, 168 sealed evaluations)

Report: research/validation/CROSS_ANATOMY_COMPARISON_20261003/REPORT.md. All methods tuned on MCA only; 13 anatomies zero-shot.
14-anatomy mean (complete / collision-free): pure RL 27.0 / 15.7; route only 97.8 / 76.4; route+avoid 98.3 / 97.4; route+avoid+wait 98.1 / 98.0; residual RL 98.0 / 97.1; residual RL + shield 98.0 / 97.7.
- Pure RL does not transfer (0% on 4 anatomies). Traditional and residual methods are tied on the 14-anatomy mean; residual RL's MCA gain (+5.7 / +4.5 vs wait rule) is offset by losses on femoropopliteal (−5.6), pulmonary (−4.8), SMA (−3.9), popliteal (−3.8), and it fixes cerebral venous sinus (+11.7).
- Ledger bug found and fixed: completion marking matched only (sha, study), so concurrent jobs sharing weights claimed each other's results (151/168 mis-pointed in this study only; all earlier studies ran sequentially and are clean). Entries re-pointed to each job's own result file; completion now matches (sha, study, label, anatomy, protocol); regression test added.

## 2026-10-03 — EXP_TEACHER_BC and EXP_ONLINE_IMITATION round 1 (branch research/graph-teacher-distillation)

- Question: can a topology-reading Graph Transformer student imitate the route+avoid+wait teacher with no route controller at inference, and transfer to held-out anatomies?
- Hypothesis: yes for training anatomies; held-out transfer better than pure RL because the scene graph carries topology instead of memorised geometry.
- Single main variable: policy (teacher vs BC student vs pure RL); for EXP_ONLINE_IMITATION r1, training data (BC + DAgger states vs BC only, same steps/schedule/init).
- Code: dc42d2e + trainer schedule fix. Seeds: student seed 0 (one seed). Budget: BC 8 epochs (24,848 steps × 64 scenes); DAgger r1 / control 14,121 steps each.
- Diagnostic result (20 layouts × 14 anatomies): teacher 98.9 / 96.0 complete (train / held-out); BC epoch 7 97.8 / 91.0; pure RL EXP40 21–33 / 16–34; DAgger r1 97.8 / 89.0 vs control 96.1 / 91.0.
- Validation / sealed test: not run (development stage).
- Interpretation: imitation works and transfers zero-shot far better than pure RL. Epoch-0 gap was under-training (offline cos 0.980→0.988 but +20 pp closed loop). DAgger r1: closer tracking of the teacher (cos +0.025, CI excludes 0), no measurable completion gain (train +1.7 [−1.7, +5.0], held-out −2.0 [−6.0, +2.0]).
- Failure modes: held-out wall contact 10–16 s vs teacher 0.01 s; femoropopliteal 75–85 and SMA 75–80; stop recall ~28%.
- Next: GAT vs Graph Transformer with the same BC data; 3 seeds; larger diagnostic sample; DAgger rounds 2–3 with β→0 only if they show a closed-loop gain.
Report: research/validation/EXP_TEACHER_BC_20261003/REPORT.md

## 2026-10-03 — EXP_FAIR_OBS_REACTIVE: fair (Turbo-style) observation (branch research/fair-partial-observation)

- Question: with only Turbo-level information (no map, no route, no allocation; straight-line clot vectors, local lumen view 4 mm, particles/teammates within 1.5 mm, 2.5 % noise), how much of the privileged teacher's performance does a reactive controller keep?
- Single main variable: controller (privileged teacher vs reactive_bearing vs reactive_path), all on the same layouts.
- Diagnostic, 20 layouts × 14 anatomies, 5 robots. 14-anatomy mean collision-free completion: teacher 97.9 %, reactive_bearing 79.6 %, reactive_path 66.1 %.
- reactive_bearing reaches its rate by scraping the wall: robots touch the wall 52 % of the time (median 101 robot-s per episode vs 0 for the teacher). With wall contact < 1 robot-s it drops to 1.1 % (teacher 96.1 %, reactive_path 64.6 %). The current success definition ignores wall contact, and the simulator lets bodies slide along the wall, so straight-line pushing is rewarded by the metric.
- reactive_path follows the true route direction with cos 0.61 (19 % wrong half-space) vs −0.05 for bearing; its failures are timeouts at junctions.
- Interpretation: removing privileged map information removes most of the reactive controller's advantage, and exposes a metric gap (wall contact). Fair results: research/validation/EXP_FAIR_OBS_REACTIVE_20261003.

## 2026-10-03 — EXP_SAFE_BASELINES: all existing controllers under the Safe Success metric

Primary metric from now on: SafeSuccess = task success AND total wall contact over all robots < 1.0 robot-s (scripts/safe_metrics.py); raw success kept as secondary, Unsafe Success Gap = raw − safe. Diagnostic split, 20 layouts × 14 anatomies, 5 robots, 3,360 episodes. Privileged and fair controllers reported separately.
- Privileged (known-map route information): teacher raw 97.9 / safe 96.1; residual EXP43 98.1 / 95.4; residual + shield 98.3 / 95.6; pure RL EXP40 28.3 / 21.9 (its observation contains the route bearing, so it is privileged too).
- Fair observation: reactive_bearing 83.2 / 1.1 (gap 82.1 pp, wall contact 52 % of robot time); reactive_path 66.1 / 64.6 (34 % timeouts).
- MCA is the anatomy where even privileged controllers lose most to wall contact (teacher safe 70 %, residual 62 %).
- Interpretation: under the safe metric, every method with fair information is far below the privileged ones; the residual/shield methods' earlier advantage over the teacher disappears (95.4–95.6 vs 96.1). Table: research/validation/EXP_SAFE_BASELINES_20261003/summary_table.md.

## 2026-10-03 — EXP_WALL_KEEPING_PROBE: best previous method under the safe metric, with explicit wall keeping

- Question: if the controller is also told to keep off the wall, what safe success does the previous best method (EXP43 residual RL + shield, privileged information) reach?
- Change (single variable): wall-keeping term in the action prior from the robot's own lumen observation (outward radial direction, clearance): inside 1 robot radius of the wall, remove the outward command component and push back by gain g. Default off (action_wall_gain = 0).
- Gain chosen on 40 MCA diagnostic layouts (g ∈ {0,1,3}, margin ∈ {1,2}): shield seed 42 safe 42.5 → 92.5 (g 3); teacher 65 → 90 (g 1). The 14-anatomy check reuses 20 of those MCA layouts, so the MCA row is optimistic; the other anatomies were not tuned on.
- 14 anatomies × 20 diagnostic layouts: residual + shield safe 95.6 → 97.7 (seeds 98.2 / 97.1 / 97.9), raw 98.1, wall contact 0.18 → 0.08 robot-s; teacher 96.1 → 97.1. MCA: shield 62 → 90, teacher 70 → 85. Held-out anatomies: shield 96.0 → 96.3 (already wall-free).
- Remaining failures: sma_embolism 87 % and femoropopliteal 95 % are timeouts, not wall contact.
- Still privileged information and an idealised simulator; not a sealed result.

## 2026-10-04 — EXP0046 registered: parallel multi-cluster thrombus removal

> Historical v1 entry. The causal crash attribution and controller descriptions
> below are superseded by the revision-2 correction later in this log. V1 raw
> results remain preserved and are not pooled with v2.

The new research line treats one controlled entity as one magnetic cluster.
Method 1 is the `N=1` sequential route+avoid+wait baseline; method 2 is
`N=2,3` parallel local-observation control with a right-of-way spacing shield.
The registered spacing sensitivity is `d_min={1,2,4} mm`, with Safe Success,
raw success, removal, time, wall/particle contact, spacing violations, yield,
and deadlock reported together. The protocol is in
`research/experiments/EXP_0046_MULTICLUSTER_PROTOCOL.md`.

The first implementation audit found that long reference-environment rollouts
can hit a native boundary-exit crash. The evaluator now defaults to a 5 s
development horizon and marks all outputs as feasibility diagnostics until the
integrator gate passes. No formal or sealed result is claimed. The current
spacing rule is an uncalibrated operational surrogate; the repository still
needs a measured magnetic coupling model for a physical independence claim.
The controller now also holds a robot briefly after repeated local junction
edge returns or a prolonged single-edge stall; this is a numerical robustness
guard, and the long-horizon gate remains open.
An episode-isolated wrapper was added after a 5 s two-episode probe still
observed a SIGILL in the second child process; the wrapper records that seed as
a crash and preserves completed rows. This confirms the issue is an unresolved
physical-integrator/runtime gate rather than a result of controller ranking.

### EXP0046 short smoke (MCA, two episodes per cell, 5 s)

The initial in-process development matrix completed, but the subsequent
episode-isolated matrix exposed native crashes in several seeds even at 5 s.
Among completed child processes, all cells had zero Safe Success because 5 s
is a feasibility horizon, not a complete-treatment horizon. Mean removal was
2.1% for the single-cluster baseline; for two clusters it was 16.7%, 5.6%,
and 0% at `d_min=1,2,4 mm`, and for three clusters it was 7.9%, 16.7%, and
12.5%. Crash counts were 2/5, 0/5, 2/5 for two clusters and 1/5, 2/5, 1/5
for three clusters at `d_min=1,2,4 mm`. These values are diagnostic only;
the crash rate is itself a numerical gate and no performance ranking is
valid. Raw JSONL and per-cell summaries are in
`research/validation/EXP0046_ISOLATED_MATRIX_20261004/`.

## 2026-10-04 — EXP0046 revision 2: protocol and observation repair

- Correction: earlier boundary-exit/junction explanations of SIGSEGV/SIGILL
  were hypotheses stated too strongly. Native-debugger and isolated full-length
  runs on unchanged source did not reproduce the failures; the native root
  cause remains unknown. The edge-stall guard is removed, not credited as a fix.
- The controller now takes only measured-observation arrays. True flow, edge
  IDs, route and direct simulator positions are unavailable to the policy.
  Imaging of local geometry/frames, tracked clot state and peers within 6 mm
  remains an explicit unverified sensor assumption.
- Method 1 is now the comparable local `single_sequential` baseline. The
  privileged `single_route` baseline is separate. The old teacher prior double
  application is fixed by disabling the environment prior.
- Paired initial scenes, nested rotated start pools and fixed aggregate
  catalytic-rate proxy (0.36/N per cluster) replace the confounded v1 protocol.
  Per-cluster-rate sensitivity remains separate. Magnetic power/material
  matching and calibrated coupling are not provided.
- Measured substep spacing is separate from predictions. Every process attempt
  remains in the denominator. Cluster-safe success requires all clearing,
  <1 cluster-s wall contact, no particle/pair contact, retention of all clusters,
  and no spacing violation. The filter has no safety certificate.
- Broad regression: 92 passed. Latest multi-cluster plus analysis checks: 31
  passed (includes the added reserved-seed gate and four analysis tests).
- Three full preflight episodes: 3/3 complete, no native failure, but one
  realized spacing violation; no cluster-safe success. See the repair report.
- Frozen experiment: 260 requested 180 s episodes; 10 paired diagnostic scenes,
  26 cells, controller/noise seeds 42/43/44. These are not RL training seeds.
  Final paired analysis is written to
  `research/validation/EXP0046_V2_PAIRED_20261004/REPORT.md` after completion.

Repair evidence: `research/validation/EXP0046_REPAIR_20261004/REPORT.md`.
V1 raw data remain intact; no sealed test or training run occurs in this batch.

### EXP0046 v2 final outcome and retrospective diagnosis

- All 260 requested attempts are recorded: **259 completed, one SIGSEGV** in
  `multi_unshielded`, N=3, d=2 mm, fixed-total budget, scene 1302010005, seed 42.
  The native trace ends in the NumPy path-length expression at
  `environments/mca_physical_dynamics.py:307`; this is not a causal diagnosis.
  The predeclared numerical gate failed. No ranking, paired superiority claim
  or sealed result is reported. Scene, source and configuration audits found
  no other mismatch.
- One immediate gdb repeat and a separately recorded 16-attempt stress batch
  (12 plain, four gdb, four workers) all completed; stress outputs share the
  same final-state hash. The failure remains intermittent and unresolved. None
  of these debug runs replaces the failed performance row.
- Descriptive primary d=2 mm cases (30 episodes each, 10 independent scenes):
  local N=1 has 5 raw / 5 cluster-safe completions; parallel N=2 has 9 / 9;
  parallel N=3 has 13 / 5. These are diagnostics, not an admissible ranking.
  Wall-threshold failures are 3 / 6 / 19; spacing violations 0 / 4 / 10.
- Two retrospective N=3 replays preserve the original final-state hashes.
  Scene 1302010003: 224 of 248 wall-contact agent-steps change from a nominal
  outward component <=0.05 to >0.05 after the spacing filter. Scene 1302010005:
  the spacing filter changes no commands and all 165 wall-contact agent-steps
  already command outward. Joint wall/separation constraints and the shared
  local navigation baseline both need attention; action/contact association
  alone is not a causal improvement estimate.
- Final targeted regression and analysis tests: **32 passed**. Failure-aware
  statistics explicitly block comparisons and preserve all-request bounds.

Reports: `research/validation/EXP0046_V2_PAIRED_20261004/REPORT.md`,
`FAILURE_ANALYSIS.md` in that directory, and the repair report. Next gates:
native fault capture; joint local wall/separation handling; stronger local N=1
navigation; fresh paired conventional scheduling comparisons; measured magnetic
coupling and sensing feasibility before physical or learning-contribution claims.

## 2026-10-04 — EXP0047 learned local maneuvers, ideal-sensor diagnostics

- Implemented parameter-shared PPO over ten common local maneuver candidates;
  shared actor/critic receive 138 local features. Heuristics use the identical
  candidate library and approximate joint wall/separation projection.
- Three training seeds (42/43/44) completed 32,768 then 65,536 environment
  steps each. The 32k validation contains 48 heuristic and 36 learned full
  episodes on 12 paired scenes; all 84 attempts completed. 64k is retained
  without evaluation because the sensing audit changed the research contract.
- At 32k, cluster-safe success is 25% for each learned seed and the strong
  joint baseline; the navigation-memory baseline is 16.67% with higher mean
  removal (85.42%) than learning (75.03% across seeds). There is no demonstrated
  learning advantage over the strongest common-filter heuristics.
- Exact velocity, clot mass fraction, true Frenet frame and ideal geometry in
  the common sensor are simulator privileges even when both arms share them.
  EXP0047 is explicitly an ideal-sensor diagnostic, not sim-to-real evidence.
- Results and paired uncertainty: `validation/EXP0047_ANALYSIS_VAL_20261004/`.
  Checkpoints: `training/EXP0047_LOCAL_LEARNING_20261004/`. No confirmation
  scenes were accessed and no negative result was discarded.

## 2026-10-04 — EXP0048 shared measured tracking, active research

- A renderer/processor boundary replaces exact velocities with historical
  tracked motion, exact mass with binary local visual evidence, and true
  Frenet axes with fixed calibrated camera/actuator coordinates. Every learned
  and conventional controller consumes the same delayed, noisy packet.
- Tracking stress settings are engineered assumptions, not measured hardware.
  Synthetic geometry, ideal identity association, camera calibration, and
  independent magnetic actuation remain explicit limitations.
- Preflight 512-step PPO completed. Initial scene 1440000000: the joint process
  exited 139 without a captured stack; its learned counterpart completed with
  no removal. A separate gdb repeat completed with the same zero-removal outcome.
  The original failure remains recorded and is not replaced by the repeat.
  The intermittent native fault remains unresolved.
- Before freezing the main run, stopped using simulator all-clear to terminate.
  Rollouts now continue until measured absence confirmation, all physical exits,
  or 180 s. Primary success requires both physical clearance and observed
  completion; contacts include this confirmation period. Physical active time
  is retained even when tracks are missing.
- Regression: 46 tests passed. Three fresh 32k PPO runs and matched heuristic
  validation are launched. No outcome claim is made while runs are pending.
- Protocol: `experiments/EXP_0048_TRACKED_OBSERVATION_PROTOCOL.md`; training root:
  `training/EXP0048_TRACKED_LEARNING_20261004/`. Frozen-source manifests and all
  failed attempts must be retained; no mixing of EXP0047 and EXP0048 data.

### EXP0048 completed 64k development and EXP0049 follow-up

- Three tracked PPO seeds each completed 65,536 environment steps. The full
  32k/64k comparison plus five conventional arms contains 132/132 completed
  validation attempts on 12 paired scenes, with no numerical failure in this
  matrix. The separate initial preflight exit 139 remains unresolved.
- No tested learned or heuristic arm achieves primary cluster-safe success.
  The strong N=3 memory heuristic removes 72.65% on average. Feedforward PPO
  averages 23.88% at 32k and 13.75% at 64k. Reduced wall contact with lost
  treatment efficacy is not a learning benefit; do not report it as superiority.
- Added the same memory controller for the N=1 baseline so method 1 is not
  represented solely by the weak joint-only navigation controller. Results:
  `validation/EXP0048_FINAL_DEVELOPMENT_20261004/REPORT.md`.
- Stop this feedforward architecture at 64k (registered 128k was a ceiling).
  Preserve all checkpoints, source snapshots and negative results. No method
  is selected for confirmation and no confirmation scene has been accessed.
- EXP0049 implements a causal six-frame GRU and a shared scorer of actual
  projected candidate commands. The hypothesis concerns history/action-outcome
  alignment and candidate-index brittleness; it is not established novelty.
  Reward, sensors, physics, conventional filters and completion rules stay fixed.
- Full preflight: original tracked joint, temporal-wrapper joint, and learned
  512-step pilot complete with bit-identical final-state hashes on scene
  1540000000. Three fresh 32k training seeds are now running. Conventional
  comparators and EXP0048 64k model transfer use the same fresh 1520000000 pool.
  Cross-revision model hashes remain explicit; shared source hashes must match.

### EXP0049 final development result and EXP0050 initialization test

- Three temporal-candidate seeds each completed 65,536 environment steps.
  Main comparison: 132/132 validation attempts completed, plus 36/36 explicit
  old-feedforward transfer attempts on the same fresh scenes. Common source
  and physical-scene audits pass. No primary safe success is observed.
- Mean 64k temporal removal is 13.17%, versus 83.33% for the N=3 memory
  heuristic. Adding short history and executable-candidate scoring does not
  establish a useful advantage. Preserve the negative architecture comparison.
- A truth-only diagnostic on separate scene 1540000000 records mean predicted
  position error 0.0886 mm, mean velocity error 0.7502 mm/s and 687/1554 contact
  agent-steps whose prior observed clearance exceeded the controller margin.
  Instrumentation produces the identical original final-state hash; it never
  feeds these true residuals to a policy. One scene is not a causal explanation.
- Primary per-policy success intervals now retain boundary uncertainty. Zero
  successes in 12 scenes has exact two-sided 95% interval [0, 26.46%]; a paired
  all-zero bootstrap interval does not establish equivalence. Secondary
  intervals remain exploratory, without multiplicity correction.
- EXP0050 keeps the same sensing, reward and candidate projection but sets the
  deterministic untrained actor equal to the strongest common memory control.
  A learned scorer can intervene; initialization gains are not attributed to
  learning. Three preflight runs (original memory, new memory, untrained actor)
  are bit-identical on scene 1640000000. Three fresh 32k pilots and five
  matched conventional arms are running, followed by registered 64k evaluation.

### EXP0050 final result: recorded numerical failure, no superiority claim

- All three memory-initialized PPO seeds completed 65,536 training steps.
  Validation records 132 requested attempts: 131 completed and one SIGSEGV
  (seed 44 / 64k / scene 1620000011). The combined numerical gate is blocked;
  no failed row is replaced. A separate gdb replay completed without reproducing
  the crash. The Python frame is the physical geodesic-contact callback, not a
  proven native root cause and not a policy information channel.
- At 32k, seed 43 shows higher clearing and less wall contact than memory, but
  more spacing violation; this isolated signal does not repeat across seeds.
  Seed 44's 12 final states exactly match the unchanged memory controller, so
  its baseline-level outcome is not a learning contribution. Completed 64k
  attempts again show efficacy collapse. No favorable seed is promoted.
- Keep source/checkpoint snapshots and all negative/failed attempts. No
  confirmation set has been opened and no sim-to-real or journal-success claim
  is established. See `validation/EXP0050_FINAL_DEVELOPMENT_20261004/` and
  its `FAILURE_ANALYSIS.md`.
- The next justified gates are numerical fault capture, causal measurement
  timing calibration, and reward/efficacy alignment under matched heuristic
  inputs. Generic network enlargement and reduced treatment activity are not
  substitutes for evidence of safe clearing.


## 2026-10-04 — EXP0051 measured hierarchical options registered and training

User requested hierarchical learning with fixed paired layouts, no heuristic navigation privileges, and physically evaluated separation. The old hierarchical allocator reads true routes/mass/stations and is not used. A new observation-only option scheduler uses measured target identity, memory navigation, drift-compensated hold, and retreat above a frozen 10 Hz controller. The joint supervisor is common to learned and strong conventional arms and has no safety certificate.

Preflight: 12/12 processes completed across two paired development scenes, actual reset snapshots match within N, and untrained deterministic policy exactly reproduces memory trajectories. Spacing violations remain, so preflight audit PASS is provenance/numerical success, not a safety claim. Preflight source snapshot was preserved before documented training-only changes to actor identity encoding/exploration.

Started six preregistered training runs: hierarchical and flat options, seeds 42/43/44, 16,384 physical control steps each; all use the same normalized reward. Planned main validation: 144 attempts on 12 fixed scenes. Planned synthetic coupling stress: 40 attempts on four additional fixed scenes. No source changes to EXP0048–50; no confirmation pool is opened. Primary sources checked: Option-Critic (1609.05140), HIRO (1805.08296), CPO (1705.10528); these algorithms are not claimed to be reproduced or novel.


## 2026-10-04 — EXP0051 completed; EXP0052 registered and started

EXP0051 completed six 16,384-step training runs (98,304 physical controls), 144/144 main evaluations and 40/40 synthetic coupling evaluations, in addition to 12/12 preflight evaluations. No new process failure in these matrices. All six learned deterministic policies exactly reproduced memory on all 12 main scenes; mean removal 73.745%, versus 83.333% for stronger measured balanced assignment. No learning gain. Original native failures remain unresolved, not erased.

A new shadow estimator subtracts logged requested commands before estimating residual drift, then integrates commands over detection latency. Four replays over two preflight scenes (memory and priority controllers) preserved the exact reference final state. Mean position errors fell from about 0.074–0.076 mm to 0.041–0.043 mm; mean reported-velocity errors from 0.553–0.584 to 0.097–0.122 mm/s. Same images/commands, truth only for scoring. These are four trajectory repeats over TWO scenes, not four independent layouts or a learning/hardware result.

EXP0052 gives this observer to every arm, extends hierarchical target commitment to 5 seconds, and initializes at measured balanced assignment while learning corrections. Baselines retain 0.1/1/5-second allocation, priority and N=1 memory. Preflight 16/16 completed with exact untrained/balanced equivalence; all multi-cluster arms cleared both preflight scenes with no spacing violation but nonzero wall contact, so no primary safe success. Six fixed-budget learning runs now train 32,768 controls each. Main fixed scene pool: 1820000000–1820000011; synthetic stress: 1850000000–1850000003; confirmation remains unopened.


## 2026-10-04 — hierarchical studies and all supplementary diagnostics completed

EXP0052: 6 × 32768 control steps, 168/168 main evaluations and 56/56 registered synthetic-stress evaluations completed. Main removal: hierarchical three-seed mean 83.5738%, flat 87.5253%, strongest-clearing 1-second allocation heuristic 89.6592%. Learned primary safe success is 0/12 in every seed; priority heuristic has 1/12. No learning superiority. All main learned spacing checks pass, but this does not generalize to the extra layouts.

Post-hoc paired observer ablation: 12/12 old-tracker attempts on the same main layouts; mean spacing exposure 0.153342 -> 0 pair-seconds, clearing 86.8061% -> 86.3515%. Other sources/configurations matched; separate first-scene renewal-interval replay had the identical final state. This is shared engineering evidence, not learning.

Post-hoc same-layout coupling control: 56/56 uncoupled attempts matched the original 56 stress attempts. Hierarchical seeds 43/44 have 0/4 spacing failures without coupling versus 1/4 and 2/4 with synthetic alpha=0.1; minimum 1.467540 mm. Flat seed 42 fails spacing on one uncoupled extra layout and zero coupled layouts, so the effect is not uniformly adverse. Four independent layouts only; no physical magnetic calibration.

This continuation completed 12 training runs / 294912 physical steps; 408 registered main/stress + 28 preflight + 68 paired component controls + 6 diagnostic replays = 510 evaluations/replays, not 510 independent layouts. 81 targeted tests pass; all frozen source/checkpoint checks and confirmation-access audit pass. All scheduled jobs finished. Prior native faults are retained and not declared solved. Full results: research/STATUS_HIERARCHICAL_20261004.md.

## 2026-10-04 — EXP0053 local TPG and upper/lower learning implemented; numerical gate failed

User approved TPG with separate 3-D Euclidean interference checks and non-Euclidean vessel travel, and explicitly requested low-level learning as well. Implemented measured-crop Dijkstra with unknown-route sentinels and ambiguous branch attachment, continuous 3-D polyline separation, persistent acyclic local reservations, event-only high scheduling, graph priority PPO, and causal GRU residual control with a next-measured-velocity auxiliary loss. Unknown global travel is never replaced with a claimed true geodesic. This is a rolling local TPG prototype, not full globally feasible MAPF or a magnetic safety certificate. Shared graph/trigger/supervisor mechanics are not learning contributions.

The direct graph-following nominal controller failed development preflight with severe wall exposure. Preserve V1 snapshots; V2 restores strong measured-memory nominal control, V3 supplies the classical controller with the same proactive retreat and reservation constraints. During the first pilot, a causal audit found image-level freshness was too weak for reservation release. Stop before any validation: three high runs completed 8192 steps each, and three low runs were interrupted at 2034/1180/1040 steps. Preserve all sources, checkpoints and interruption metadata. Repair release and auxiliary targets to require individual fresh detections.

Corrected version: 46 targeted tests pass. Default V4 full preflight completed 3/4; one SIGILL (-4), retaining the failure. Corrected high/low training checks each completed 2048 steps with the other branch unchanged. Both completed a 1024-step update before SIGSEGV. Debugger evaluation replay finished normally but does not replace the failure; debugger training replay captured the native SIGSEGV in NumPy `_multiarray_umath`, with `__svml_dcbrt_cout_rare_internal` / `array_subtract` frames. Root cause is not proven, and a Python stack location alone does not establish its origin.

Four preregistered CPU-dispatch diagnostic repetitions: default 1/2 completed, AVX2/FMA3 disabled 2/2 completed. A subsequent complete preflight with the disabled features still completed only 3/4 (another SIGSEGV), so this mitigation is rejected as a passed admission gate. Runtime flags apply only to child processes; packages/global environment unchanged. No 52-row main matrix is started, no efficacy ranking admitted, no failed row replaced, no best seed promoted. Main validation and confirmation pools remain unopened.

Final machine audit confirms all 23/29 frozen source hashes of EXP0051/52 remain unchanged, all 40 new confirmation seeds are guarded, failed preflight blocks the runtime launcher, and no research processes remain running. See `research/STATUS_TPG_20261004.md`, `research/validation/EXP0053_FINAL_AUDIT_20261004/FINAL_AUDIT.json`, and the preserved native/runtime diagnostic directories. Next prerequisite is a minimal native reproduction and validated common runtime, not another favorable-seed search.

## 2026-10-04 — Verification status and prospective ablation/comparison plan

Rechecked the final EXP0053 audit, runners, paired scene factory, reward and metric definitions in response to the user's verification question. Added `research/experiments/VCTPG_EVALUATION_PLAN_20261004.md` and the prospective 52-row `VCTPG_PILOT_MATRIX_20261004.json`. These are planning artifacts only: no new training/evaluation ran, no reserved scene was reset, no performance advantage was established, and frozen executable/configuration files were not changed.

The plan separates the implemented high/low 2x2 factorial from proposed history, auxiliary loss, graph encoder, timing and reservation ablations; distinguishes same-interface rule controls from legacy strong heuristics and N=1 resource-proxy comparisons; records missing current-interface flat/planning baselines and missing anatomy/N>3 support. It specifies paired layouts, seed/scene uncertainty, failed-attempt retention, censored completion times, source/runtime admission and confirmation freeze.

Clarified two easily overstated claims: registered `cluster_safe_success` permits <1.0 cluster-second wall contact rather than requiring exact zero; deployment navigation excludes true outcomes, while simulator-truth training rewards and independent scoring remain. An observation-boundary counterfactual leakage audit, full confirmation statistics and hardware magnetic calibration are still prospective, not completed validations.

## 2026-10-04 17:10 Asia/Shanghai — Isolated generic NumPy runtime admitted; paired pilot started

User requested actual experiments and then asked how the two previously supplied Nature papers chose comparators. Read archived Medany full text and Turbo supplement, retrieved publisher Fig. 3/Fig. 2 pages and saved URL/hash provenance. Added `VCTPG_BASELINE_SELECTION_20261004.md`: Turbo compares GTrXL/GRU/MLP, domain-randomization/reward sensitivity and human tasks; Medany compares DreamerV3 with tuned PPO and reward/frame-skip/training-ratio settings. Proposed adapted recurrent/Transformer/multi-agent controls are not claimed implemented; frozen EXP0053 pilot stays unchanged.

Checked installed NumPy/SciPy native files against RECORD: 20/116 files, no mismatch. An additional default-runtime gdb replay of the 2048-step dual-level check completed normally; debugger exit 1 was the post-exit thread query, not an inferior crash. This does not overwrite prior failures. Preregistered a same-version NumPy 1.26.4 source-build experiment without SVML/CPU optimization and external BLAS/LAPACK in an isolated system-site-packages venv. Original environment and frozen algorithm files remain unchanged. The first build invocation failed because ensurepip's pip lacked config-settings; preserved the log, updated pip only inside the isolated venv, and built with supported options. The setup failure is not counted as a successful experiment.

Candidate passed 20,000 numerical checks, Torch/NumPy interoperation, a superset of 83 targeted tests, all 3 registered dual-level 2048-step regression runs, and 8/8 registered preflight evaluations in two batches. Untrained/rule and between-batch initial/final hashes match. Root cause of the original native failures remains unproven; this is a declared alternative-runtime admission, not erasure of old failures.

Started the complete corrected-source 9-run / 73,728-control-step / 52-evaluation pilot at `research/validation/EXP0053_TPG_GENERIC_STUDY_20261004`. Coordinator `scripts/run_tpg_generic_runtime.py` preserves requests, return codes, source and runtime provenance and stops admission/ranking on any failed requested run. Confirmation remains guarded. No performance conclusion yet; current state is in `EXP0053_RUNTIME_REPAIR_20261004/generic_coordinator_status.json`.

## 2026-10-04 17:16 Asia/Shanghai — EXP0053 full pilot completed; no learned behavioral gain

All 9/9 corrected-source training runs and 52/52 evaluations completed under the declared generic NumPy runtime. Source, checkpoint, exact within-runtime initial pairing and untrained identity checks pass. Each learned checkpoint matches the rule controller's final-state hashes on all four development scenes. Learned/rule mean clearing is 87.50%, AUC 72.59%, wall exposure 31.797 cluster-s, versus balanced_1s 93.75%, 71.23%, 51.145 cluster-s. Every primary safe-success count is zero. Lower event scheduling frequency belongs to the shared rule mechanism, not learning. All seeds retained; 4 scenes remain 4 independent scenes.

New plotting script produces the per-seed comparison figure only after an admitted complete matrix. Baseline selection note records why recurrent PPO, adapted Transformer-PPO and fair multi-agent controls should supplement this pilot, and why N=1 learned control is also needed for a clean learning-versus-parallelism claim. These extra baselines have not been run.

Cross-runtime audit finds identical accepted seeds but tiny initial coordinate differences (max about 5.7e-9 / 2.1e-8 mm in two preflight scenes) and different long-run trajectories. Within-runtime repeated preflights are exact. New results are conditional on the declared build and cannot be pooled with old runtime results as identical replications. Original native failures are preserved and not declared fixed. A separate measured-state shadow audit has been started to distinguish weight updates from changed greedy decisions; it cannot replace registered results or select a favorable checkpoint.

Post-hoc shadow audit completed on all four rule trajectories with exact initial/final replay agreement. Each of six low-level checkpoints chose the rule candidate on all 13,010 active measured states, including 12,596 with multiple valid actions. Score-head weights changed, but greedy choices did not. High-only seed44 differed at one of 50 priority events; other high checkpoints agreed. This does not alter the registered no-gain result. Actual high training sample counts are only 39–113 per run, despite 8192 physical controls. Added the diagnostic report and a final provenance/resource audit; N=1 shares the same scene/start subset and aggregate catalytic-rate proxy, with no material/power equivalence claim. Main matrix and diagnostic jobs have completed; confirmation remains unopened.

## 2026-10-04 — User correction: cooperative MARL becomes the primary algorithm comparison

The user correctly challenged prioritizing ordinary joint PPO/Transformer baselines over multi-agent methods. Revised `VCTPG_BASELINE_SELECTION_20261004.md` and `VCTPG_EVALUATION_PLAN_20261004.md`: prioritize measurement-only recurrent MAPPO and IPPO interface adaptation, then MAT as a strong joint-coordination comparator. Ordinary joint recurrent PPO and Turbo-inspired architectures are auxiliary structure controls. Keep strong measured coordination, N=1 sequential treatment, and the high/low 2x2 ablation. Verified primary MAPPO, IPPO and MAT papers; no claim that these are the latest algorithms or have been reproduced here.

Added separate actor/critic and communication audits, the current nine discrete low-level candidates plus coordination-capability requirement, and distinct system-level versus common-TPG module comparisons. The current upper controller uses a team measurement graph and is not automatically decentralized MARL. If all agents receive full team histories, IPPO/MAPPO information conditions may collapse; do not manufacture a distinction by withholding observations from a baseline. A decentralized-information experiment must also restrict the proposed method. Training rewards and independent scoring still use simulator truth; no navigation-truth claim is broadened to whole-pipeline absence of truth.

Synchronized stale evaluation-plan status with the already completed generic-runtime pilot. This correction changes documentation only: no new algorithm implementation, training, evaluation, performance result, frozen EXP0053 source/configuration edit or confirmation-scene access. The completed 52-evaluation pilot contains no MAPPO/IPPO/MAT and establishes no learning advantage.

## 2026-10-04 — EXP0054 registered: measured MARL and action-conditioned learning

Implemented communication-measured recurrent MAPPO/IPPO adaptations with per-agent PPO ratios, recurrent individual or pooled measured-history value estimates, and event-only learned bids for admissible priorities. Implemented V-CTPG graph priority with candidate-conditioned next-measured-velocity input to the low policy, and a no-action-condition ablation. All share target allocation, event triggers, TPG, candidates, reward, sensing and projection. These are common-TPG module comparisons, not yet independent end-to-end planner baselines; MAT remains prospective. The current team packet is shared by every actor; only own temporal history reaches each local action score. Central critics additionally pool measured histories. No navigation truth access is introduced; truth-based training reward and scoring remain.

42 relevant tests passed, including possible priority orders, nonpreemption, MAPPO/IPPO identical actor initialization and distinct critic history access, masked per-agent updates, GAE duration/termination, action-conditioning ablation, controlled truth-container perturbation and confirmation guards. This boundary test is not a full sensor-to-hardware proof. Original 35 frozen EXP0053 source hashes are unchanged.

Prospective study: four variants × three seeds × 32768 physical controls; all 8192 and 32768 checkpoints evaluated on the same eight development scenes 1960000000–1960000007, plus shared TPG rule, strong 1-second balanced heuristic and N=1. 216 main evaluations, 10 full-duration identity preflights and four 2048-step learning regressions; run admission blocks on any failed process or identity. Original and new confirmation pools remain guarded. No efficacy result yet; no best seed/checkpoint selection.

## 2026-10-04 — EXP0055 conditional reward-control follow-up preregistered

Before viewing the completed EXP0054 main matrix, registered a conditional follow-up if the proposed method has zero safe successes, spacing violations, or no AUC gain over the strongest classical comparator. The fixed original reward trades 27 cluster-seconds of wall contact for one percentage point of clearing; the safety objective tests a 1-second exchange, without changing evaluation thresholds. This is a training remedy, not algorithmic novelty or a physically certified tradeoff.

All four algorithms and all three seeds get two matched fine-tuning regimes from their own fixed 32768-step parents: 16384 more controls under the original reward versus the same budget under the safety reward. Adam restarts in both arms; no exact-resume claim. Planned 24 fine-tunes / 393216 additional controls, 16 parent-policy identity evaluations and 192 main paired evaluations. Original and new reward settings both retain every seed; eight already-designated development scenes reused, all confirmation pools stay closed. Parent results must be complete and admitted before starting.

Two new adapter tests plus the 16 measured-MARL tests passed (18 tests), including exact paired physical trajectory under reward-only changes and checked parent-weight initialization. The original 42-test suite remains passed; these are 44 unique relevant tests, not 60 independent tests. All frozen EXP0054 sources remain unchanged.

A registry-only serialization attempt encountered an existing malformed legacy CSV row. Recovered the exact committed prefix and all eight previously read EXP0046–53 raw rows, then added only the new EXP0054 entry. Recovery artifact retained under EXP0054_REGISTRY_RECOVERY_20261004; subsequent registry edits are scoped to individual rows. No source, training or outcome artifact was affected.

EXP0055 pre-run refinement: both reward regimes now separately clip actor/critic gradients at 0.5, with unclipped norms logged. This shared optimization correction avoids critic-scale-dependent policy clipping; it is not a claimed novel algorithm or proven sole cause of the old no-gain result. Added an exact actor-gradient invariance test; 19 adapter/MARL tests pass (45 unique relevant tests including the original 42-test set). No EXP0054 frozen source changed. Parent-to-child gains will not be attributed solely to the reward change.

## 2026-10-04 — EXP0054 completed negative; preregistered EXP0055 triggered

Completed 12/12 training runs (393216 physical controls), 216/216 main evaluations, 10 identity evaluations and four 2048-step regressions. All processes succeeded and source/runtime/actual-initial-state pairing gates pass. Independent safety-definition and N=1 catalytic-rate/start/particles/targets/mass audit also passes. All learning arms and TPG have zero measured spacing exposure in this development pool, but every registered safe-success count remains zero. Strong balanced_1s has one spacing failure in eight scenes.

At the preregistered final32768 checkpoint: vctpg_ac clearing47.5984%, AUC40.0145%, wall96.689 cluster-s; no-ac72.9167%,60.1921%,97.758; MAPPO75.00%,61.8101%,63.376; IPPO73.9583%,62.3299%,27.534. RuleTPG78.125%,64.3099%,24.571; balanced79.0282%,63.0568%,34.044; N1 40.625%,25.7550%,20.134. No learning advantage. All earlier8192 checkpoints retained without substituting their better results. Proposed learner changes 17/24 final-state hashes versus rule; behavior changed, outcomes worsened. Figures include every seed; corrected wall-axis range to include the worst seed rather than clipping its point.

EXP0055 automatically triggered by zero safe success and no AUC gain. Sixteen parent-policy identity evaluations pass, with identical physical trajectories across reward regimes. All 24 matched fine-tunes now run from their own final32768 parents under original/safety reward,16384 additional controls each, fresh Adam and separate actor/critic clipping shared by all algorithms. No confirmation access, no weakening baselines, no current follow-up efficacy claim.

## 2026-10-04 — EXP0055 native gate failed; diagnostic repetitions retained separately

All24 matched fine-tunes completed (393216 additional physical controls), and16/16 parent-policy identity evaluations passed. Main evaluation191/192 completed; the final safety_reward/vctpg_no_ac/seed44/scene1960000007 process exitedSIGSEGV(-11). The Python stack points to NumPy clip called by measured_tpg.segment_distance; it does not establish the native root cause. This is a new failure under the previously admitted generic same-version NumPy build. Root cause remains unresolved; do not continue calling this runtime a proven fix.

All232 requested EXP0055 jobs have completion records. The191 completed rows have unique expected keys, exact paired initial/scenario hashes, and unchanged frozen sources. The original failed row and empty output remain. Wrote AUDIT.json with valid_comparison_gate=false and INCOMPLETE_MATRIX_AUDIT.json; no partial-data performance ranking or success claim is admitted. No confirmation pool opened.

Preregistered exactly3 diagnostic replays in a separate folder after the failure: one gdb and two serial plain replays. All3 completed normally with exact matching initial/final hashes; gdb inferior exited normally. A separate20000-case seeded scalar clip / segment-distance symmetry stress passed (max symmetry error8.88e-16). None replaces the failed denominator or proves intermittent failure repaired. No further best-seed/best-retry selection performed. All research/diagnostic processes finished.

This continuation completed36 planned training runs totaling786432 physical controls, plus4 admission regressions totaling8192 controls. Main evaluation totals:216/216 EXP0054 and191/192 EXP0055, plus26/26 initialization evaluations and3 post-hoc diagnostic replays. These are repeated evaluations on eight main development scenes, not hundreds of independent layouts. EXP0054 is an admitted negative algorithm result; EXP0055 is numerically incomplete and cannot rank the reward repair.

## 2026-10-04 EXP0056 预登记

根据用户要求更换失败方法。取消低层速度预测，增加共享有界残差和测量风险加权 KL；以图/竞价高层与有/无 KL 做四组匹配消融。从零训练三种子，32768步/组。协议见 experiments/EXP0056_CONSERVATIVE_RESIDUAL_PROTOCOL.md。25项相关测试通过；旧原生崩溃未解决，每项请求一次，失败不得补行或排名。

### EXP0056 完成但数值准入失败

12/12训练、119/120主评估；memory_n1_1960000007在environments/mca_physical_env.py:995–996接触回调SIGSEGV。全部请求只执行一次；不补行，不排名。47个源文件哈希不变。准备隔离系统Python3.12与官方CPU轮子验证；原生根因尚未确定，不把环境更换当作已经修复。

## EXP0057 冻结策略清洁运行时部署对比预登记

对全部12个EXP0056最终权重统一迁移至系统Python3.12隔离环境，官方NumPy1.26.4/SciPy1.11.4/Torch2.3.0+cpu。不是新训练，不改变动作/奖励/网络，不按旧不完整结果挑选策略。新运行时10个全时长身份预检、完整120行开发评估，实际初态在新运行时内匹配。原矩阵保持失败；不混用新旧浮点轨迹；单次完整预算，任何失败继续保留并阻止排名。清洁依赖不是原生根因修复证明。

### EXP0057完整负结果；EXP0058连续动作预登记

EXP0057 120/120通过，51测试、130重算初态哈希、12父训练溯源通过。graph_anchor清除71.875% vs MAPPO79.1667%，两个anchor组全部低层残差为0；不保留离散KL为有效贡献。连续残差高斯PPO现在实现为四组共享动作权限，小幅mean变化可直接执行；blocked状态只训练价值不训练无效动作，仍是测量输入。EXP0058固定12×16384控制步，120评估，三种子，先测试/预检再训练；比上一轮预算短，是独立开发试验而非直接学习曲线比较。

### EXP0058 完整结果

12/12训练、120/120评估、10/10身份预检、4/4回归通过。连续低层全部产生实际修正；graph_anchor清除73.8645%、AUC59.9728%、壁面22.5920集群秒、安全1/24；MAPPO80.2083%、63.0972%、28.7496、安全2/24。壁面均值降低21.4178%，但清除率低6.3438个百分点；开发重采样区间跨零，不是整体或稳定优势。传统基线24/24初末状态哈希及指标与EXP0057精确相同。继续保留全部负结果，不升级图/锚定模块为有效创新；后续应诊断高层学习权限/任务分配与拥塞恢复。所有本轮运行已结束。

### EXP0059 高层隔离实验完成

固定EXP0058连续低层，交叉部署规则/MAPPO bid/graph PPO/graph anchor高层，4×2×3×8=192条，全部通过。固定r_mappo低层时四类高层清除率均80.2083%；固定graph_ppo低层时前三者75.0%，graph anchor73.9583%。因此EXP0058图方法落后MAPPO主要跟随低层策略，而非高层图排序本身。高层只改变少数壁面/AUC，未改善清除或安全。交叉加载是接口诊断，不是端到端重新训练比较。

## 2026-10-05 — Unified benchmark v1: baselines first, then motivated improvements (branch research/fair-partial-observation)

Report: research/validation/BENCHMARK_V1_REPORT_20261005.md (development split only; test split unopened).
- Tasks N=1 sequential / N=2,3 parallel (2 mm spacing), 14 anatomies × 30 fixed dev scenes, shared pre-operative allocation A, Safe success primary plus clearance, T50/T90/T100, AUC, path, wall, particle, spacing, deadlock, timeout.
- Baselines (Safe, N=1/2/3): roadmap route_follow 78.1/87.1/79.5; fair local_follow 39.8/57.9/68.8; old reactive 17–44; BC graph student 37.4/52.4/56.4; BC local 28.3/38.8/43.6; local + DAgger r1 41.7/52.4/52.4 (DAgger vs step-matched BC +10–12 pp, CI excludes 0).
- I0 edge-level junction-aware following (motivation: station-level following oscillates at bifurcations and cannot enter acute side branches): N=1 privileged clears 10/10 vs 0/10.
- I1 frontier tabu memory (motivation: 14/15 local failures were limit cycles): Safe +25.2 / +20.7 / +6.4 pp vs local_follow (N=1/2/3), held-out +20.7 / +21.3 / +2.0.
- I2 learned frontier selector (privileged geodesic label at training, fair features at inference): decision accuracy 98.0 vs 83.1 %, closed-loop Safe tied with I1; N=2 T100 −10.7 s, path −21 mm.
- Negative: tube/backoff/hold shields, conflict-aware allocation, visit-count memory. All kept.
- Pure RL (parameter-shared PPO from scratch, same fair observation + target slot, 60 min / 26.8 M agent steps, 1 seed): Safe 0 % at every N, removal 16.6 / 23.3 / 29.9 %. Training removal still rising (~45 %), so this is a budget-limited result.
- I2 three selector seeds, Safe N=1/2/3: 66.2/67.6/68.3, 78.6/78.6/78.3, 75.7/76.4/75.2 (I1: 65.0, 78.6, 75.2).

## 2026-10-05 noon — final comparison vs the previous strongest traditional controller
- plan_route (allocation A + previous station-level route+avoid+wait teacher) on the unified benchmark: Safe 71.0/77.6/71.2.
- I0 edge-level junction-aware follower: 78.1/87.1/79.5; paired +7.1/+9.5/+8.3 pp (N=3 CI excludes 0), T100 −27 s (N=2) / −20 s (N=3), path −51/−53 mm (CIs exclude 0).
- I1/I2 with local-only information match plan_route (which has the roadmap): +1.0/+4.5 pp at N=2/3, −5 to −6 at N=1, CIs cross 0.
- Follower v2 (committed path + max-margin crossing) fixed 27/56 offline stuck cases but lost 15–20 pp in the full benchmark: reverted, recorded as negative.
- TEST SET (one evaluation, frozen b2e6dcf, 1,400 scenes per N): I0 Safe 78.6/88.9/77.9 vs previous strongest traditional 71.9/80.1/71.4; held-out anatomies 78.0/91.6/81.2 vs 61.2/74.0/68.6. Scene-level sign test p = 8e-6 / 2e-13 / 2e-7; anatomy-clustered bootstrap +6.6/+8.8/+6.5 pp (lower bounds −10.0/−0.7/−0.6). T100 −26 s (N=2) / −21 s (N=3), path −48 / −51 mm, CIs exclude 0.

## 2026-10-05 afternoon — Codex HRL port, TPG spacing coordinator, deployable information model

- Codex EXP0058 hierarchical RL ported onto the unified benchmark (same scenes/physics/anatomies/horizon; Codex's own measured sensors and allocation; its networks only support N=3). 14 anatomies × 30 dev scenes: V-CTPG-HRL 4.1 % safe, MAPPO 4.6 %, rule TPG 4.8 % (N=3), rule TPG N=2 3.7 %; removal 83–84 %; wall contact ~27 s per episode is the dominant failure. Not an equal-information comparison.
- I3 spacing-safe TPG coordinator (precomputed 3-D conflict zones on the roadmap, prioritised timed schedule, hold-before-zone, park finished clusters; replaces sideways deflection): N=3 safe 85.5 vs 79.5 (+6.0 pp [−5.2, +15.5]), wall contact −5.6 s [−10.5, −2.1], spacing violations and particle contacts 0; but timeouts 9.8 vs 1.9 %. N=2 −2.1 pp (n.s.). Timeouts come from conservative waiting.
- Deployable information model (marl/deployable_sensing.py): imaging estimates only (noise 0.05 mm, 1-frame latency, dropout, map matching, local particle detection, finite-difference velocity). I0+TPG on 10 MCA scenes collapses (N=1 0/10, N=3 2/10 safe) with 30–290 s wall contact.
  Root cause (verified by trace): near a side-branch take-off the simulator confines a body to the daughter edge's tube while its 3-D position is also inside the parent lumen; the transport switches edges only on axial end crossings. Which tube constrains the body is therefore not a function of its position, so no position-based estimator can know it. Lumen-normalised and stall/inconsistency-triggered map matching did not fix it. This is a simulator artefact (real lumens at a junction are a union of tubes), not a sensing limitation.

## 2026-10-05 evening — union-of-tubes junction model (simulator realism fix)

- New DynamicsConfig.junction_model='union' (default 'graph' keeps every earlier result reproducible). The accessible region near a body is the union of the capsules of every edge within two junction hops; feasibility depends only on 3-D position; side branches open where their tube pierces the parent wall; the membership edge is the containing capsule with the smallest radial/radius ratio. Implemented identically in the reference and compiled backends (tests/test_union_junction.py: backend agreement on two anatomies; an axis-following body enters the MCA perforator only under 'union').
- Stopped the 150-min PPO runs at ~60 min (physics superseded); partial logs kept.
- Deployable classical pursuit (pre-operative route + imaging estimates only) under 'union', MCA 10 dev scenes: N=1 4/10 safe (9/10 cleared), N=3 3/10 safe (10/10 cleared) with the TPG spacing coordinator (0 spacing violations; without TPG 2/10 with 2 violations). Remaining failures are wall contact of 1–3 robot-s, i.e. just above the 1 s threshold, in perforators whose clearance (~0.06 mm) is comparable to the 0.05 mm localisation error. A clearance-proportional slow-down variant was much worse (stalls against the flow): negative, recorded.
