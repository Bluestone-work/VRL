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
