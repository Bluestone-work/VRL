# Innovation tracker

EXP_0015 已注册但尚未形成性能结论：`geometric_dynamic` 只提供当前最近动态障碍物的局部状态，
`risk_aware_connectivity` 只改变血栓目标分配并记录当前路线风险；动作仍是 Direct Local
Frenet policy 输出。路径长度采用所有机器人实际位移之和，同时报告每单位清除质量的路径代价。
85% 是预注册验收门槛，不满足时必须报告失败，不能通过测试集选参或修改障碍物轨迹来宣称达到。

EXP_0015 已完成 3 seed × 1M transitions、420 回合独立动态验证：成功率
53.33% ± 19.68pp，质量清除 87.27% ± 7.51pp，路径长度 7.6163 ± 0.8020，
路径/清除质量 3.2220 ± 0.2637。85% gate 失败；seed 44 的 73.6% 不能替代三 seed
均值。该结果保留为真实的研究候选，不声称最优，也不跳过后续 observation/scheduler
消融或动态基线对照。

EXP_0016 已注册并运行：在 EXP_0015 基础上加入当前状态的 0.45 步常速度预测、
TTC/预测间隙/碰撞概率/漂移不确定性特征，以及 `predictive_risk_connectivity`
调度。训练早期没有数值异常；最终性能尚未验证，85% gate 仍然有效。

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

EXP_0014：实现并完成三 seed × 1M transitions 的 `adaptive_edge_gat` 对照。每个
通信边的相对位置/速度特征同时产生每头 attention bias 和 sigmoid value-message
gate；Direct Local 动作、观测、奖励与物理保持不变。prospective validation 相对
EXP_0005 的成功率 +3.33pp、质量清除 +3.32pp、wall contact -11.79pp、pair collision
-3.84pp。该结果支持“自适应通信门控值得继续研究”的工程/实验假设，但不等于已完成
顶会文献新颖性验证；仍需多次独立复现、gate/edge-feature 消融和文献核查。

EXP_0001已完成：3seed×1M transitions；原协议成功72.8571%±1.8898pp、清除92.7773%±0.6408pp，与归档指标和actor/critic张量一致；额外840回合开发验证成功74.6429%±0.3571pp。详见[BASELINE_REPORT](BASELINE_REPORT.md)。

EXP_0001分类：Engineering Improvement（可追溯基线与只读测量）。Algorithmic Improvement：无。Scientific Contribution：固定模拟器和本机条件下的描述性复现证据、跨设备及场景限制。Potential Novel Contribution：无新增主张。Needs Literature Verification：I1–I7全部。

F004揭示部分场景消融停滞与高wall contact并存，值得后续检验可控性/分配/接触因素；这不支持I1–I7的性能贡献，也不证明world model能修复。H1–H5的证据状态没有改变。

EXP_0002已完成：同权重840对CPU/GPU回合出现10个success翻转，GPU−CPU成功率+0.9524pp但清除−0.0414pp；84对同设备重复一致。此结果约束未来算法对照必须固定设备。新增完整geometry身份与未来互斥清单属于Engineering Improvement；Scientific Contribution限本机测量敏感性证据；Algorithmic Improvement无，Potential Novel Contribution无。I1–I7无新增性能或消融证据，H1–H5仍未验证。Literature verification required.

EXP16–20 v2（2026-09-27 15:07 完成）：arm 17（六帧历史 GRU 位移预测）在 2M 达 76.7%±4.4、3M 73.8%±3.3，为批次最佳；arm 16（CV 预测+adaptive-edge）67–71%；arm 18（八候选动作+解析风险评分）与 arm 20（+世界模型规划）崩溃至 0.7–4%，失败根源定位在 18 的控制器架构（每步 ~1500 次干预、coronary 场景 wall_rate 0.84），世界模型门槛三项全过且 20 的 model_trusted ~1499/1500 步仍与 18 同水平，排除世界模型为失败原因。85% 开发验证阈值未达到，负结果按协议保留。尾迹取证（同日）：61% 失败=最后一栓拖尾，机器人距存活栓 0.05–0.12 原地漂 227 步——收尾行为没学会，是训练密度与终止信号问题，非观测/物理问题。

EXP_0010–0013（2026-09-27 17:47 prospective validation 完成）：variable-N 训练 +5.24pp（方差减半）；粒子训练+评估 +7.4~+12.0pp（16 最佳 81.0%，训练/评估贡献未分离，confounded）；separated-init −18.10pp、每步重规划 connectivity allocator −9.76pp 为保留负结果。全部为 validation 证据，非 test 结论。

2026-10-03 EXP_TEACHER_BC: topology-aware Graph Transformer student distilled from the route+avoid+wait teacher, no route controller at inference. Diagnostic: 97.8% train / 91.0% held-out anatomies vs teacher 98.9 / 96.0 and pure RL 21–33 / 16–34. Category: algorithmic (imitation with topology graph); not yet validated on sealed test, single seed. DAgger round 1: null on completion.

EXP0046（2026-10-04）登记多集群并行清栓：方法 1 为 `N=1` 顺序 route+avoid+wait
baseline，方法 2 为 `N=2,3` 的局部观测并行控制与最小间距让行协议，`d_min={1,2,4}`
mm。当前属于 development feasibility；短回合可运行，长回合在参考积分器的边界退出处仍会
触发 native crash，不能报告正式性能。最小间距目前只是未标定的控制代理，尚无磁场耦合模型，
也没有硬件或临床结论。见 `research/experiments/EXP_0046_MULTICLUSTER_PROTOCOL.md`。

EXP0046 revision 2 更正（2026-10-04）：上段“边界退出处触发”的因果归因没有证据支持，
仅保留为历史记录。未修改源码的 debugger/完整回合复现未再触发故障，native 根因仍未查明。
方法 1 改为同观测的局部顺序 baseline，privileged route baseline 单列；方法 2 为局部观测
反应导航与启发式间距过滤。已修复双重 prior、非配对环境、资源预算混杂、真实/预测间距
混用与失败分母问题。当前分类为 Engineering Improvement；尚无新增学习算法或物理可控性
贡献。260 回合的配对诊断用于判断下一步假设，不是已验证的新颖性或期刊接受概率证据。

EXP0046 v2 最终状态：259/260 完成，1 次 SIGSEGV；预先规定的数值门槛未通过，
不作方法排名。17 次后续诊断重跑未复现，也不等于修复。保持原始失败分母。
局部回放发现两类待解决症状：间距过滤可引入朝壁指令，基础导航也可在没有过滤干预时贴壁。
这支持优先检验“同观测下联合避壁/间距约束 + 更强的局部导航 baseline”，但尚不证明
强化学习必要、物理磁场独立控制或新算法贡献。详见 EXP0046 v2 的 FAILURE_ANALYSIS.md。

## 2026-10-04 — learning contribution and sensing audit

EXP0047 implements learning but does not demonstrate an advantage over the
strong common-filter heuristic. Improvements of the hand-written joint safety
projection must not be attributed to PPO. The old shared sensor still exposes
idealized velocities, mass fractions and frames; its success is not portable
evidence. Preserve all 32k/64k artifacts as diagnostics.

EXP0048 trains the same learned selector from scratch using measured-history
velocity estimates, noisy image-like local geometry and delayed binary visual
clearance. Both heuristics and learning share this pipeline and observed
completion rules. Generic PPO plus a filter is not an established novelty
claim. A learning contribution requires a reproducible paired advantage against
strong matched heuristics; magnetic actuation and imaging feasibility remain
unverified. Confirmation scenes stay untouched until a method is frozen.

### EXP0048–EXP0050: do not relabel failed learning as innovation

EXP0048 (tracked feedforward) and EXP0049 (tracked temporal candidate scoring)
both fail to beat the strongest memory heuristic through three seeds at 64k.
All-zero safe-success rates do not establish equality, and lower activity or
contact with poorer clearing is not a useful learning contribution. The new
EXP0050 nominal policy is exactly memory control; only trained deviations that
improve its matched outcomes can count. Real imaging/association/calibration,
magnetic independence and the intermittent native fault remain open.


### EXP0051 — hierarchical options: implemented, no deployment gain

Options/SMDP PPO, persistent measured target IDs and a shared joint supervisor were implemented and tested against flat learning and several conventional schedulers. Six trained deterministic policies matched memory control on all 12 main scenes, despite nonzero weight updates. All 184 main/stress attempts completed, but no learning improvement was established; stronger balanced allocation remains a comparator. Temporal abstraction and the shared supervisor are not novel by themselves.

Command-aligned shadow estimation reduced errors on the same four reference trajectories in two scenes without changing controller actions or final states. Category: measurement/engineering diagnostic, not learning, hardware transfer, or magnetic independence. EXP0052 gives this tracker to every policy and tests five-second hierarchical options from the strong measured allocation prior against identical-budget flat learning and classical allocation at 0.1/1/5-second intervals. All outcomes are pending until its full paired matrices finish.


### EXP0052 completed — useful observer diagnosis, hierarchy hypothesis unsupported

All 224 registered evaluations completed. Five-second hierarchical options achieved 83.57% mean removal versus 87.53% for the matched flat learner and 89.66% for the best-clearing conventional scheduler. Learning safety success remains 0/12 per seed. The current persistent-option design does not establish an algorithmic gain. The matched observer control reduced spacing exposure on the main layouts, but all arms share that processing; do not relabel it as learning. Extra-layout, same-checkpoint synthetic coupling controls still contain separation failures. No calibrated magnetic-independence, hardware-transfer, guaranteed-safety or publication-novelty claim is supported. Event-triggered termination is a subsequent untested hypothesis, not completed work.

### EXP0053 — TPG prototype plus upper/lower learning: implemented, not established as an advance

Graph priority selection at observation-triggered events and temporal local-frame residual selection with a measured-response auxiliary loss are implemented as independently ablatable learned modules. Trigger rules remain hand-written. The cropped graph, Dijkstra costs, continuous Euclidean proximity, dependency projection and nominal memory controller are shared conventional mechanics, not learning innovation. A graph-global geometry oracle is not introduced; unresolved path distance stays unknown.

The initial default and CPU-dispatch-mitigated full preflights each retain one native failure out of four; the original corrected dual-level training check also failed. Debugging located a NumPy native stack but did not prove its root cause. A subsequent isolated same-version generic NumPy build passed 83 targeted tests, three joint-training regression checks and eight preflight evaluations, then completed all nine 8192-step training runs and 52 paired development evaluations. All nine learned checkpoints reproduce the rule controller's final states on all four scenes. Mean removal is 87.50% for rules and learned variants, versus 93.75% for balanced_1s; all primary safe-success counts are zero. The lower scheduler-call count and wall contact are shared rule/graph behavior, not learning gains. Four scenes and this small training budget cannot establish learning's eventual limit.

The matrix is admitted under this declared runtime only. Old/new runtime initialization differs at tiny floating-point scales and closed-loop results differ; do not combine them as unchanged replications or claim the original fault is fixed. All previous failures/interrupted runs remain. No learned-termination, certified magnetic safety, sim-to-real or publication-novelty claim is supported. Confirmation remains unopened. Publisher-based comparator choices are in `experiments/VCTPG_BASELINE_SELECTION_20261004.md`; added recurrent/Transformer/MAPPO/Dreamer controls are proposals, not completed reproductions. Full status: `STATUS_TPG_20261004.md`.


## EXP0054/55 (2026-10-04): MARL comparison and failed safety-objective follow-up

- EXP0054: Recurrent MAPPO/IPPO communication-measured adaptations and V-CTPG action-conditioned learner/no-AC ablation share the same TPG and execution interface.12 training runs,216 evaluations,8 fixed development scenes; admitted comparison. Proposed final mean clearing47.60% versus MAPPO75.00%, IPPO73.96%, ruleTPG78.13%; all safe-success0. No learning novelty benefit established. Final checkpoints preserved despite worse outcomes than early checkpoints.
- EXP0055: matched original/safety-reward fine-tuning for all4 methods and3 seeds, with fresh optimizers and separate actor/critic clipping shared by both regimes.24 training runs complete,191/192 evaluation processes complete; oneSIGSEGV under generic runtime. Ranking blocked; diagnostic replays do not replace failure. Reward scaling is not algorithmic novelty.
- Missing: independent planning-system MARL baselines, MAT, fair tuning/convergence study, stable native runtime, positive reproducible learning contribution, hardware field calibration and sim-to-real validation. No confirmation access or publication-success claim.

## 2026-10-04：EXP0056–58 残差控制迭代

- EXP0056：共享有界离散残差 + 图/竞价高层 × KL/无KL四组；12次训练完成，119/120评估，N1崩溃。禁止整轮排名。
- EXP0057：同十二权重在统一清洁运行时完整部署评估，120/120与独立审计通过。graph_anchor清除71.875%，MAPPO79.1667%；两个KL组所有部署低层残差均为零。撤回其改善控制/低层有效学习的假设；不把共同安全奖励、残差界限或依赖环境变化算成学习创新。
- 高层图评分器尚无优于竞价高层的证据。改变少量调度次序不等于任务或安全收益。严格安全成功最多是graph_ppo的1/24，不能据此证明方法成熟。
- EXP0058：连续高斯残差替代离散选项的默认偏置门槛；所有MARL臂同样能力。无效动作不训练策略，价值函数仍学习阻塞状态。33项相关测试通过，已预登记12×16384步与120行矩阵。仍属于已知PPO/残差控制/参考正则的应用适配，不先行声称算法新颖性或优越性。

### EXP0058结果

120/120通过。连续残差解除离散部署零修正，但graph_anchor仍弱于匹配MAPPO清除与安全成功；壁面仅有描述性下降且区间跨零。graph高层/风险KL未被确认为有效创新。保留连续动作接口作为研究基础，停止以新命名或局部最佳指标宣称方法优越。

### EXP0059 高层隔离

固定低层后高层替换几乎不改变清除率：r_mappo低层四类高层均80.2083%，graph_ppo低层规则/MAPPO/graph PPO均75.0%。此前将整体差距主要归因于图高层的解释被修正；应优先研究低层动作接口、奖励信用分配和高低层闭环耦合。
