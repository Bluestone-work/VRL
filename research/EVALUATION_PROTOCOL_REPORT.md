# EXP_0002 — Evaluation protocol report

执行版本：`c7830951f2674aa7899b25da2ff0d49d8e81d8ef`。分析脚本hash：`d33ca8569fba99dbb5f328879d9d6ab230aa66def0b2379f0b83ba4a824a3566`。

本轮不训练，复用EXP_0001全部三个最终权重。唯一性能变量是CPU vs cuda:0；同一840回合配对，另做84次同设备重复。均值±样本标准差来自3个训练seed，不能把840回合解释为840独立模型。
开发验证，不是封存test；GPU未来统一设备在看到结果前已固定。

| Device | Success | Mass removal | Episode length | Wall / robot-step | Collision / pair-step |
|---|---:|---:|---:|---:|---:|
| cpu | 74.6429 ± 0.3571% | 94.9299 ± 0.3601% | 152.3857 ± 2.4573 | 34.7250 ± 0.8269% | 8.1387 ± 0.1522% |
| cuda:0 | 75.5952 ± 0.7435% | 94.8884 ± 0.2695% | 150.5321 ± 2.7320 | 34.3926 ± 0.7891% | 7.9576 ± 0.1663% |

## Per-seed paired results

| Seed | CPU success | GPU success | GPU−CPU success (pp) | CPU removal | GPU removal |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 75.0000% | 76.4286% | +1.4286 | 95.3144% | 95.1109% |
| 43 | 74.2857% | 75.3571% | +1.0714 | 94.8745% | 94.9657% |
| 44 | 74.6429% | 75.0000% | +0.3571 | 94.6006% | 94.5887% |

## Actual result and comparison

成功率GPU−CPU=+0.9524pp，相对+1.2759%；清除率差-0.0414pp；长度差-1.8536步；wall差-0.3324pp；pair collision差-0.1811pp。
这些是测量设备效应，不称Algorithmic Improvement或临床差异。CPU与父实验同回合的非时间指标和全部trace数组精确一致，基线差0。
成功结局翻转10/840：仅CPU成功1，仅GPU成功9；跨设备全部trace数组精确一致0/840。
相同初始状态下首次执行动作的最大绝对差2.38418579e-07；共同时间前缀内最大机器人位置L2差0.903776646（模拟单位）。不补齐已结束轨迹，不把不同长度直接作整段相减。
首次动作已出现差异能定位到policy/控制计算链的设备影响；尚未定位具体浮点算子或证明哪种设备更接近物理真值。放大机制仍为HYPOTHESIS — NOT VERIFIED。

## Correctness, stability and costs

完整回归177 passed、1 skipped。56回合smoke通过；正式CPU父实验840回合一致；84对同设备重复精确一致；42个历史GPU控制回合与旧接口的success/steps/wall一致且removal差≤1e-7。没有训练、梯度更新或checkpoint选择。
CPU完整策略接口推理：0.6668 ± 0.0032ms/step；GPU：1.0964 ± 0.0025ms/step。包含critic/动作转换与传输，在并发执行下测得，不是纯actor或受控硬件benchmark。
正式任务作业耗时总和CPU=254.29s、GPU=308.27s，包含载入/重复/trace记录，不等于并行总wall-clock。
Training Stability、Training Cost、Sample Efficiency改善：N/A（Training Steps=0）。Completion Rate=Success Rate；任务质量/碰撞定义沿用EXP_0001；未引入新的unsafe流速阈值或物理单位。

## Split audit

内容hash包含点、半径、分支、连接、flow_fraction和派生几何数组；它检查精确有序内容身份，不是图同构或近似重复检测。初始状态另包含机器人/血栓/RNG身份。
未来train/validation/test分别560/140/280个实例；各split内部及三者之间内容hash无重复。封存test策略执行数0。
该清单是供后续训练消费的prospective split；尚未接入训练器，不可宣称旧策略没有训练泄漏。所有集合来自相同14类场景，不是unseen-topology test。
历史评估与选择验证的复用：{"42": {"selection_validation_episodes": 70, "final_validation_episodes": 280, "overlapping_episodes": 70, "overlap_fraction_of_final": 0.25}, "43": {"selection_validation_episodes": 70, "final_validation_episodes": 280, "overlapping_episodes": 70, "overlap_fraction_of_final": 0.25}, "44": {"selection_validation_episodes": 70, "final_validation_episodes": 280, "overlapping_episodes": 70, "overlap_fraction_of_final": 0.25}}。
已保存训练树共462条记录、420个独立完整hash；checkpoint无法提供完整训练历史。
实际generation从1开始。原清单的missing_generations_through_last_saved按0..max列举，包含不存在的generation0；本报告仅由observed_generations按1..max重新计算实际树覆盖。保留原始字段，不静默改写。
该分母为创建过的树generation数量，可能包含初始化后重置、未用于采样的树，不能直接解释为全部实际训练转移覆盖。

- seed42：已保存的子环境generation 140/266，未覆盖126；不足以认证历史训练与评估完整互斥。
- seed43：已保存的子环境generation 140/266，未覆盖126；不足以认证历史训练与评估完整互斥。
- seed44：已保存的子环境generation 140/266，未覆盖126；不足以认证历史训练与评估完整互斥。

全部集合的交集计数如下；saved_historical_train仅代表有保存证据的子集。

| Pair | Identical content hashes |
|---|---:|
| train__validation | 0 |
| train__test | 0 |
| train__saved_historical_train | 0 |
| train__legacy_final | 0 |
| train__development | 0 |
| validation__test | 0 |
| validation__saved_historical_train | 0 |
| validation__legacy_final | 0 |
| validation__development | 0 |
| test__saved_historical_train | 0 |
| test__legacy_final | 0 |
| test__development | 0 |
| saved_historical_train__legacy_final | 0 |
| saved_historical_train__development | 0 |
| legacy_final__development | 0 |

## All outcome flip cases

全部10个翻转回合均列出；1表示全部血栓清零。

| Training seed | Scenario | Episode seed | CPU success | GPU success | GPU−CPU removal (pp) |
|---|---|---:|---:|---:|---:|
| 42 | coronary_rca | 920014 | 0 | 1 | +0.4675 |
| 42 | ica_siphon | 960001 | 0 | 1 | +0.0700 |
| 42 | carotid_bifurcation | 980000 | 0 | 1 | +0.4013 |
| 42 | carotid_bifurcation | 980002 | 0 | 1 | +1.3204 |
| 43 | coronary_rca | 920010 | 0 | 1 | +0.3165 |
| 43 | popliteal_calf_dvt | 940006 | 0 | 1 | +12.1657 |
| 43 | basilar_vertebral | 990019 | 0 | 1 | +21.4276 |
| 44 | mca_m1_lvo | 950016 | 1 | 0 | -13.3225 |
| 44 | ica_siphon | 960009 | 0 | 1 | +0.9009 |
| 44 | sma_embolism | 1010014 | 0 | 1 | +1.1258 |

## Failure cases, observed behaviors and interpretation

全部失败与翻转回合在paired_episodes.csv和各设备episodes.jsonl；没有按成功挑选或重跑。轨迹图沿用EXP_0001预固定的肺动脉与股腘动脉两个episode ID。
What improved：测量来源、设备、配置、身份和未来划分现在可审计。Why：同设备和父接口控制提供直接证据。
What worsened：新增检查和trace记录有执行成本；设备间性能指标的变化见上表，不能概括为统一改善。
What did not change：权重、物理、奖励、观测、动作变换和开发验证回合。
Hypothesis support：通过同设备/接口复现检查；跨设备是否精确一致由翻转与轨迹计数判断。H1–H5未测试。
New question：如何定位最早浮点分歧及其接触/投影敏感性，怎样在未来训练中强制消费split清单？这些应独立实验，不在此轮改数值物理。
Engineering Improvement：协议检查与数据身份。Algorithmic Improvement：无。Scientific Contribution：本机固定模拟器下的设备敏感性证据。Potential Novel Contribution：无。Needs Literature Verification：I1–I7。
NOVELTY CLAIM — NEEDS LITERATURE VERIFICATION。Literature verification required.

## Figures and reproducibility

![EXP_0002 paired_device_metrics](figures/EXP_0002_paired_device_metrics.png)

![EXP_0002 trajectory_divergence](figures/EXP_0002_trajectory_divergence.png)

![EXP_0002 fixed_episode_comparison](figures/EXP_0002_fixed_episode_comparison.png)

原始数据、attempt、协议、snapshot、硬件依赖、父输入hash与split清单：`research/runs/EXP_0002/`。
分析入口：`scripts/research_protocol_report.py`；结果版本独立归档。未来测试清单未获得性能分数。

## Conclusion and next experiment

EXP_0002协议gate通过；不认证历史完整训练划分或跨设备等价。允许EXP_0003完善科研指标及准备未来split消费接口，不直接进入世界模型规划。
