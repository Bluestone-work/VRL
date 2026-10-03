# Failure log

截至预登记时，本轮尚无运行失败。历史失败和潜在风险不是本轮实测失败；见PROJECT_AUDIT A01–A12。

每个新增失败保留 Experiment / Failure / Symptoms / Root Cause Hypothesis / Evidence / Fix / Scientific Lesson。native crash不自动解释为算法失败；同样不在缺乏证据时归因为硬件。任何重试保留全部attempt及原始stderr。

## EXP_0001 — F001 — smoke post-check import path

- Experiment：EXP_0001，smoke4242，2026-09-20。
- Failure：训练成功后，外层coordinator载入带环境恢复状态的checkpoint失败。
- Symptoms：`ModuleNotFoundError: No module named 'environments'`，发生在research_baseline.inspect_run的torch.load，而非训练update。
- Root Cause Hypothesis：以scripts路径直接运行coordinator且未设置其PYTHONPATH；训练子进程已显式设置冻结源码PYTHONPATH，所以能成功训练。
- Evidence：smoke/attempt_1.json记录returncode=0、耗时38.2094s；stdout完成两次update与最终验证；随后外层torch反序列化找不到自定义环境类。
- Fix：不修改冻结代码、不重新训练；以`PYTHONPATH=/home/wj/桌面/vascular_marl_research_EXP_0001`重跑原inspect_run，并在正式coordinator使用同一搜索路径。检查恢复结果另存smoke/postcheck_recovery.json。
- Scientific Lesson：checkpoint可保存不等于在当前Python路径可还原，coordinator本身也需要明确运行环境。此失败不应计作策略不收敛，更不能删除该attempt。

## EXP_0001 — F002 — analysis script syntax preflight

- Experiment：EXP_0001，分析工具预检查，尚未产生报告或结果。
- Failure/Symptoms：新增research_report.py在py_compile阶段发现多余右括号。
- Root Cause Hypothesis/Evidence：exp.write_text的字符串拼接提前闭合；编译器定位行260。
- Fix：修正分析脚本括号并重新编译。它不属于已冻结的训练/评估执行代码；未更改原始数据或评价指标。
- Scientific Lesson：分析工具也应在接触正式结果前做语法检查；保留工程失败，不把它解释成科研负结果。

## EXP_0001 — F003 — cross-device trajectory reproducibility limitation

- Experiment：EXP_0001，smoke最终checkpoint的14回合逐场景回放。
- Failure：CPU回放未与原GPU回放逐轨迹一致；不是训练失败。
- Symptoms：14回合success均一致，但4个场景存在步数、撞壁或质量差异，最大质量清除率绝对差0.02906534（2.9065pp）。
- Root Cause Hypothesis：CPU/GPU浮点实现差异经接触/分支边界放大。具体最早分歧运算尚未定位，因果细节为HYPOTHESIS — NOT VERIFIED。
- Evidence：checks/diagnostic_replay.json保留CPU结果；将设备恢复cuda:0、保持同一checkpoint/seed/只读诊断器后，checks/diagnostic_replay_gpu.json中14回合success、steps、wall_total和removal均完全一致，最大误差0。
- Fix：不改数值实现或结果；原复现主指标保留GPU，额外CPU开发验证明确标设备。后续比较应固定评估设备，跨设备bitwise复现不作保证。
- Scientific Lesson：加入测量代码本身未改变这14回合同设备结果，但跨设备轨迹存在敏感性；不能把不同设备下的小差异归因于算法。

## EXP_0001 — F004 — incomplete clearance and high wall contact

- Experiment：EXP_0001，development validation；实验复现通过，但任务失败回合完整保留。
- Failure：840回合中213回合未清空全部血栓。股腘动脉场景仅7/60成功（11.6667%）；不能用92.97%的平均质量清除替代成功率。
- Symptoms：预先固定的seed42 / femoropopliteal_pad / episode1020000运行300步，清除98.3676%，最后50步新增消融0；wall contact/robot-step为87.1333%。全股腘动脉场景wall contact为80.26%。
- Root Cause Hypothesis：局部可控性、接触范围、目标切换或约束投影可能造成无法清除末端残余；这些原因尚未分离，全部为HYPOTHESIS — NOT VERIFIED。未证明reward hacking，也未证明世界模型能解决。
- Evidence：`runs/EXP_0001/analysis/development_by_territory.csv`、`fixed_episode_cases.json`及对应`evaluation/seed_42/traces/`；图见`figures/EXP_0001_clot_mass_and_flow.png`和`EXP_0001_robot_trajectories.png`。
- Fix：本轮没有修改物理、策略或奖励；保留失败并在评估协议核验后安排单变量行为诊断。
- Scientific Lesson：必须联合看成功、剩余质量曲线和安全事件。高总体清除率掩盖了部分场景的完成困难；碰撞率是离散几何事件，不能直接换算组织损伤。

## EXP_0002 — F005 — inventory generation boundary interpretation

- Experiment：EXP_0002，冻结后源码复核；影响清单描述，不影响策略执行或评价指标。
- Failure：清单工具的`missing_generations_through_last_saved`使用0..max枚举，会把未使用的generation0也列出。
- Symptoms：`VectorVascularEnv`初始化计数0，而`_sample_tree()`先加1；首个实际血管编号为1。
- Root Cause Hypothesis / Evidence：`scripts/research_protocol.py`的range(max+1)和`environments/vector_env.py`初始化/采样语义不一致；这是可定位的边界解释错误。
- Fix：不改冻结执行代码或原始清单。独立分析器从`observed_generations`按1..max重新推导实际覆盖，保存`actual_training_generation_coverage`；报告显式说明0不代表遗失的真实血管。
- Scientific Lesson：统计缺失覆盖必须尊重源系统编号语义。直接证据是原始observed generation集合；不能把派生字段当作独立事实。未来复用清单工具应在新实验版本修正该字段。

## EXP_0002 — F006 — cross-device outcome flips

- Experiment：EXP_0002；协议实验完成，失败指跨设备逐轨迹/结局一致性不成立，不是训练失败。
- Failure / Symptoms：840对固定权重/初始状态回合中10对success翻转，仅CPU成功1、仅GPU成功9；完整trace数组精确一致0，位置分歧838。
- Root Cause Hypothesis：数值差异在policy/控制链出现，并可能在接触、路由或投影处被放大；具体算子与后续机制仍为HYPOTHESIS — NOT VERIFIED。
- Evidence：84对同设备重复精确一致，CPU840回合与父实验精确一致；初始动作最大差2.38418579e-7，共同前缀位置L2最大差0.903776646。见`runs/EXP_0002/analysis/paired_episodes.csv`、`success_flip_cases.json`和原始trace。
- Fix：保持原模拟器和权重不变；未来比较使用预登记cuda:0。未通过调整奖励/容差/seed掩盖翻转，也不宣称GPU更接近物理真值。
- Scientific Lesson：不到1pp的平均成功差也可能来自设备，且质量清除变化方向相反；必须固定设备并联合报告各指标。

## EXP_0002 — F007 — incomplete historical geometry provenance

- Experiment：EXP_0002，历史数据可追溯性审计。
- Failure：不能认证EXP_0001完整训练几何与评估互斥。
- Symptoms：每训练seed保存到140个子环境/generation组合；截至最终checkpoint累计创建generation计数为266，另有未保存部分。该分母可能含初始化后丢弃的树，不等于全部实际训练树数量。
- Root Cause Hypothesis / Evidence：`no_dataset=true`、树周期重采样但checkpoint间隔更长、initial/best权重没有完整env状态。原始checkpoint清单与observed generations保存于`runs/EXP_0002/splits/saved_training_geometries.json`。
- Fix：新建560/140/280互斥实例清单，test只登记不评分；后续训练需显式接入消费/生成限制。本轮未重建或伪造缺失训练历史。
- Scientific Lesson：在恢复到的420个独立树hash中未发现与验证的交集，只能说明这一子集无精确重复，不能证明全历史无泄漏。另已确认final validation复用模型选择回合比例25%，它始终是validation。

## EXP_ONLINE_IMITATION_R1 — DAgger round 1 gives no completion gain (2026-10-03)
- Symptoms: vs a step-matched BC-only control, train complete +1.7 pp [−1.7, +5.0], held-out −2.0 pp [−6.0, +2.0]; tracking of the teacher on the student's own states improves (cos +0.025 [+0.010, +0.040]).
- Root cause hypothesis — NOT VERIFIED: the BC epoch-7 student already rarely leaves the teacher's state distribution on training anatomies, so DAgger adds little; the held-out gap (femoropopliteal, SMA) is an out-of-range scale problem that training-anatomy DAgger states cannot cover.
- Fix: none applied; recorded as a null result.

## EXP_TEACHER_BC — held-out wall contact and long-anatomy failures (2026-10-03)
- Symptoms: held-out wall contact 12.3 s vs teacher 0.01 s; femoropopliteal_pad 80 vs 100, sma_embolism 80 vs 85.
- Hypothesis — NOT VERIFIED: femoropopliteal (241 mm) is longer than every training anatomy (max 181 mm); geodesic and depth features are out of training range.
- Lesson: offline imitation accuracy (cos 0.988) badly under-predicts closed-loop behaviour; select on rollouts.
