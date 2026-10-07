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

## EXP0060 — initial runtime failures and null short-training comparison (2026-10-06)
- Unrestricted CPU affinity: original runtime baseline SIGILL and SIGSEGV; clean runtime baseline SIGSEGV; clean pilot worker SIGSEGV followed by parent EOFError after three updates. All attempt directories retained; details in EXP0060/ADMISSION.json. These are runtime failures, not policy metrics.
- Used already-documented taskset exclusion of CPU 6/7. Subsequently completed 7 relevant tests, 40 PPO updates and 24 paired evaluation episodes. No claim that this identifies or repairs the hardware cause.
- Short deterministic pilot: memory-MAPPO 5/12 Safe, traditional 5/12; removal 100% both; AUC delta -1.47e-7. The policy retained mostly the original pursuit option. No learning superiority established. Formal longer training/ablations queued with all seeds retained.
- GUI development: VTK automatic smooth normals failed on a combined strip mesh while returning process exit 0; screenshots were visually checked and corrected by preserving tube normals and disabling that automatic smoothing stage. This is a rendering-only fix, not a physics or result change.

## EXP0061 — longer PPO and first world-assisted repair do not beat pursuit (2026-10-06)
- Full EXP0060 seed0 parent: 12 paired development scenes, Safe 5/12 (same as pursuit), clearance 97.92% vs 100%, wall contact 22.54 s vs 0.502 s. Longer training did not preserve the pilot's baseline-level behavior.
- First matched 120-update repair: continued PPO wall 22.637 s / world PPO 22.590 s; both clearance 97.92%, Safe 5/12. World-minus-PPO AUC delta 0.0000516, paired 95% interval includes zero. Both failed the registered no-regression acceptance gate; scene and actual initialization hashes matched.
- Prediction audit on fixed-behavior trajectories: next-state prediction only marginally beats persistence; reward prediction is worse than zero reward; particle Brier is worse than zero-risk prediction; no spacing-positive events were observed. Low aggregate model losses alone do not establish useful world modeling or avoidance.
- Hypotheses, NOT VERIFIED: inherited option preference and unstable/poorly scaled actor-critic optimization may contribute to wall regression. Rare hazard labels may limit risk learning. These are not established causal diagnoses.
- Automatic next action: raise both arms' wall/particle penalties and reward horizon, keeping equal parent and budget; round 01 confirmed running. All failed models and evaluation records retained. Source and training labels remain separate from actor observations.

- 2026-10-07 obstacle PL-TS G1 is still running; 11 paired development scenes show promising AUC/wall deltas but more obstacle events than APF. No superiority claim accepted before all 84 scenes.

## PL-TS G1 interim status (2026-10-07)
At 41/84 paired N=1 development scenes, privileged lookahead teacher Safe Success was 13/41 vs APF 6/41 (+17.1 pp); mean AUC delta +0.0029, mean wall-contact delta -6.09 s, but mean obstacle-event delta +0.439. This remains interim and does not pass the pre-registered G1 threshold until all 84 scenes and paired uncertainty are complete.

## EXP_PLTS_G1 — privileged lookahead teacher gate complete (2026-10-07)
- Completed all 84 registered paired N=1 development scenes; zero infrastructure errors.
- Teacher vs APF cluster Safe Success: 31/84 (36.90%) vs 16/84 (19.05%), paired difference +17.86 pp, bootstrap 95% CI [+5.95, +29.76] pp; this passes the registered G1 +15 pp threshold.
- Teacher task success: 71.43% vs 32.14%, difference +39.29 pp, CI [+27.38, +50.03] pp. AUC difference +0.00853, CI [-0.02521, +0.04246], not established.
- Teacher wall contact: 0.089 s vs 6.528 s, difference -6.439 s, CI [-12.323, -2.148]. Obstacle events: 0.524 vs 0.060 per episode, difference +0.464, CI [+0.333, +0.607]. Lost difference -0.0238, CI [-0.0595, 0].
- Interpretation: G1 passes the pre-registered Safe gate and demonstrates a safety trade-off, not an unconditional win. The teacher reduces wall/lost failures but incurs more obstacle events; student G2 must retain obstacle events as a primary safety constraint.
- Evidence: `research/validation/OBST_BENCH_20261006/gates/G1_teacher_vs_apf.jsonl`, 84 rows, paired bootstrap seed 20261007.

## EXP_PLTS_G2_R0 — offline-distilled students are unsafe in closed loop (2026-10-07)
- Students: Transformer/GRU/MLP distilled from 54 teacher episodes (89,822 train samples; soft-target CE alpha .7 tau 5), checkpoint selected on validation teacher-cost regret (1.69 / 1.70 / 2.30; catastrophic-choice rate 1.6/1.6/2.1 % per decision).
- G2 stopped early once the outcome was unambiguous; all partial paired records retained in gates/G2_{transformer,gru,mlp}_r0.jsonl (zero errors).
- Symptoms (paired subsets, n=26/22/16): obstacle events per scene 7.96 / 5.91 / 8.44 vs APF 0.12-0.19 and teacher 0.54-0.81; Safe 3.8 % / 0 % / 0 % vs APF 7.7 / 9.1 / 0 %. Task success 54 / 55 / 50 % vs APF 15-18 %: students keep progressing but run through obstacles.
- Root cause hypothesis — NOT VERIFIED: compounding error. 1.6 % catastrophic choices per 10 Hz decision is ~50 bad decisions per 300 s episode, and student-induced states are absent from teacher-driven data; CE gives little weight to how bad a wrong option is.
- Fix (next): DAgger on student-driven states plus an expected-teacher-cost loss term.

## EXP_PLTS_G2_DG1 — DAgger students beat APF on task/removal but fail Safe; a simple switch rule matches the teacher (2026-10-07)
- 84/84 paired dev scenes, 0 errors. GRU DAgger student vs APF: task success 59.5 vs 32.1 % (+27.4 pp, CI [+16.7, +39.3]); removal 78.5 vs 56.2 %; wall 1.39 vs 6.53 s (CI [-10.6, -1.2]); obstacle events 3.23 vs 0.06 (CI [+2.6, +3.8]); Safe 2.4 vs 19.0 %. Transformer similar (Safe 3.6 %, obstacle events 4.02). Collisions split evenly static/dynamic.
- Deployable proximity shield (detected gap < 0.3 / 0.5 mm -> wall-aware APF) on the GRU student: Safe 19 % / lower; obstacle events 0.5 / 0.2. No gain over APF.
- New stronger deployable baseline `rule_switch` (pursuit unless a detected surface is within 0.3 / 0.5 mm, then wall-aware APF): Safe 38 % / 37 %, task 71 % / 57 %. This equals the privileged v1 teacher (Safe 37 %, task 71 %). Any learning claim must be made against rule_switch, not only APF.
- Failure modes: v1 teacher fails almost only by obstacle hits (42 % of scenes), rule_switch 0.3 by obstacle hits (55 %), rule_switch 0.5 by wall contact (38 %).
- Hypothesis — NOT VERIFIED: the v1 teacher horizon (10 steps = 1 s, option held throughout) is too short to set up an obstacle pass, so its labels encode reactive choices; per-step (10 Hz) student decisions multiply the per-decision error.
- Fix: teacher v2 (option committed 0.5 s, then 2.5 s continuation under the deployable switch rule; 9th option 'switch'); student decides at the same 2 Hz period.

## EXP_PLTS_G2_V2R0 — v2-distilled students still fail Safe (2026-10-07)
- Data: 66 teacher-v2 episodes on train seeds 2613000000+ (teacher Safe 66.7 %, 1.03 obstacle events/episode), 23,301 decisions at 2 Hz; labels 71.7 % pursuit.
- 84/84 paired dev scenes. Transformer / GRU: Safe 2.4 % / 2.4 % vs rule_switch0.3 38.1 % (CI excludes 0); task success 83.3 % / 81.0 % vs rule_switch 71.4 % (CIs [+6.0,+19.0] / [+2.4,+17.9]) and vs APF 32.1 %; obstacle events 8.3 / 9.4 per scene vs 5.7 (rule_switch) and 1.55 (teacher v2). GRU wall contact 1.95 s vs 10.66 s (CI [-17.2, -2.2]).
- Hypothesis — NOT VERIFIED: offline macro recall 0.17 — the student collapses to the majority 'pursuit' label and does not reproduce the minority avoidance decisions that make the teacher safe; deployable tokens may not carry enough information to predict which 0.5 s commitment avoids a collision 2-3 s later. Next: DAgger v2 round 1 (running).

## EXP_PLTS_G2_V2DG1 — DAgger with teacher v2 does not fix student safety (2026-10-07)
- 66 DAgger episodes (student executes 70 %), GRU retrained on v2_bc+v2_dg1 (cost weight 4). 84/84 dev scenes: Safe 4.8 % vs rule_switch 38.1 %; task 79.8 vs 71.4 %; obstacle events 5.58 vs 5.69; wall 5.85 vs 10.66 s; 95 % of scenes have at least one hit.
- Conclusion for the imitation route: three rounds (v1 BC, v1 DAgger, v2 BC, v2 DAgger) gave students that out-navigate the heuristics but never reach their Safe Success. The deployable token apparently does not contain enough information to reproduce the privileged teacher's avoidance choices (HYPOTHESIS — NOT VERIFIED). Route switched to RL from the reactive prior (WM-OPPO).

## VP-PPO vp_wm_s0 — training crash at 1.89M steps caused by a live code edit (2026-10-07 ~13:10)
- Symptom: `ValueError: not enough values to unpack (expected 5, got 4)` in a pool worker. The worker-job tuple of scripts/train_vision_ppo.py was extended (reward weights) while the run was alive; replacement workers (maxtasksperchild) imported the new code while the parent kept sending the old tuple.
- Fix: resumed from the last checkpoint step01516k (actor only; critic, optimiser and reward scale restart) as vp_wm_s0_resume for the remaining steps to 3M. Reported as a resumed run. Lesson: never change the worker interface of a running script; copy the script per run.
- Separate fix: vp_bc_s0 first attempt used soft-target temperature 2 while teacher costs differ by ~0.1-1 (near-uniform targets; start entropy 1.76). Aborted, kept as vp_bc_s0_tau2_aborted; relaunched with tau 0.1.
