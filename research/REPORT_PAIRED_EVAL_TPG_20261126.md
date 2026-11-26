# 配对评估与 TPG 修复报告

## 直接回答

在相同场景和执行保护下，IPPO 的可观测导航贡献只出现在训练侧 3 个场景中的 2 个胜场和 1 个负场，留出侧 100 个场景逐场成败与 pursuit、MAPPO 完全相同。IPPO 的差异集中在局部动作选择与壁接触轨迹，不能从这些结果推出安全优势。

修正后的 MAPPO V2 critic 将联合状态和当前集群 one-hot 身份一起输入，actor 仍只读局部可部署观测。短训练已完成评估、保存、重载和确定性动作核对。跨 S41/S42/S43 的修正版重训尚未完成，因此不能宣称跨种子复现。

固定三靶点六场景重跑显示 N=3 的 TPG-on/off 全覆盖均为 4/6。TPG-on 最后到达时间 64.575 s，off 为 61.175 s。间距违规均值为 3.826 与 6.101 pair·s。TPG 降低间距违规，增加了等待代价，未显示同步率收益。当前计量单独记录导航到达、药物释放和磁场干扰，本文结果只涉及导航到达和安全间距。

## 版本和输入

远端 `origin/fix/paired-eval-tpg-20261125` 在 `d6940d0f352381af774a1711374c03d33aca963f` 后没有新提交。工作分支从实际起始 SHA `d6940d0f352381af774a1711374c03d33aca963f` 建立。

两两配对身份键为 `split`、`scene_seed`、`scenario_hash`、`realized_scenario_hash`、`accepted_seed`、`navigation_targets`。重复、缺失、场景集合不一致和身份字段不一致均使比较抛出错误。旧输入 `research/COMPARISON_20261125/ippo_s41_train180_chunked.json` 与 `pursuit_train180.json` 的严格身份交集为 4，留出对应输入交集为 2。

## 旧产物回归

训练导航成功为 pursuit 152/180、IPPO 153/180、MAPPO 151/180。IPPO 对 pursuit 为 2 胜/177 平/1 负，IPPO 对 MAPPO 为 2/178/0。留出侧三种方法逐场成败均为 80/100，严格成功均为 77/100。留出全平局标为 `observed_zero_difference`，没有标为方法等效。

生成文件为 `research/COMPARISON_20261125_REPAIRED/paired_comparisons.csv`、`paired_episode_rows.csv`、`paired_comparisons.json` 和 `paired_local_control_table.csv`。

## 动作回放

目标场景为 `scene_seed=3612010002`，身份哈希为 `219ffa8456025353b7efceb851de452b29916e3bf000fd8652f505103c351568`。回放输出位于 `research/replay_target_20261126`，包含逐步 JSON 和时间轴 PNG。三种方法均导航失败，IPPO 壁接触为 364.7959 robot·s，MAPPO 为 25.0000 robot·s。回放记录策略动作索引、候选局部命令、WallGuard/SettleGuard 后命令、SDF 投影占位命令、TPG hold、最终世界命令、位置、目标距离、壁接触和到达状态。pursuit 的终局摘要在旧适配器中不含统一 episode 字段，报告使用逐步轨迹累加并保留该适配差异。

已从训练配对表自动找到成败不同场景 `3404062011`、`3405052004`、`3413006012`，以及全成功对照 `3400000000`。当前已完成回放的是目标场景和 `3400000000` 对照。其余三个差异场景已冻结身份，尚未完成三方法轨迹回放。

## MAPPO 修复和测试

`scripts/train_sb3_ippo.py` 中新增版本化 `MAPPOMultiInputActorCriticPolicyV2`。V2 的 centralized state 对每个 critic 样本附加 one-hot 集群身份，actor 路径不读取该字段。旧 `MAPPOMultiInputActorCriticPolicy` 保留，旧 checkpoint 入口仍可加载。训练结束评估显式按策略是否有 centralized critic 配置观测。

短训练输出位于 `research/smoke_mappo_v2_20261126_c`，物理步数 256、智能体步数 768、评估 1 回合，保存和重载动作均为 `[10, 10, 10]`。评估回合导航失败属于短随机试跑结果，不作为性能结论。

运行的测试为 `42 passed`，覆盖配对回归、MAPPO actor 隔离、V2 critic 标签拟合、局部观测动作、SpacingTracker 和固定三靶点实体接触终局读取。

## TPG 计量

`scripts/evaluate_fixed_three_lesions.py` 现在记录 `tpg_gate_calls`、`tpg_trigger_count`、`waiting_transitions`、`waiting_robot_s`、`spacing_violation_pair_s` 和终局读取的 `entity_pair_contact_s`。六场景 on/off 重跑文件为 `research/COMPARISON_20261125_REPAIRED/fixed_three_tpg_on_recount.json` 和 `fixed_three_tpg_off_recount.json`。N=3 on 的 gate 调用总数为 9396，hold 边沿 5，等待 robot·s 346.5，实体接触为 0。off 的 gate 调用、hold 和等待均为 0，实体接触为 0。N=3 间距违规均值分别为 3.8257 和 6.1009 pair·s。

## 解剖来源和未完成矩阵

当前固定三靶点实验使用仓库已有 `mca_m1_lvo` 血管几何和历史病灶标注。仓库没有可直接验证的肝动脉 HCC 三靶点几何输入，因此尚未把 MCA 结果写成 HCC 解剖结论。迁移还需要肝动脉中心线和半径、三靶点坐标及访问顺序、起点池、流场边界、障碍分布和固定仿真时限。

尚未完成 S42/S43 修正版 MAPPO 重训、完整 2×2×2 图分配/TPG/IPPO 消融、N=1/2/3 资源预算矩阵和三个成败差异场景的三方法轨迹回放。恢复命令如下。

```bash
python scripts/summarize_repaired_comparison.py
python scripts/replay_paired_execution.py --split train --scene-seeds 3404062011,3405052004,3413006012,3400000000 --ippo-model research/COMPARISON_20261125/models/ippo_s41_final_model.zip --mappo-model research/COMPARISON_20261125/models/mappo_s41_final_model.zip --out research/replay_remaining_20261126
python scripts/train_sb3_ippo.py --algo mappo --seed 42 --timesteps 500000 --out research/mappo_v2_s42
```
