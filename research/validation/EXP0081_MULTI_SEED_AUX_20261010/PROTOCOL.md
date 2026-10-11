# EXP0081：nav_tf_v3 归因、多个种子与动力学辅助任务

目标：把成功率分解为路线/TPG、学习控制和安全层贡献，并检查显式流场—响应状态估计是否改善分布外动力学。

## 配对消融

- `full`：nav_tf_v3 + TPG + WallGuard + spacing shield。
- `no_guard`：关闭 WallGuard，保留 TPG 和 spacing shield。
- `no_shield`：关闭 spacing shield，保留 TPG 和 WallGuard。
- `no_safety`：同时关闭 WallGuard 和 spacing shield。
- `no_tpg`：关闭 TPG，保留 nav_tf_v3 和安全层；采用 park fallback，避免完成集群去帮助其他集群。

消融使用 `research/validation/V5_PHYSIO_20261008/*_nav_tf_v3.jsonl` 的注册场景作为场景清单，逐回合重放同一解剖、种子、N 和扰动强度。

## 多种子

- `nav_tf_v3_s7102`、`nav_tf_v3_s7103`：原始 v3，新增训练种子。
- 已有 `nav_tf_v3_s7100`、`nav_tf_v3_s7101` 作为前两种子。

## 动力学辅助任务

- `belief_s7102`、`belief_s7103`：在 PPO 上增加训练期特权辅助头，预测归一化瞬时流速向量和有效响应增益；仿真真值只进入辅助损失，部署时不使用。
- belief 头输出均值与 log variance，使用高斯 NLL；不确定度还用于限制策略残差幅度。

## 指标

每个评估单元报告 raw/full、relaxed、strict、清除率、AUC、T50/T90/T100、壁接触、壁接触阈值、间距违规、机器人接触、耦合暴露、TPG hold、路径长度、耗时和清除血栓数。

## 后台部署

训练和评估通过 tmux 会话运行：`train_nav_tf_v3_s7102`、`train_nav_tf_v3_s7103`、`train_belief_s7102`、`train_belief_s7103`、`nav_decomp_v3`、`nav_auto_eval`。
