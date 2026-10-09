# 状态汇总：多集群破栓导航（v4 → v6），2026-10-09

供外部分析（例如 GPT）阅读的入口文档。所有数字都来自开发集（14 解剖 × 10 场景 × N=1/2/3，每格 140 回合），测试集未打开。
学习方法全部是单种子 pilot（95–120 min）。逐回合原始结果见各 `research/validation/*/*.jsonl`，字段说明见 `scripts/benchmark_lysis.py` 的 docstring。

## 1. 任务与评价（用户 2026-10-08 定）

- 1–3 个磁性微机器人集群，在三维血管树中清除 4 个血栓，尽量少碰壁。去掉了动态碎片避障。**本文方法必须是学习类导航。**
- 只用可部署信息：术前 CTA 地图（中心线、健康半径、血栓位置）、双平面图像的集群位置估计、血栓是否仍可见。仿真真值只用于指标和训练奖励。
- 四类指标：
  - 效率：T50 / T90 / T100、AUC；
  - 完整度：清除率、全清率；
  - 互不影响：间距违规 pair·s、偶极耦合暴露；
  - 碰壁：robot·s、壁 ≥1 s 回合比例。

## 2. 环境版本

| 版本 | 内容 | 入口 |
|---|---|---|
| v4 | union-of-tubes 管腔、图像感知、无障碍；稳态血流，各解剖入口流量相同（0.01 mL/min × U[1.5, 2.5]） | `scripts/benchmark_lysis.py` |
| v5 | v4 + 个体化 / 时变动力学，强度 s：脉动流、个体流量、响应增益 / 漂移 / 方向偏差 / 延迟、窄管形变、近壁阻力、黏附、溶解速率 | `marl/physio_variation.py`，参数 `variation=s` |
| v6（定标中） | v5 + 按解剖标定的入口流速（`flow_inlet_mm_s`），可切换稳态 / 脉动 | 同上，参数 `flow_inlet_mm_s`、`variation={'s':1,'pulsatile':False}` |

## 3. 方法清单

| 名称 | 类别 | 说明 |
|---|---|---|
| classical_settle | 经典最强 | 术前测地分配 A + TPG + 拓扑修正纯追踪 + 壁护 + 卡住重规划 + 到栓减速 |
| stpg | 基线（AAAI 2025） | 可切换 TPG，在线重优化通行顺序（`marl/lysis_baselines.py`） |
| pac_nmpc | 基线（CASE 2025） | PAC-NMPC 式分布式采样 NMPC（`marl/lysis_baselines.py`） |
| turbo / turbo_v5 | 基线（NMI 2026） | 因果 Transformer PPO 直接输出速度（`scripts/train_lysis_local.py --mode direct`） |
| nav_tf_v3 | **本文（当前最好）** | 路线 Frenet 坐标系的 Transformer PPO：速度相对 settle 先验的修正 + 横向瞄准，v5 域随机化（`scripts/train_lysis_nav.py --speed-prior`） |
| nav v1 / v2 / v4 / ECG | 本文迭代 | 均为负结果，原因见 `research/RESEARCH_LOG.md` |
| alloc_mappo / residual 系列 | v4 时期的学习尝试 | MAPPO 在线分配 ≈ A；在 settle 先验上加残差 PPO 为负；从只有壁护的先验学起，N=3 全清 +3.6 pp |
| oracle_flow | 特权上界（只用于分析） | 知道真实血流和响应并反解（`FlowOracle`） |

## 4. 关键结果

**v4（理想）**：经典最强全清 89.3 / 95.7 / 94.3%。已清场景的 T100 是理想下界的 1.00–1.03 倍，效率已经到顶。STPG 与之持平，PAC-NMPC 低 4–5 pp，Turbo 1–30%。
详见 `research/validation/V4_LYSIS_20261008/REPORT.md`、`TABLES.md`。

**v5，s=1**（全清 % / T90 s / 壁 ≥1 s %，依次 N=1 | 2 | 3）：

| 方法 | N=1 | N=2 | N=3 |
|---|---|---|---|
| nav_tf_v3（本文） | 85.7 / 189 / 7 | 92.9 / 124 / 14 | 92.9 / 95 / 14 |
| classical_settle | 85.7 / 188 / 8 | 90.7 / 124 / 12 | 95.0 / 87 / 11 |
| stpg | — | 91.4 / 121 / 12 | 95.0 / 87 / 11 |
| pac_nmpc | 82.1 / 194 / 5 | 84.3 / 136 / 6 | 83.6 / 115 / 9 |
| turbo_v5 | 0 / 300 / 8 | 0 / 300 / 16 | 1.4 / 298 / 18 |
| oracle_flow（特权） | 89.3 / 179 / 4 | 97.1 / 108 / 3 | 96.4 / 79 / 2 |

- 本文 v3 不再退化，但也没有超过经典最强。
- 上界与经典之间的空间约 +2 到 +6 pp，T90 快 8–16 s。oracle 消融显示这部分空间主要来自"知道瞬时血流"。
- s=0 和 s=1.5 的结果见 `research/validation/V5_PHYSIO_20261008/REPORT.md`。

**v6 定标**：入口平均流速 0.1 mm/s 太强。狭窄处峰值超过集群最大速度 1 mm/s，经典方法全清只有 4–18%，oracle 也只有 42–49%。0.025 和 0.05 mm/s 的扫描在跑，结果在 `research/validation/V6_FLOW_20261008/`。

## 5. 已知局限与待决问题

1. 血流的绝对尺度：v4 / v5 的流量没有按解剖缩放，大血管里几乎没有流，所以流速相关的失败只出现在小动脉。v6 正在修正。
2. 脉动流是整树按同一波形缩放的准稳态模型，没有 Womersley 效应和管壁弹性。
3. 学习方法都是单种子短预算；正式结论需要 3 种子和更长训练（需用户确认算力）。
4. 2 mm 间距只是磁干扰的代理，没有标定；"全局磁场耦合"（多个集群共用一个场）还没有建模。
5. 下一步计划：在 v6 选定的流速下，重跑所有基线和上界，训练带血流估计辅助头的路线坐标系 PPO。

## 6. 文件地图

- 环境 / 评测：`scripts/benchmark_lysis.py`、`marl/physio_variation.py`、`scripts/summarize_lysis.py`、`scripts/report_lysis_v4.py`
- 本文方法：`scripts/train_lysis_nav.py`（v5 导航）、`marl/lysis_alloc.py` + `scripts/train_lysis_alloc.py`（分配）、`scripts/train_lysis_local.py`（残差 / Turbo）
- 基线：`marl/lysis_baselines.py`（PAC-NMPC、STPG）、`scripts/benchmark_lysis.py` 里的 FlowOracle 和 AdaptiveSettleGuard
- 过程记录：`research/RESEARCH_LOG.md`（2026-10-08 起的条目）
- 训练日志：`research/runs/V4_20261008/*/log.jsonl`、`research/runs/V5_20261008/*/log.jsonl`（权重 *.pt 不入库）
