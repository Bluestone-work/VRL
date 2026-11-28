# NMI 图件 v2 与配套实验（2026-11-28）

## 产物

- 图件：`research/figures/NMI_v2_20261128/`（fig1–fig6 的 PDF/PNG、自动图注 `*_caption.txt`、数据来源清单 `DATA_SOURCES_*.md`）
- 摘要与引言草稿：`research/figures/NMI_v2_20261128/abstract_introduction.md`（含故事线说明、待补实验和待核对的参考文献）
- 出图脚本：`scripts/make_nmi_figures_v2.py`，版式依据 `~/桌面/VRL_NMI_图片布置.pptx`

复现：

```bash
PY=/home/wj/miniconda3/envs/v/bin/python
$PY scripts/make_nmi_figures_v2.py --out research/figures/NMI_v2_20261128          # 全部
$PY scripts/make_nmi_figures_v2.py --out research/figures/NMI_v2_20261128 --only fig3
```

## 本次新跑的实验（`research/NMI_FIG_BASELINES_20261128/`）

| 文件 | 命令 | 结果 |
|---|---|---|
| `stpg_train180.json` / `stpg_holdout100.json` | `scripts/evaluate_classical_manifest.py --methods stpg --split train/holdout_test --clusters 3 --max-episodes 180/100` | 152/180，80/100，0 错误 |
| `pac_train180.json` / `pac_holdout100.json` | 同上，`--methods pac` | 151/180，84/100，0 错误 |
| `geo_tpg*.json`、`geo_off*.json` 及 `*_trace.json` | `scripts/evaluate_fixed_three_lesions.py --allocation geodesic [--no-tpg] --count 6 --seed 2944200000 --trace-seeds 2944200004` | 复现 `navigation_budget_20261127/geodesic_tpg.json`（seed 2944200004 N=3 为 73.9 s，TPG 等待 8.2 robot·s），另存逐步轨迹供 Fig.1c/5a/5b 使用 |

日志与 JSON 同名（`.log`）。两种方法与已有 IPPO/MAPPO/pursuit 的 `scenario_hash` 集合逐一一致，出图脚本在载入时校验配对，不一致即报错。

## 结论（N=3，冻结 navigation_dataset_v1，280 个配对场景）

| 方法 | 训练解剖 /180 | 留出解剖 /100 |
|---|---:|---:|
| IPPO（S41/S42/S43） | 153 / 152 / 153 | 80 / 80 / 80 |
| MAPPO-V2（S41/S42/S43） | 152 / 152 / 153 | 80 / 80 / 80 |
| Pure pursuit | 152 | 80 |
| Switchable TPG | 152 | 80 |
| PAC-NMPC | 151 | 84 |

- IPPO 与 MAPPO 在 840 个配对结果中 837 个相同，支持"高层已解耦集群，独立学习足够"的论点。
- IPPO 与规则方法成功率相同。回放中 IPPO 91.5% 的步选择了经典路由候选（候选 1），7.7% 减速，0.8% 停止。
- PAC-NMPC 在留出解剖上比 IPPO 高 4 个点（配对 4/88/8）。
- N=1/2/3 三靶点实验（pure pursuit 局部控制，只比较高层）：最后到达 86.45 / 94.75 / 44.30 s，全覆盖 6/6、5/6、6/6。
- 当前数据不能支持"RL 优于对比方法"。需要补的实验见草稿末尾的说明。

## 与 pptx 的差异

- Fig.2：pptx 写"冻结运动预测器 μ/σ"，当前 IPPO 代码使用在线最小二乘流估计（`ObservedMotion`），图按代码画；Fig.2d 流估计消融未跑，为占位。
- Fig.5d：没有间距护盾消融数据，只做了 TPG / CCBS / 无协调。
- Fig.6：体外实验，全部为占位。

## 同时提交的 11-27 未提交工作

本分支一并提交了此前未提交的代码改动与结果，因为上述图件依赖它们：CCBS 固定路线适配器修正（`marl/ccbs_adapter.py`、`tests/test_ccbs_adapter.py`）、三靶点评估的 CCBS/轨迹/资源统计（`scripts/evaluate_fixed_three_lesions.py`）、`scripts/evaluate_mr_nav_2d.py`、`scripts/report_navigation_budget.py`，以及 `research/navigation_budget_20261127/`、`research/ccbs_*_20261127/`、`research/replay_remaining_corrected/`、`research/mr_nav_2d_paired_20261127.json`、`research/NAVIGATION_COMPLETION_20261127.md`。500k MAPPO 长训练目录只提交配置与进度日志，不提交检查点。
`tests/test_ccbs_adapter.py`、`tests/test_repaired_comparison.py`、`tests/test_navigation_comparison_controls.py` 共 19 项通过（2026-11-28）。
