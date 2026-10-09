# 明早验收入口

- `研究进展与DRL方法_20261006.docx`：持续更新的中文报告、网络图和真实 GUI 图片。
- `REPORT.md` / `SUMMARY.json`：逐方法数据与配对差异；只报告开发结果。
- `STATUS.json`：后台队列阶段、完成作业及所有失败。计划 2026-10-06 08:35 前结束，预留报告时间。
- `QUEUE_PROTOCOL.json`：3 方法 × 3 种子、480 PPO updates/模型、相同观测/预算/场景。
- `eval/`：全部正式评估原始 JSONL，保留失败作业日志，不补造结果。
- `PAIRING_AUDIT.json`：实际初始化哈希、重复检查、训练/开发场景重叠检查。
- `PILOT_COMPARISON.json`：已完成先导结果。RL 与传统追踪均为 5/12 Safe，清除均 100%，目前没有超过基线。
- `source_snapshot/` / `SOURCE_SHA256.json`：冻结的训练与评估源代码。图像展示修复在主目录中独立更新，不修改冻结的算法/物理；最终图像版本另有 `MEDIA_FINALIZED.json`。
- `jobs/` / `overnight_supervisor.log`：逐作业日志及调度日志。

图像位于 `research/figures/EXP0060_20261006/`：`network_architecture.png/.svg`、`gui_pilot/`（已完成的真实先导策略回放）、完成后 `gui/`（正式 seed0 策略）。截图固定初始、10 s、30 s、最后记录时刻，不按成功与否挑场景。

既有 Claude 初稿不覆盖。本轮完整训练、统计显著性、跨真实装置迁移均不能在尚未完成时宣称成功；2 mm 是未经实验校准的磁场间距代理。
