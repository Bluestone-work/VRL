# Decision log

## D001 — 2026-09-20 — Baseline selection
选择当前 `geodesic_v + gat + flow_guided`，而非旧README最高数字。理由：源码/3seed配置/原始验证/权重均存在；接触条件更严格；不把它宣称为所有场景的绝对最佳。先保留同一配置复现。

## D002 — Preserve dirty source
不提交或覆盖用户原分支的已有变更；使用临时Git index创建当前源码快照提交，再建立 `research/EXP_0001-baseline-reproduction` 独立worktree。数据写入原项目research/runs，源码运行用冻结worktree。后续记录变化不修改实验代码。

## D003 — Budget and statistical unit
正式固定42/43/44，各1,000,000 transitions，两个GPU最多各一作业。开发smoke单独目录，seed4242，不计入正式均值。重复运行仅限记录在案的native崩溃恢复；不按分数重试。

## D004 — Evaluation separation
主复现指标沿用历史final validation。额外900000-seed评估仅为独立development validation，不对它调参，不包装成unseen topology test。未来EXP_0002审核后再固定最终test集。

## D005 — Research boundary
本轮不修改核心算法，不启用WM/课程/42维观测/新奖励。新增执行、审计、只读诊断和图表脚本属于工程仪表，不归因为算法创新。记录未测指标N/A。

## D006 — 2026-09-20 — Gate and next scientific question
EXP_0001三seed原协议success、removal、长度、wall均与归档一致，actor/critic张量也一致；通过预登记工程复现gate。保留CPU/GPU回放差异和213个开发验证失败回合。下一步为EXP_0002评估协议核验，随后完善仪表；不能用基线可复现替代世界模型预测gate。

## D007 — Separate execution and result versions
执行commit固定为`8c159ab153e2f651a5cbdfa7c22706c86dc203d8`。报告生成器、完成后的记录、图表和精简原始证据另存`research/EXP_0001-results`分支；最终commit及文件数见`runs/EXP_0001/results_archive.json`。大checkpoint/trace/log留在本地且记录SHA256，不把只有hash误称为数据备份。用户分支和index不切换。

## D008 — EXP_0002 scope and future evaluation device
用户继续后执行EXP_0002，不提前进入WM。主要变量是同权重同episode的CPU/GPU设备；未来统一评估预先固定cuda:0，不按此次分数选择。配置/完整内容hash/未来split属于测量校验；不改变受测控制行为。

## D009 — Historical coverage and prospective split
历史无全量训练树清单，只能核对保存checkpoint中的树；报告generation覆盖不足，不宣称历史无泄漏。未来same_family_geometry_v1固定40/10/20实例每场景，禁止test策略评估；若hash重叠直接失败而非重抽。新清单未接入训练前不称作已实现的训练隔离。

## D010 — EXP_0002 outcome and Phase 0 exit
840对设备回合出现10个success翻转；同设备84对重复、CPU父实验840回合、历史GPU接口42回合均通过。保留已预先选择的cuda:0，不能因其本轮success高0.9524pp而声称方法提升。清除率低0.0414pp、GPU完整接口更慢均如实报告。

## D011 — Gate scope and next experiment
Phase0允许进入EXP_0003科学仪表。未来same-family split精确hash互斥，但训练器尚未强制消费；历史训练全量身份不完整。主要创新仍需后续评估/预测gate，不将本轮协议通过解释为H1–H5或新颖性成立。下一实验只验证测量完整性，split接入训练与数值物理修复另立问题。

## D012 — EXP_0002 archival boundary
执行commit为`c7830951f2674aa7899b25da2ff0d49d8e81d8ef`。最终文档、图、分析脚本、全部精简原始JSONL和split清单写入`research/EXP_0002-results`分支；commit回执在`runs/EXP_0002/results_archive.json`。NPZ轨迹留本地并保存hash清单。原用户branch/index和EXP_0001文件只读校验，未切换或覆盖。
