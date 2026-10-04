# 方法工作名：V-CTPG / V-CTPG-HRL

登记日期：2026-10-04。用于项目文档和后续新图表；不改写已经冻结的实验编号、源文件或原始结果。

## 图调度机制

**V-CTPG — Vascular Conflict-aware Temporal Plan Graph**

中文：**血管冲突感知时序规划图**。

这是血管场景下的局部 TPG 改进原型工作名。保留 TPG 后缀，明确继承时序依赖调度思想；不能通过改名把原有图算法宣称为原创。

三个具体含义：

- **Vascular**：三维血管的曲折、分叉、狭窄与局部观测限制；已观测曲线上的路程和未知全局路程明确区分。
- **Conflict-aware**：通行依赖依据独立的三维欧氏接近关系建立，拓扑不同的分支也可能冲突。这里的 conflict 不宣称已经标定磁场解耦或保证安全。
- **Temporal Plan Graph**：用通行依赖组织执行和事件驱动的调整；当前是滚动局部原型，不等同于完整全局多活动 TPG 求解器。

## 整套学习方法

**V-CTPG-HRL — Vascular Conflict-aware Temporal Plan Graph-guided Hierarchical Reinforcement Learning**

中文：**血管冲突感知时序图引导的分层强化学习**。

- 上层：图优先级学习。
- 下层：因果时序残差控制与测量响应辅助学习。
- 事件检测、测量处理、图构建和约束投影属于共享工程，不因方法名带 HRL 就自动变成学习贡献。
- 保留上层、下层和联合学习的独立消融。没有训练增益证据之前，不在名称或摘要中加入 optimal、guaranteed-safe、independent magnetic control 等未经证实的表述。

## 后续表格的显示名

| 原始实验标识 | 后续显示名 | 说明 |
|---|---|---|
| `tpg` | V-CTPG (rule-based) | 相同图和候选执行层的传统规则对照 |
| `untrained` | V-CTPG-HRL (untrained) | 无训练策略；不是学习结果 |
| `high` | V-CTPG-HRL (high-only) | 只学习上层 |
| `low` | V-CTPG-HRL (low-only) | 只学习下层 |
| `both` | V-CTPG-HRL | 上下层联合训练 |
| `balanced_1s` | Measured balanced allocation (1 s) | 原有强启发式，名称保持独立 |
| `memory_n1` | Single-cluster sequential baseline | 方法 1 |

独立复现的传统 TPG 若后续加入，仍称 TPG，不能和本项目共享几何/执行改动后的 V-CTPG 规则对照混为一谈。

## 命名检查与证据边界

本日对 `V-CTPG`、英文全称及 `Vascular Temporal Plan Graph` 做过初步网页检索，未找到直接对应的同名血管导航方法；这不是穷尽性查重，也不构成命名独占结论。单独的 CTPG 已见于其他领域及 NeurIPS 2024 的 Cross-Task Policy Guidance，因此后续始终保留 V- 前缀和首次出现的英文全称。

名称是工作名。是否构成有价值的研究改进取决于与强公平基线的完整验证。当前运行时准入受阻、原生失败及无有效性能排名的状态不因命名改变，详见 `STATUS_TPG_20261004.md`。
