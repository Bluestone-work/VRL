# EXP0063：航段进展停滞与承诺式大障碍绕行

2026-10-07，结果产生前登记。沿用 EXP0062 frozen v6 的物理、感知、动作执行、奖励和指标；
只加入局部进展监测与失败恢复。既有 PPO/DAgger 权重不改，不接触密封测试。

## 假设

上一轮 iliac_may_thurner/2600000000 访问12路线站点、目标不变，85次静态事件全在 reference。
位置在约0.5 mm区域振荡，原有“连续2秒位置不动”不能可靠识别这种情况。
APF 在近障碍阈值反复切换；全局近中心线 carrot 也不适合作为大障碍绕行过程中的唯一目标。

## 单变量比较

- legacy_replan：原 v6 hybrid_replan，必须复现之前逐场结果。
- route_replan：只加入5秒无目标航段最佳剩余距离改善的触发，仍使用原回撤执行器。
- progress_bypass：相同触发时，不重复全局重规划；从连接的全局路线取障碍之后的航段子目标，
  以测得大障碍中心/半径生成承诺式切向绕行，低层每步反馈壁/障碍间隙。
- 排除正在溶栓的近目标区域、协调 hold、已有局部承诺和小障碍；不根据解剖/场景编号选择。
- 大障碍只由测得尺寸识别，无静态/动态真实类别。术前健康管腔只作代理，不声称认证安全。
- 保持严格Safe、RSafe和全部接触指标不变。先用已知失败场景pilot，再完整84开发场景验证；
  候选接纳还需在新开发种子确认。重复使用解剖，不称未见解剖或密封测试。
- 不接纳降低严格Safe、增加静态/动态事件或平均壁接触的方案；负结果保留。

入口：`scripts/evaluate_progress_navigation.py`。配置、源哈希、初始状态哈希、审计和watchdog保留。

## Pilot v2

原始 `progress_bypass` 在三个场景中持续重复同一绕行子目标，未完成任务，暂不进入全量。
新增独立消融 `all_semantic`：对任一前方检测障碍使用已有语义绕行，再由横向壁面护盾执行；
不使用真值静态/动态类别，也不修改已冻结 v6。三个场景 pilot 任务完成2/3，平均静态事件16.3/局、
壁接触4.0 s；严格 Safe仍为0/3，所以只能作为待验证候选。全量只运行该消融，防止把失败几何绕行
混入结论。

## Full results: all_semantic with wallguard

Completed 84 paired development scenes. Compared with frozen EXP0062 `switch`:

| Metric | switch | all_semantic + wallguard | delta |
|---|---:|---:|---:|
| strict Safe | 15/84 (17.86%) | 25/84 (29.76%) | +11.90 pp |
| RSafe | 54/84 (64.29%) | 47/84 (55.95%) | -8.33 pp |
| task success | 59/84 (70.24%) | 69/84 (82.14%) | +11.90 pp |
| wall contact | 7.795 s | 10.864 s | +3.069 s |
| longest continuous wall contact | 4.562 s | 4.565 s | +0.003 s |
| static events / episode | 2.964 | 1.357 | -1.607 |
| dynamic events / episode | 1.286 | 0.631 | -0.655 |

Paired bootstrap strict-Safe delta is +11.90 pp, 95% interval [+3.57,+21.43] pp.
The candidate is not accepted: RSafe and wall contact regress, so it is not a drop-in replacement.
It remains useful as a diagnostic: semantic passing substantially reduces obstacle events and raises task
completion, but the same behavior spends too much time near walls in narrow anatomies.

## Failed clearance-gate pilot

A map-clearance gate (`semantic_min_clearance_mm=0.3`) was tested on MCA, basilar and pulmonary pilot scenes.
It produced 0/3 task completion, 89.13 s mean wall contact and 27 s mean longest continuous contact.
The healthy-map clearance estimate is therefore not a reliable switch for this purpose; no full run was started.
The pilot is retained as a negative result and is not part of the accepted candidate.

## Decision

Do not change PPO/DAgger weights or deployment defaults. The best current research direction is not a hard
map-clearance gate. The next iteration should learn or estimate a short-term **wall-contact risk residual**
from measured position/velocity/map offset and use a hysteretic mode switch, while keeping the existing
semantic pass as the obstacle branch. It must be evaluated first on the three narrow-anatomy regressions,
then on all 84 paired scenes. The risk residual must not use simulator wall truth as an input.
