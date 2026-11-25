# 固定安全协议后的 HCC 导航复测

评估文件：[HCC_FIXED_PARITY_S41_20261123_full.json](runs/HCC_FIXED_PARITY_S41_20261123_full.json)

协议固定为同一批配对场景种子、HCC multifocal anatomy、四档障碍/流速、零延迟、300 s horizon，以及 IPPO 与 pursuit 都使用同一套近目标减速和 WallGuard 安全层。IPPO 使用已有 `EXP0141.../final_model.zip`，没有重新训练 checkpoint。

| 方法 | 回合数 | 成功率 | Strict 成功率 | 丢失率 | 平均壁接触 robot·s |
|---|---:|---:|---:|---:|---:|
| pursuit | 12 | 83.3% | 83.3% | 0.0% | 0.00 |
| IPPO-S41 | 12 | 16.7% | 8.3% | 16.7% | 3.22 |

IPPO 相对 pursuit 的成功率差距为 66.7 个百分点。分难度成功率如下。

| 难度 | pursuit | IPPO-S41 |
|---|---:|---:|
| easy | 100.0% | 33.3% |
| medium | 100.0% | 0.0% |
| hard | 100.0% | 0.0% |
| stress | 33.3% | 33.3% |

本次修复还增加了高速流不可行检测。颗粒在末端 cap 连续贴壁且 `available_control_speed / axial_flow_speed < 1` 达到 8 步时，episode 以 `termination_reason="infeasible_flow"` 结束并计为失败，避免把不可逆物理状态重复喂给训练器。默认训练建议打开 `--flow-curriculum --integration-mode adaptive`，最终评估仍使用 `flow_scale=1.0`。
