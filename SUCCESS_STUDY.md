# 成功率改进研究（2026-09-05）

## 本轮结论

- 流速感知残差 MAPPO：三训练种子平均成功率 **70.83% ± 2.18%**，平均质量清除率 **93.11%**；± 为种子样本标准差。
- 相同独立测试回合：旧 seed43 MVE 为 **21.43%**，无训练流速感知控制器为 **34.64%**。
- 新策略比旧策略提高 **49.40 个百分点**；配对 bootstrap 95% 区间为 **[44.64, 54.40]** 个百分点。
- 每个策略评估 14 场景×20 回合，7 个对照共 1960 回合；成功必须清空全部血栓，物理、奖励与步数上限不变。
- 推荐权重：`experiments/success_study_20260905/selection/flow_guided_43.pt`，由验证集选择，不按测试集挑种子。
- 短板仍在：股腘动脉场景 5.00%、右冠状动脉场景 21.67%，并非所有解剖场景都已解决。
- 全量测试曾 136 passed、1 skipped，但收尾复检连续 9 次 native 崩溃；该稳定性问题未修复。3 MP4、3 GIF、3 张回合截图和 3 张统计图均已保存。

## 对照设计

本轮在不改变模拟器动力学、奖励、horizon 和成功标准的前提下，比较：

- 上轮验证表现较强的 seed43 MVE checkpoint。
- `local`：局部坐标动作的 GAT-MAPPO。
- `guided`：中心线路由引导与学习残差。
- `flow_controller`：无需训练的流速感知控制器。
- `flow_guided`：流速感知控制器加 GAT-MAPPO 有界残差，训练种子 42/43/44。

完整结果：`experiments/success_study_20260905/reports/report.md`。
所有回合记录：`experiments/success_study_20260905/heldout/`。
视频、GIF、截图及解码验证：`experiments/success_study_20260905/artifacts/`。

## 前置条件

```bash
cd ~/桌面/vascular_marl_local.tar./vascular_marl_local
export PY=/home/wj/miniconda3/envs/v/bin/python
```

训练和评估均使用 5 个机器人、名义 3 个血栓、300 步上限、半径 0.0011。
几何观测是 36 维；`world` 是所有旧 checkpoint 的默认动作语义。
新 checkpoint 在 `meta` 记录控制模式、残差幅度和引导速度。
`eval_checkpoint.py`、`make_gif.py`、`watch_gui.py` 通过统一 loader 自动应用控制变换。
不要把新权重的裸神经网络输出直接传入环境；手动调用时使用 `agent.env_action(...)`。

## 重现实验

为避免覆盖本轮结果，请使用新的输出目录。

```bash
bash scripts/run_success_training.sh local 42 0 500000 experiments/success_reproduction
bash scripts/run_success_training.sh guided 42 1 500000 experiments/success_reproduction
for seed in 42 43 44; do
  bash scripts/run_success_training.sh flow_guided "$seed" 0 1000000 experiments/success_reproduction
done
```

训练脚本仅对 native segfault/abort/illegal-instruction/bus-error 有限重试，使用最近的完整 checkpoint 恢复。
配置错误或断言错误不自动重试。
训练验证集每场景 3 回合，seed 基数 400000；最优权重只按验证成功率、清除率选择。
控制器初始策略也参与验证选择，防止训练退化时丢失初始性能。

独立评估单个 checkpoint：

```bash
env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" scripts/eval_success_study.py \
  --policy experiments/success_study_20260905/selection/flow_guided_43.pt \
  --episodes 20 --seed 700000 --output experiments/my_independent_evaluation
```

无训练控制器对照：去掉 `--policy`，添加 `--controller flow_guided`，使用另一个输出目录。
评估保存逐回合 JSONL，并校验 checkpoint SHA256 和配置，原配置中断后可以续跑。
测试种子与验证种子分开，14 个场景均固定评估 20 回合。

## 重新生成报告及媒体

```bash
env -u DISPLAY PYTHONPATH=. "$PY" scripts/report_success_study.py
env -u DISPLAY PYTHONPATH=. "$PY" scripts/render_success_study.py
env -u DISPLAY PYTHONPATH=. "$PY" -m pytest tests/ -q -s
```

报告需要全部已列出的对照结果，不能仅有一份 checkpoint。
视频展示一例改善、同回合旧策略、一例仍失败的案例。
`media_validation.json` 记录真实解码帧数、像素变化、文件大小及对应测试回合。

## 解释边界

- 成功率是所有血栓清零的回合比例，不是质量清除率。
- `flow_guided` 是模型辅助的残差强化学习，不是纯端到端控制，须与无训练控制器对照。
- 新旧训练预算不完全相同，不能仅将全部提升归因于算法名称。
- 场景类别已见过，仅随机几何与回合种子独立，不是未见解剖类别泛化。
- 原世界模型以 world-frame 动作为输入，新控制模式拒绝直接使用旧世界模型。
- 撞壁按完整回合累计，原评估脚本的末步计数不再误标为回合累计。
- 固定物理标度的控制器针对当前模拟器，结果不构成临床效果或安全性证据。
