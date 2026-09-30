# vascular_marl_local

> 当前研究进度（2026-09-30）：MCA体外低流量仿真已支持血栓、机器人和动态粒子全部随机初始化、纯MAPPO训练及VTK三维回放。EXP29三seed新增50万步的完整清栓率为33%/15%/27%，无粒子碰撞且全机器人保留的清栓率为8%/2%/5%，尚未达到80%。EXP30修复PPO带mask小批次损失分母，正在比较原actor学习率与降低学习率。世界模型暂缓接入。详见[EXP29环境说明](research/experiments/EXP_0029_ALL_RANDOM_AVOIDANCE.md)、[EXP30修正实验](research/experiments/EXP_0030_PPO_REPAIR.md)与[续接记录](research/CONTINUATION_20260928.md)。下文早期环境的高成功率属于历史实验，不代表当前随机MCA任务。

当前验证：`python -m pytest tests -q`，401通过、1跳过；EXP30两组分别完成16,384环境步GPU预检。不要对含历史源码快照的整个仓库执行pytest收集。训练权重、逐步日志、原始大样本诊断和视频留在本机，Git保留源码、配置、研究说明和轻量验证证据；新机器需按配置重新验证/训练或取得协议中哈希对应的归档权重。

自包含的血管内微型机器人 MARL 训练包，用于在本机复现实验并用 PyBullet GUI
实时观察仿真。

## 内容

```
environments/vessel_geometry.py         血管几何（分支拓扑 + 测地路由 + Frenet 框架）
environments/vascular_3d_marl_env.py    环境（每机器人独立 3D 速度控制 + 接触溶栓）
environments/vector_env.py              向量化环境（一次推进 N 个 episode）
marl/maddpg_policy.py                   MADDPG（共享 actor + 置换不变中心化 critic）
scripts/train_vector.py                 训练入口（向量化，推荐）
scripts/train_vascular_maddpg.py        训练入口（单环境）
scripts/ab_obs_vector.py                观测消融（向量化）
scripts/ab_obs_mode.py                  观测消融（单环境交叉验证）
tests/                                  100 个测试，含所有已修 bug 的回归测试
watch_gui.py                            GUI 实时观看已训练策略
make_gif.py                             离屏渲染导出 GIF
checkpoints/run_2_best_policy.pt        run_2 旧权重（16.6% 成功率，legacy 观测）
ab_obs_vector.json                      观测消融的实测结果
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

没有 CUDA 显卡也可以，所有命令加 `--device cpu`。

跑测试：

```bash
python -m pytest tests/ -q
```

## 这一版改了什么，以及为什么

上一版有一个已确认的瓶颈：约 45% 的 episode 里机器人**从头到尾没碰到过任何
血栓**（`contact_miss`）。两轮奖励函数改动都没有改善，反而更差。原因是这不是
奖励塑形问题，而是**可观测性问题**加三个物理 bug。

### 实测结果

固定物理、奖励、网络、种子，只切 `obs_mode`，3 个种子 × 300k transitions
（`scripts/ab_obs_vector.py`，结果存在 `ab_obs_vector.json`）：

| | legacy (20 维) | geometric (36 维) |
|---|---|---|
| **contact_miss** | 0.354 | **0.002** |
| **success** | 0.165 | **0.923** |
| **removal** | 0.373 | **0.971** |
| return | +27.3 | **+95.9** |

三个种子结果一致（geometric 的 contact_miss 分别是 0.005 / 0.001 / 0.001）。
诊断是对的：瓶颈在观测，不在奖励。

### 1. 观测里没有血管几何（主要原因）

原来 20 维观测里，只有第 16 维 `clearance`（一个标量径向余量）和血管有关。
机器人知道"墙很近"，但不知道**墙在哪个方向**、**血管往哪延伸**，也感知不到
分叉的存在。唯一可用的导航信号是到血栓的欧氏方向——而在分叉血管里，这个
方向经常指向管壁。

现在 `obs_mode="geometric"` 是 36 维，加了：

| 维度 | 内容 |
|---|---|
| 3:6 | 速度（局部 Frenet 坐标系） |
| 6:15 | **路由化前视**：沿测地路径前方 3 个站点的相对位置 |
| 15:18 | 管壁外法向（局部坐标系） |
| 18:20 | 径向余量、局部管腔半径 |
| 20 | 被血栓阻塞后的半径比 |
| 21:24 | 局部血流速度（局部坐标系） |
| 24 | 近壁润滑系数 |
| 25:29 | 血栓方向 + **测地距离** |
| 29:32 | 欧氏距离、接触标志、剩余质量比 |
| 32:36 | 最近邻相对位置与距离 |

**路由化前视是最关键的一项**：它沿着通往该 agent 目标血栓的测地路径取样，所以
在分叉处会朝正确的分支弯曲。实测在分叉点，血栓在上/下分支两种情况下，
前视特征块的差异达到 2.66（归一化单位），而 legacy 观测在这两种情况下几乎
完全相同。方向性量全部表达在局部 Frenet 坐标系里，否则网络得在血管每一点
分别学习"哪边是下游"。

`obs_mode="legacy"` 保留了原来的 20 维布局，所以这个改动是可对照的 A/B，
不是一次不可比较的重写。

### 2. 最近中心线点用的是全局 argmin（bug）

在分叉处，上分支里的机器人的最近中心线点完全可能落在**下分支**上，于是它拿到
错误的血流方向和错误的管壁投影。现在用前一步的站点做 hint，把搜索限制在图上
若干跳以内——这就把"运动连续性"编码进去了：不经过分叉点就不可能换分支。
回归测试 `test_hint_prevents_branch_jump`、`test_no_branch_jumping_during_an_episode`。

### 3. 奖励塑形用欧氏距离（bug）

在分叉血管里，"欧氏距离下降的方向"经常就是穿墙方向。实测：把机器人放在一条
分支、血栓放在另一条分支，沿直线走的步子里有 **64%** 是欧氏距离变短但
沿血管距离变长的——也就是说旧的塑形项在**付钱让 agent 走错分支**。
（这个比例随血栓远端程度上升：中点约 30%，末端约 81%。）

现在改用 Dijkstra 测地距离，并且做了亚站点插值——只取最近站点的距离场会把
信号量化到站点间距，机器人走满一步却停在同一站点时塑形恰好为 0，形成"几步
静默然后跳变"的阶梯信号。回归测试 `test_shaping_is_continuous_not_a_staircase`。

血栓分配也从欧氏最近改成测地最近：直线最近的血栓常在另一条分支，只能退回
分叉点才够得着。

### 4. 血流是一个全局常数（物理错误）

`stenotic` 场景把半径乘了 0.42~0.58，但流速不变——按连续性方程 A·v = const，
半径减半流速应该约 4 倍。狭窄段变成了"更窄但一样慢"，把这个场景最有意思的
挑战抹掉了。

现在流速满足连续性（`v ∝ (r_ref/r)²`，并设上限防止近全堵时速度爆掉）和
Murray 定律（每条分支按 `flow_fraction` 分配流量，所以子血管比母血管慢）。
半径按 `r_parent³ = Σ r_daughter³` 递减。

### 5. 血栓不影响流场

血栓原来是无体积的点。现在它按质量比例阻塞管腔（`clot_occlusion=0.65`），
所以溶栓既移除目标又重新打开血管——溶栓和流场真正耦合起来了。

### 6. 贴墙是免费的（奖励漏洞）

Poiseuille 剖面让贴壁流速降到近 0，所以**贴墙是躲避逆向血流最省事的办法**，
代价只有一个 0.1 的惩罚。这个漏洞现在从**物理上**堵掉而不是加惩罚：近壁润滑
阻力让贴墙的机器人推力上不去（`lubrication_floor=0.35`）。撞墙也不再只是把
位置拉回、动量原样保留，而是把法向速度分量清零。机器人之间的重叠现在会被
推开，不只是罚分。

### 7. 训练侧算力浪费

- **`info["agent_rewards"]` 算了但从来没用**：只存 team scalar，导致 N 个
  critic 学的是**完全相同的函数**——N 倍算力换零信息。现在存 per-agent 奖励。
- **`clot_state` 被整个丢掉**：现在 flatten 后作为全局状态喂给中心化 critic
  （CTDE 下 critic 看全局是正当的）。
- **actor 更新用 buffer 里的旧动作**：标准 MADDPG 用其他 agent **当前** actor
  的输出再 detach。用旧动作等于在过时的联合动作分布上评估 Q。
- **critic 是拼接输入**：agent 数一变就得重训，而且得自己学会"peer 3 和
  peer 7 可以互换"。现在对 peer 做 mean-pooling，置换不变，宽度无关。
- **参数不共享**：机器人是同质的，独立网络把经验切成 N 份学 N 份同样的东西。
  现在默认共享（`--no-share-parameters` 可关掉），有效样本量乘 N。
- **truncation 当成 terminal**：时间上限截断不是终止状态，当成终止会教会
  值函数"世界在 horizon 处结束"。现在只有 `terminated` 才切断 bootstrap。
- **100k 步后探索完全归零**：确定性策略驱动 off-policy replay 就不再产生新
  经验了。现在噪声下限保持在 0.05。

一个实测的性能问题：梯度更新比环境步贵得多（更新 5.7 ms vs 环境步 1.2 ms），
所以默认每步一次更新时，**优化器而不是仿真器**是吞吐上限。`--update-freq 4`
可以用样本效率换 wall-clock。critic 的 per-agent 循环也批量化了，数学上等价
（`test_policy_gradient_matches_explicit_maddpg` 逐梯度验证，最大误差 6e-11），
但 N=8 时从 39.9 ms 降到 10.6 ms。

## 训练

推荐用向量化环境（快约 60 倍，实测 3900 transitions/s @ 4090，单环境约 65）：

```bash
python scripts/train_vector.py \
  --n-envs 64 --robots 3 --clots 3 --horizon 300 \
  --timesteps 800000 --obs-mode geometric \
  --curriculum --device cuda --seed 42
```

单环境版本（逐 episode 日志，调试用）：

```bash
python scripts/train_vascular_maddpg.py \
  --robots 3 --clots 3 --horizon 300 \
  --timesteps 200000 --obs-mode geometric \
  --curriculum --scale-noise --update-freq 4 \
  --seed 42 --log-interval 20 --device cuda
```

课程学习从 1 个近端血栓起步，成功率过 50% 才放宽（难度影响血栓数量和远端
程度）。诊断出的瓶颈是探索——碰不到血栓的 swarm 根本观察不到溶栓奖励，也就
没有东西可学。

日志写到 `logdir/.../run_K/`，每个 episode 一行 JSON，含 `contact_miss`
（这一版重点关注的指标）。

复现旧的 run_2（对照组）：

```bash
python scripts/train_vascular_maddpg.py \
  --robots 3 --clots 3 --horizon 300 --timesteps 200000 \
  --reward-mode baseline --obs-mode legacy --no-share-parameters \
  --no-critic-state --seed 42 --device cpu
```

注意即使这样也不是逐位复现：分支感知的最近点查询、测地塑形、连续性流场这些
物理修正没有开关，它们是 bug 修复而不是可选变体。

## 观测消融

```bash
# 向量化版本，3 种子 × 300k transitions，4090 上约 10 分钟
python scripts/ab_obs_vector.py --transitions 300000 --seeds 3 --device cuda

# 单环境版本（慢很多，作为独立交叉验证）
python scripts/ab_obs_mode.py --timesteps 40000 --seeds 3 --device cpu
```

固定物理、奖励、种子，只切 `obs_mode`，报告 contact_miss / success / removal。
向量化环境在 `tests/test_vector_env.py` 里逐项对齐过单环境（观测、动力学、
奖励、溶栓、阻塞、测地距离），所以两边的差异可以归因到观测而不是仿真器。

## 看已训练的策略跑

```bash
python watch_gui.py --policy logdir/.../best_policy.pt --episodes 3 --device cpu
```

鼠标左键拖动旋转，滚轮缩放。`--fps` 控制播放速度。不加 `--policy` 就是随机
动作。导出 GIF（无头机器也能跑）：

```bash
python make_gif.py --policy logdir/.../best_policy.pt --out out.gif
```

旧的 `checkpoints/run_2_best_policy.pt` 是 legacy 观测训练的，要看它得加
`--obs-mode legacy`（不加会直接报错说明原因，而不是抛 shape 错误）。

## 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| robots / clots | 3 / 3 | 12 个机器人在旧版本无法收敛 |
| horizon | 300 | 改这个会让不同 run 无法按 episode 对比 |
| obs_dim | 36 / 20 | geometric / legacy |
| action_dim | 3 | 每机器人独立 3D 速度命令（非共享磁场） |
| max_speed | 0.018 | 归一化单位 |
| flow_speed | 0.004 | 入口平均流速，实际流速按连续性缩放 |
| clot_contact_radius | 0.035 | 进入这个半径才开始溶栓 |
| lysis_rate | 0.018 | 每步每机器人的溶解量 |
| lysis_saturation | 4.0 | 单个血栓上超过这个数的机器人不再增加效率 |
| clot_occlusion | 0.65 | 满质量血栓阻塞管腔的比例 |
| lubrication_floor | 0.35 | 贴壁时剩余推力比例 |
| actor_lr / critic_lr | 1e-4 / 1e-3 | |
| gamma / tau | 0.99 / 0.01 | |
| batch / buffer | 256 / 100000 | |

动作空间是每个机器人**各自**的 3D 速度命令，不是整个集群共享一个磁场命令。
这个环境刻意去掉了磁驱动约束，否则多智能体 RL 无从谈起。

## 关于 Isaac Sim

**不要用 Isaac Sim 跑训练主循环。** Isaac Sim / PhysX 是刚体和关节机器人
仿真器；这个任务是微米尺度低雷诺数流体 + 接触溶解，不在它的物理模型里，
它没有内置 CFD。每个并行环境还要背一整套 USD 场景和渲染栈，用它跑这个训练会
比现在的 numpy 环境**更慢**。

真正该用的是 **Warp**（`pip install warp-lang`，或用 Isaac Sim 自带的
`extscache/omni.warp.core-*/warp`）。用它写 GPU kernel 做数千并行 episode 的
推进、血管 SDF 碰撞查询、流场插值——这是数量级吞吐提升的路径，而且是纯
Python DSL，不需要写 CUDA。

流场的正确解法是**离线 CFD + 在线插值**：真实血管几何（VMR 数据库或病人 CT
分割）→ SimVascular / OpenFOAM 跑一次稳态或脉动 CFD → 速度场存成规则网格 →
训练时三线性插值查流速，每步 O(1)。

Isaac Sim 值得用在：导入真实血管 USD/STL 做**可视化和 demo**（出论文图和答辩
视频比 PyBullet 那堆半透明球体好看得多）；如果以后研究对象换成**导丝/导管**
这类弹性杆，那 Isaac Sim + Newton（`isaacsim.physics.newton`，Warp-based 可微
引擎）和 Isaac for Healthcare 就非常对口。

## 还没做的

- 真实血管几何（现在是 Catmull-Rom 样条 + Murray 定律的合成树）
- 离线 CFD 流场（现在是连续性 + Poiseuille 的解析近似）
- Warp GPU kernel（向量化环境现在是 numpy，把 `step` 移到 Warp 还能再快一个
  数量级；接口已经是批量的，可以原地替换）
- GAT 策略（`adjacency` 已经在观测里输出，但当前 actor 没用图结构）
- 脉动流（现在是稳态）
- 向量化环境每个 batch 共享一棵血管树（见 `vector_env.py` 顶部说明）；要完全
  独立的几何得跑多个实例
