# 自动科研系统使用指南

## 📋 概述

自动科研系统会：
1. 建立基线（当前最佳配置）
2. 依次添加创新点
3. 自动训练、记录、对比每个实验
4. 生成详细的对比报告

所有数据自动保存，可随时中断和恢复。

---

## 🚀 快速开始

### 1. 快速测试（5-10分钟）

验证系统是否正常工作：

```bash
python auto_research_quick_test.py
```

这会运行3个小规模实验：
- 基线（bifurcation场景，16k步）
- 课程学习
- GAT（配置记录）

### 2. 完整自动科研（数小时）

```bash
python auto_research.py
```

这会运行6个完整实验：
- 基线：当前最佳配置（MCA场景，500k步）
- 创新点1：课程学习（3阶段）
- 创新点2：图注意力网络
- 创新点3：智能体通信
- 创新点4：层次化策略
- 创新点5：组合最优

---

## 📊 结果位置

### 目录结构

```
auto_research_results/
├── experiment_history.json          # 所有实验的历史记录
├── research_report.md               # 自动生成的对比报告
├── 000_baseline/                    # 基线实验
│   ├── config.json                  # 实验配置
│   ├── result.json                  # 性能指标
│   ├── best_policy.pt               # 最佳模型
│   ├── final_policy.pt              # 最终模型
│   ├── training_log.jsonl           # 训练日志（每个episode一行）
│   ├── train_stdout.log             # 标准输出
│   └── train_stderr.log             # 标准错误
├── 001_curriculum_learning/
├── 002_graph_attention/
├── 003_communication/
├── 004_hierarchical/
└── 005_combined_best/
```

### 关键文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `experiment_history.json` | 所有实验的元数据 | 恢复、对比 |
| `research_report.md` | Markdown报告 | 论文素材 |
| `XXX/training_log.jsonl` | 逐episode训练日志 | 绘制学习曲线 |
| `XXX/best_policy.pt` | 最佳模型 | 评估、演示 |
| `XXX/result.json` | 性能指标 | 快速对比 |

---

## 📈 性能指标

系统自动记录以下指标：

### 训练指标
- `final_success_rate`: 最终成功率（最后100个episode平均）
- `final_removal_rate`: 最终溶解率
- `final_return`: 最终平均回报
- `first_contact_step`: 首次接触血栓的平均步数（探索效率）
- `wall_collision_rate`: 碰壁率
- `robot_collision_rate`: 机器人间碰撞率

### 学习效率
- `peak_success_rate`: 训练过程中达到的最高成功率
- `convergence_step`: 收敛步数（达到峰值90%的步数）

### 相对改进
- `improvement_vs_baseline`: 相对基线的改进百分比
  - `success_rate`: 成功率改进
  - `removal_rate`: 溶解率改进
  - `return`: 回报改进
  - `convergence_speed`: 收敛速度改进（负值=更快收敛）

---

## 🔬 实验配置

### 基线配置

```python
ExperimentConfig(
    name="baseline",
    description="当前最佳配置",
    innovation="无（基线）",
    
    # 环境
    scenario="mca_stroke",        # MCA中风场景（复杂）
    n_envs=64,                    # 64个并行环境
    robots=3,                     # 3个机器人
    clots=3,                      # 3个血栓
    horizon=300,                  # 最大300步
    
    # 训练
    timesteps=500000,             # 500k transitions
    obs_mode="geometric",         # 36维几何观测
    reward_mode="milestone",      # 里程碑奖励
    
    # 网络
    hidden_dim=128,
    actor_lr=1e-4,
    critic_lr=1e-3,
    
    # 其他
    seed=42,
    device="cuda",
)
```

### 创新点1：课程学习

```python
curriculum_stages=3  # 3阶段
```

阶段序列：
1. `bifurcation` + 1血栓（简单）
2. `multilevel` + 2血栓（中等）
3. `mca_stroke` + 3血栓（困难）

每个阶段需达到60%成功率才进入下一阶段。

### 创新点2：图注意力网络（GAT）

```python
use_gat=True
```

**注意**：GAT网络结构尚未实现，当前只记录配置。
后续需要在 `marl/maddpg_policy.py` 中实现：
- 使用 `torch_geometric.nn.GATConv`
- 输入：`obs["nodes"]` + `obs["adjacency"]`
- 输出：聚合后的节点特征

### 创新点3：智能体通信

```python
use_communication=True
```

**注意**：通信机制尚未实现。
后续实现方案：
- 每个智能体广播意图向量（2-4维）
- 邻居接收并加权聚合
- 聚合后的信息作为额外观测输入

### 创新点4：层次化策略

```python
use_hierarchical=True
```

**注意**：层次化架构尚未实现。
后续实现方案：
- 高层：目标分配器（为每个智能体分配血栓）
- 低层：子群协调器（执行分配的任务）
- 高层每N步更新，低层每步更新

### 创新点5：组合最优

```python
curriculum_stages=3
use_gat=True
use_communication=True
```

组合表现最好的创新点（假设前3个最优）。

---

## 🛠 自定义实验

### 修改研究计划

编辑 `auto_research.py` 中的 `define_research_plan()`：

```python
def define_research_plan() -> list[ExperimentConfig]:
    # 添加你的实验
    my_experiment = ExperimentConfig(
        name="my_innovation",
        description="我的创新点描述",
        innovation="创新点简要说明",
        
        # 修改参数
        scenario="mca_stroke",
        timesteps=600000,
        hidden_dim=256,  # 加大网络
        
        # 启用某些创新点
        use_gat=True,
        curriculum_stages=3,
    )
    
    return [baseline, my_experiment, ...]
```

### 添加新的创新点参数

1. 在 `ExperimentConfig` 中添加字段：

```python
@dataclass
class ExperimentConfig:
    # ...
    use_my_innovation: bool = False
```

2. 在 `_build_training_command()` 中传递参数：

```python
if config.use_my_innovation:
    cmd.extend(["--use-my-innovation"])
```

3. 在 `train_vector_enhanced.py` 中处理：

```python
p.add_argument("--use-my-innovation", action="store_true")
# ...
if args.use_my_innovation:
    # 实现你的创新点
    pass
```

---

## 📉 分析结果

### 1. 查看报告

```bash
cat auto_research_results/research_report.md
```

报告包含：
- 每个实验的详细信息
- 对比表格
- 相对基线的改进百分比

### 2. 绘制学习曲线

```python
import json
import matplotlib.pyplot as plt

# 加载训练日志
episodes = []
with open("auto_research_results/000_baseline/training_log.jsonl") as f:
    for line in f:
        episodes.append(json.loads(line))

# 提取数据
timesteps = [ep["timestep"] for ep in episodes]
returns = [ep["return"] for ep in episodes]
success = [ep["success"] for ep in episodes]

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(timesteps, returns)
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Return")
ax1.set_title("Training Curve - Return")

# 滑动平均成功率
window = 50
success_smooth = [sum(success[max(0,i-window):i+1]) / min(i+1, window) 
                  for i in range(len(success))]
ax2.plot(timesteps, success_smooth)
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Success Rate")
ax2.set_title("Training Curve - Success Rate (50-ep moving avg)")

plt.tight_layout()
plt.savefig("learning_curve.png")
```

### 3. 对比多个实验

```python
import json
from pathlib import Path

results_dir = Path("auto_research_results")
experiments = []

for exp_dir in sorted(results_dir.glob("*_*/")):
    result_file = exp_dir / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            experiments.append(json.load(f))

# 对比成功率
for exp in experiments:
    name = exp["config"]["name"]
    success = exp["final_success_rate"]
    improvement = exp.get("improvement_vs_baseline", {}).get("success_rate", 0)
    print(f"{name:25s} {success:.1%} ({improvement:+.1f}%)")
```

### 4. 评估最佳模型

```bash
# 使用最佳模型观看演示
python watch_gui.py \
  --policy auto_research_results/005_combined_best/best_policy.pt \
  --scenario mca_stroke \
  --episodes 10

# 在测试集上评估
python evaluate.py \
  --policy auto_research_results/005_combined_best/best_policy.pt \
  --scenario mca_stroke \
  --episodes 100 \
  --seed 999
```

---

## ⚠️ 注意事项

### 1. 创新点实现状态

当前状态：
- ✅ 课程学习：已实现
- ⚠️ GAT：仅配置记录，网络未实现
- ⚠️ 通信：仅配置记录，机制未实现
- ⚠️ 层次化：仅配置记录，架构未实现

系统会打印警告：
```
⚠️  注意: GAT/Communication/Hierarchical 网络尚未实现
    当前使用基线网络结构，只记录配置用于后续实现
```

### 2. 训练时间估算

单个实验（500k步，64环境，3机器人）：
- CPU: 约60-90分钟
- GPU (RTX 4090): 约20-30分钟

完整自动科研（6个实验）：
- CPU: 约6-9小时
- GPU: 约2-3小时

### 3. 磁盘空间

每个实验约占用：
- 训练日志：10-50 MB
- 模型文件：1-5 MB
- 标准输出：1-10 MB
- **总计**：约20-100 MB/实验

完整研究计划：约500 MB

### 4. 中断和恢复

训练可以随时中断（Ctrl+C）。系统会：
- ✅ 保存已完成实验的结果
- ✅ 询问是否继续下一个实验
- ❌ 不会恢复未完成的实验（需要重新运行）

---

## 🔧 故障排查

### 问题1：CUDA out of memory

**解决方案**：减少并行环境数

```python
n_envs=32  # 原来64
```

或减少批次大小：

```bash
--batch-size 128  # 原来256
```

### 问题2：训练超时（2小时）

**解决方案**：减少训练步数

```python
timesteps=300000  # 原来500000
```

或在 `auto_research.py` 中增加超时：

```python
timeout=7200  # 改为 10800 (3小时)
```

### 问题3：日志解析失败

**症状**：`final_success_rate = 0.0`，但训练明显有输出

**解决方案**：
1. 检查 `training_log.jsonl` 是否存在
2. 手动检查日志格式
3. 修改 `_parse_training_output()` 的解析逻辑

### 问题4：实验失败但想继续

系统会询问：
```
是否继续下一个实验？[y/N]:
```

输入 `y` 继续，`N` 停止。

---

## 📝 论文写作建议

### 实验章节

```markdown
## 实验设置

我们在改进的血管环境中进行了系统的消融研究。基线采用...

## 结果

表1展示了各创新点的性能对比：

| 方法 | 成功率 | 溶解率 | 收敛步数 |
|------|--------|--------|----------|
| 基线 | 82.3% | 89.1% | 250k |
| +课程学习 | 91.5% (+9.2%) | 95.3% | 180k |
| +GAT | 93.2% (+10.9%) | 96.7% | 170k |
| 组合 | 96.1% (+13.8%) | 98.2% | 150k |

## 分析

课程学习显著提升了探索效率，将收敛速度提升了28%...
```

### 图表

- 学习曲线对比（所有方法叠加）
- 消融研究柱状图
- 成功率 vs 训练步数
- 各场景泛化性能

---

## 🎯 下一步

1. **运行快速测试**：验证系统工作
   ```bash
   python auto_research_quick_test.py
   ```

2. **实现缺失的创新点**：GAT、通信、层次化

3. **运行完整研究**：
   ```bash
   python auto_research.py
   ```

4. **分析结果**：生成图表、撰写论文

5. **迭代改进**：根据结果调整实验设计

---

## 📚 相关文件

- `auto_research.py` - 主系统
- `auto_research_quick_test.py` - 快速测试
- `scripts/train_vector_enhanced.py` - 增强版训练脚本
- `IMPROVEMENTS.md` - 环境改进文档
- `README.md` - 项目总体说明

---

## 🤝 贡献

添加新的创新点：
1. 在 `ExperimentConfig` 中添加参数
2. 在训练脚本中实现
3. 在 `define_research_plan()` 中添加实验
4. 运行并记录结果
