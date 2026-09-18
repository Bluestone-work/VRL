# 🎯 自动科研系统运行状态

## ✅ 当前状态

**自动科研系统正在后台运行中！**

- 进程ID: 2288615
- 启动时间: 2026-08-21 21:24
- 日志文件: `auto_research_practical.log`
- 结果目录: `auto_research_results/`

## 📊 研究计划

系统正在自动运行以下4个实验：

| # | 实验名称 | 创新点 | 场景 | 训练步数 | 预计时间 |
|---|----------|--------|------|----------|----------|
| 1 | baseline | 无（基线） | bifurcation | 200k | 10-15分钟 |
| 2 | curriculum_2stage | 2阶段课程 | bifurcation | 200k | 10-15分钟 |
| 3 | complex_scene | MCA场景 | mca_stroke | 250k | 15-20分钟 |
| 4 | curriculum_mca | 课程+MCA | mca_stroke | 250k | 15-20分钟 |

**总预计时间**: 约1-2小时

## 🔍 监控进度

### 方法1：实时监控脚本

```bash
python monitor_progress.py
```

显示：
- 已完成的实验数量
- 最新实验的性能指标
- 运行时间

按 Ctrl+C 停止监控（不会影响训练）

### 方法2：查看日志文件

```bash
# 查看最新100行
tail -100 auto_research_practical.log

# 实时跟踪日志
tail -f auto_research_practical.log
```

### 方法3：查看结果目录

```bash
# 查看已完成的实验
ls -lh auto_research_results/

# 查看历史记录
cat auto_research_results/experiment_history.json
```

### 方法4：查看报告

```bash
# 实验完成后会自动生成报告
cat auto_research_results/research_report.md
```

## 📁 结果文件位置

每个实验的结果保存在：
```
auto_research_results/
├── 000_baseline/
│   ├── best_policy.pt          # 最佳模型
│   ├── final_policy.pt         # 最终模型
│   ├── training_log.jsonl      # 训练日志（每个episode一行）
│   ├── config.json             # 实验配置
│   ├── result.json             # 性能指标
│   ├── train_stdout.log        # 训练输出
│   └── train_stderr.log        # 错误日志
├── 001_curriculum_2stage/
├── 002_complex_scene/
├── 003_curriculum_mca/
├── experiment_history.json     # 所有实验的历史记录
└── research_report.md          # 自动生成的对比报告
```

## 🎓 实验完成后

### 1. 查看对比报告

```bash
cat auto_research_results/research_report.md
```

报告包含：
- 所有实验的性能对比表
- 每个实验的详细指标
- 相对基线的改进百分比

### 2. 观看最佳策略

```bash
# 基线
python watch_gui.py \
  --policy auto_research_results/000_baseline/best_policy.pt \
  --scenario bifurcation --episodes 5

# 课程学习
python watch_gui.py \
  --policy auto_research_results/001_curriculum_2stage/best_policy.pt \
  --scenario bifurcation --episodes 5

# MCA场景
python watch_gui.py \
  --policy auto_research_results/002_complex_scene/best_policy.pt \
  --scenario mca_stroke --episodes 5
```

### 3. 分析训练曲线

```python
import json
import matplotlib.pyplot as plt

# 加载训练日志
with open("auto_research_results/000_baseline/training_log.jsonl") as f:
    episodes = [json.loads(line) for line in f]

# 提取数据
timesteps = [ep["timestep"] for ep in episodes]
returns = [ep["return"] for ep in episodes]
success = [ep["success"] for ep in episodes]

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(timesteps, returns, alpha=0.3)
ax1.set_ylabel("Return")
ax1.set_title("Training Curve - Return")
ax1.grid(True)

# 滑动平均
window = 50
success_smooth = [sum(success[max(0,i-window):i+1]) / min(i+1, window) 
                  for i in range(len(success))]
ax2.plot(timesteps, success_smooth)
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Success Rate")
ax2.set_title("Training Curve - Success Rate (50-ep moving avg)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print("图表已保存: training_curve.png")
```

### 4. 对比多个实验

```python
import json
import pandas as pd
from pathlib import Path

results_dir = Path("auto_research_results")
data = []

for exp_dir in sorted(results_dir.glob("*_*/")):
    result_file = exp_dir / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            result = json.load(f)
            data.append({
                "实验": result["config"]["name"],
                "创新点": result["config"]["innovation"],
                "成功率": f"{result['final_success_rate']:.1%}",
                "溶解率": f"{result['final_removal_rate']:.1%}",
                "回报": f"{result['final_return']:+.2f}",
                "收敛步数": result["convergence_step"],
                "训练时间(分钟)": f"{result['duration_seconds']/60:.1f}",
            })

df = pd.DataFrame(data)
print(df.to_string(index=False))
```

## 🛑 如果需要停止

### 停止训练

```bash
# 找到进程ID
ps aux | grep auto_research_practical.py

# 停止进程（优雅停止）
kill 2288615

# 如果不响应，强制停止
kill -9 2288615
```

**注意**：停止后，已完成的实验结果会保留，但当前正在运行的实验会丢失。

## 📊 预期结果

基于当前配置，预期的性能：

### 基线（bifurcation，200k步）
- 成功率：~60-75%
- 溶解率：~70-85%
- 收敛：~100k-150k步

### +课程学习
- 成功率：+5-10%
- 收敛速度：提升20-30%

### MCA场景（更难）
- 成功率：~50-65%（初期）
- 需要更多训练步数

### 课程学习+MCA
- 成功率：~65-80%
- 收敛：比直接训练MCA快30-40%

## 🔧 故障排查

### 进程已停止

检查日志：
```bash
tail -100 auto_research_practical.log
```

常见问题：
- CUDA OOM：减少 `n_envs`
- 训练失败：查看 `train_stderr.log`

### 结果看起来不对

检查：
1. 训练是否完成（查看 `final_policy.pt` 是否存在）
2. 日志文件大小（应该>10KB）
3. episode数量（应该>100个）

## 📚 后续工作

实验完成后：

1. **分析结果**：查看哪个创新点效果最好
2. **写论文**：使用生成的报告和图表
3. **继续迭代**：
   - 调整超参数
   - 增加训练步数
   - 尝试新的创新点
   - 组合最优的方法

## 💡 提示

- 自动科研系统会自动保存所有数据
- 可以随时中断（Ctrl+C或kill），已完成的实验会保留
- 日志文件会持续更新，可以实时查看
- 所有模型都可以用于观看和评估

---

**当前任务**：等待实验完成（约1-2小时）

**下一步**：查看报告并分析结果

祝科研顺利！🚀
