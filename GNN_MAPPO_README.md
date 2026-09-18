# GNN-MAPPO for Vascular Thrombolysis

基于 [DGR_VDS](https://github.com/Bluestone-work/DGR_VDS) 项目的图神经网络（GNN）和多智能体近端策略优化（MAPPO）方法，应用于血管内微型机器人集群协同清除血栓的任务。

## 🎯 核心创新

### 1. 图注意力网络（GAT）建模智能体交互
- 将多个微型机器人建模为图节点
- 使用注意力机制捕获机器人之间的动态交互
- 基于空间邻近度构建邻接矩阵
- 置换不变的架构，可扩展到不同数量的智能体

### 2. MAPPO 算法
- **在线策略学习**：无需经验回放缓冲区，更简单更稳定
- **集中训练分散执行（CTDE）**：训练时使用全局信息，执行时仅用局部观测
- **PPO 裁剪目标**：稳定的策略梯度更新
- **广义优势估计（GAE）**：低方差的优势函数估计

### 3. 与参考项目的对应关系

| DGR_VDS 组件 | 本项目实现 | 说明 |
|-------------|----------|------|
| `gat_policy.py` | `marl/gat_policy.py` | 图注意力网络策略 |
| `mappo_mlp_model.py` | `marl/mappo_policy.py` | MAPPO 算法实现 |
| `train_gnn_mappo_full.py` | `scripts/train_gnn_mappo.py` | 训练脚本 |
| 环境状态图表示 | `build_adjacency_matrix()` | 基于位置的邻接矩阵构建 |

## 📁 文件结构

```
marl/
├── gat_policy.py          # GAT encoder, actor, critic
├── mappo_policy.py        # MAPPO 算法实现
└── maddpg_policy.py       # MADDPG 基线（原有）

scripts/
├── train_gnn_mappo.py     # GNN-MAPPO 训练脚本
├── run_comparison.py      # 对比实验脚本
└── train_vascular_maddpg.py  # MADDPG 基线训练

environments/
└── vascular_3d_marl_env.py  # 血管导航环境
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy gymnasium matplotlib
```

### 训练 GNN-MAPPO

```bash
# 基础训练（3个机器人，3个血栓）
python scripts/train_gnn_mappo.py \
  --robots 3 \
  --clots 3 \
  --timesteps 500000 \
  --use-gat \
  --curriculum \
  --device cuda

# 更多机器人（挑战性更高）
python scripts/train_gnn_mappo.py \
  --robots 8 \
  --clots 5 \
  --timesteps 1000000 \
  --hidden-dim 256 \
  --num-gat-layers 3 \
  --device cuda
```

### 运行对比实验

比较 GNN-MAPPO、MLP-MAPPO（消融）和 MADDPG（基线）：

```bash
python scripts/run_comparison.py \
  --robots 3 \
  --clots 3 \
  --timesteps 300000 \
  --seeds 3 \
  --device cuda
```

这会生成对比图表和统计表格。

### 可视化训练好的策略

```bash
# 使用 PyBullet GUI 观看
python watch_gui.py \
  --policy logdir/gnn_mappo/seed_42/best_policy.pt \
  --episodes 5

# 导出 GIF
python make_gif.py \
  --policy logdir/gnn_mappo/seed_42/best_policy.pt \
  --out gnn_mappo_demo.gif
```

## 🧠 算法详解

### GNN-MAPPO 架构

```
观测 (每个机器人)
    ↓
[GAT Encoder]  ← 多层图注意力
    ├─ 自注意力机制
    ├─ 多头注意力
    └─ 邻接矩阵（基于距离）
    ↓
图感知特征
    ↓
┌─────────────┬─────────────┐
│             │             │
Actor         Critic (集中式)
(分散式)       ├─ 观测编码 (GAT)
│             ├─ 动作编码
高斯策略       ├─ 全局状态
│             └─ 价值估计
动作采样
```

### 关键特性

1. **动态图构建**
   ```python
   # 基于机器人位置自动构建邻接矩阵
   adj_matrix = build_adjacency_matrix(
       positions,
       threshold=0.15,  # 15cm 邻近阈值
       include_self=True
   )
   ```

2. **多头注意力**
   - 捕获不同类型的交互模式
   - 更丰富的特征表示
   - 默认 4 个注意力头

3. **集中式训练**
   - Critic 观察所有机器人的状态和动作
   - 包含全局血栓信息
   - 训练时稳定，执行时高效

4. **课程学习**
   ```bash
   --curriculum  # 从 1 个血栓开始，逐步增加到目标数量
   ```
   - 成功率超过 60% 时增加难度
   - 避免早期探索困难
   - 提高样本效率

## 📊 实验结果

### 预期性能对比

基于类似任务的经验（需要在你的环境中验证）：

| 方法 | 成功率 | 平均回报 | 血栓清除率 | 训练稳定性 |
|-----|-------|---------|----------|-----------|
| **GNN-MAPPO** | ~85-95% | ~95 | ~97% | ⭐⭐⭐⭐⭐ |
| MLP-MAPPO | ~70-80% | ~75 | ~85% | ⭐⭐⭐⭐ |
| MADDPG | ~60-70% | ~50 | ~75% | ⭐⭐⭐ |

### GNN 的优势

1. **更好的协调**：显式建模机器人间交互
2. **可扩展性**：相同网络可用于不同数量的机器人
3. **泛化能力**：学到的是关系而非固定位置

## ⚙️ 超参数调优

### 推荐配置（3 机器人，3 血栓）

```bash
--hidden-dim 128          # 隐藏层维度
--num-gat-layers 2        # GAT 层数
--num-heads 4             # 注意力头数
--lr-actor 3e-4          # Actor 学习率
--lr-critic 1e-3         # Critic 学习率
--gamma 0.99             # 折扣因子
--gae-lambda 0.95        # GAE lambda
--clip-epsilon 0.2       # PPO 裁剪参数
--entropy-coef 0.01      # 熵正则化系数
--n-steps 2048           # 每次更新的步数
--n-epochs 10            # PPO 更新轮数
--batch-size 256         # 小批量大小
```

### 大规模场景（8+ 机器人）

```bash
--hidden-dim 256          # 增加网络容量
--num-gat-layers 3        # 更深的图结构
--num-heads 8             # 更多注意力头
--n-steps 4096           # 更大的批量
```

## 🔬 消融研究

### 测试 GAT 的贡献

```bash
# 有 GAT
python scripts/train_gnn_mappo.py --use-gat

# 无 GAT（纯 MLP）
python scripts/train_gnn_mappo.py --no-gat
```

### 测试课程学习的影响

```bash
# 有课程学习
python scripts/train_gnn_mappo.py --curriculum

# 无课程学习（直接用全部血栓）
python scripts/train_gnn_mappo.py
```

## 🐛 调试技巧

### 1. 检查图连接

```python
from marl.gat_policy import build_adjacency_matrix
import torch

# 模拟机器人位置
positions = torch.randn(1, 3, 3)  # [batch, n_agents, 3]
adj = build_adjacency_matrix(positions, threshold=0.15)

print("Adjacency matrix:")
print(adj[0])  # 应该看到对角线和近邻有 1
```

### 2. 监控训练

```bash
# 实时查看日志
tail -f logdir/gnn_mappo/seed_42/log.txt

# 分析 episode metrics
python -c "
import json
with open('logdir/gnn_mappo/seed_42/episode_metrics.json') as f:
    data = json.load(f)
    successes = [e['success'] for e in data[-100:]]
    print(f'Recent 100 episodes success rate: {sum(successes)/len(successes):.2%}')
"
```

### 3. 可视化注意力权重

```python
# 在 gat_policy.py 的 GATLayer.forward 中添加：
# self.last_attn_weights = attn_weights.detach()

# 然后可视化
import matplotlib.pyplot as plt
attn = model.gat_encoder.gat_layers[0].last_attn_weights[0, 0]  # [n_agents, n_agents]
plt.imshow(attn.cpu().numpy())
plt.colorbar()
plt.title("Attention Weights (Head 0)")
plt.show()
```

## 📈 与 DGR_VDS 的对比

### 相同点
- ✅ 使用 GAT 编码智能体交互
- ✅ MAPPO 作为训练算法
- ✅ CTDE 范式（集中训练分散执行）
- ✅ 多层图神经网络

### 差异点
- 🔄 **任务**：DGR_VDS 是多机器人路径规划，本项目是血管内血栓清除
- 🔄 **观测空间**：本项目包含血管几何、血流、血栓信息
- 🔄 **奖励设计**：溶栓接触奖励、测地距离塑形、分流惩罚
- 🔄 **物理模型**：低雷诺数流体、接触溶解动力学、血管几何约束

### 借鉴的关键思想
1. **图结构建模**：将智能体关系显式建模为图
2. **注意力机制**：自动学习哪些邻居重要
3. **置换不变性**：网络架构对智能体顺序不敏感
4. **分层编码**：先局部特征，再图传播，最后全局聚合

## 🚧 已知限制和未来工作

### 当前限制
- 邻接矩阵基于固定阈值（可以改为动态或学习的）
- GAT 层数和注意力头数需要手动调优
- 还没有实现异构智能体（所有机器人相同）

### 未来改进方向
1. **动态图结构**
   ```python
   # 学习阈值而非固定
   self.threshold = nn.Parameter(torch.tensor(0.15))
   ```

2. **边特征**
   ```python
   # 在 GATLayer 中加入边特征（距离、相对速度）
   edge_features = compute_edge_features(positions, velocities)
   ```

3. **分层图**
   ```python
   # 机器人级别 + 血栓级别的二部图
   robot_graph = build_robot_graph(robot_positions)
   clot_graph = build_clot_graph(clot_positions)
   bipartite = build_bipartite_graph(robots, clots)
   ```

4. **Transformer 替代 GAT**
   ```python
   # 使用完整的 Transformer encoder
   from torch.nn import TransformerEncoder
   ```

## 📚 参考文献

1. **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2021
2. **GAT**: Veličković et al., "Graph Attention Networks", ICLR 2018
3. **MADDPG**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017
4. **DGR_VDS**: Bluestone-work, https://github.com/Bluestone-work/DGR_VDS

## 💡 使用建议

### 开始实验
1. 先用小规模（3 机器人，1-2 血栓）验证代码
2. 启用课程学习，让训练更稳定
3. 从预训练的 MADDPG 策略开始也是个好主意

### 调试技巧
- 如果成功率很低，检查邻接矩阵是否合理
- 如果训练不稳定，减小学习率或增大 batch size
- 如果策略退化，检查 entropy coefficient 是否太小

### 性能优化
- 使用向量化环境（见 `environments/vector_env.py`）
- GPU 加速训练（`--device cuda`）
- 增大并行环境数量可以提高吞吐量

## 🤝 致谢

本实现参考了 [DGR_VDS](https://github.com/Bluestone-work/DGR_VDS) 项目的 GNN-MAPPO 方法，并将其应用于血管内微型机器人协同控制任务。感谢原作者的开源贡献！

---

**问题反馈**: 如果遇到问题或有改进建议，欢迎提 issue 或 PR！
