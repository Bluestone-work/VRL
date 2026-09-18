# 🔬 高级研究模块和优化方法

## 📦 新增研究模块

### 1. 高级GNN架构 (`marl/gnn_advanced.py`)

#### ✅ AdaptiveAdjacencyBuilder
- **功能**: 学习最优连接半径而非固定阈值
- **创新点**: 边权重通过MLP学习，根据agent状态动态调整
- **应用**: 自适应通信范围

#### ✅ EdgeFeatureGAT  
- **功能**: 带边特征的GAT层
- **创新点**: 将距离、相对速度等边特征纳入注意力计算
- **特征维度**: 8D (相对位置3D + 距离1D + 相对速度3D + 接近指标1D)

#### ✅ CommunicationBottleneck
- **功能**: 模拟带宽受限的通信
- **创新点**: 压缩观测到低维消息空间
- **应用**: 真实机器人通信约束

#### ✅ GraphMetricRewardShaper
- **功能**: 基于图度量的神经奖励塑形
- **指标**: 密度、平均度、直径、聚类系数、连通分量
- **应用**: 引导形成最优协作拓扑

#### ✅ HierarchicalGNN
- **功能**: 双层GNN（机器人级+集群级）
- **创新点**: 局部协调与全局策略分离推理
- **层级**: Robot-level (fine-grained) + Swarm-level (coarse)

### 2. Transformer架构 (`marl/transformer_policy.py`)

#### ✅ TransformerActor
- **优势**: 全对全注意力，无邻居限制
- **特性**: 
  - 位置编码处理空间结构
  - 多层Transformer编码器
  - 高斯策略头

#### ✅ TransformerCritic
- **集中式**: 观察所有agent + 全局状态
- **架构**: Transformer编码观测 + MLP编码动作

#### ✅ SparseAttentionMask
- **功能**: k-近邻稀疏注意力
- **优化**: O(N²) → O(Nk)，适合大规模集群
- **默认**: k=8 neighbors

#### ✅ CrossAttentionCritic
- **创新**: Agent-local stream + Global stream
- **交叉注意力**: 个体查询全局上下文
- **应用**: 更强的中心化价值估计

### 3. 渐进式训练框架 (`marl/progressive_training.py`)

#### ✅ AdaptiveCurriculum
- **动态难度**: 基于成功率、收敛速度、探索度
- **反停滞**: 检测性能平台期并强制增加难度
- **参数映射**: 难度 → (血栓数量, 距离, 流速)

#### ✅ PretrainingTasks
- **三阶段预训练**:
  1. **Navigation**: 无血栓，学习避障和跟随流
  2. **Single Clot**: 单血栓，学习接近和溶解
  3. **Coordination**: 多血栓，学习分流

#### ✅ PopulationBasedTraining (PBT)
- **种群规模**: 8个agent
- **Exploit**: 底部25%复制顶部25%权重
- **Explore**: 20%概率扰动超参数
- **超参数**: lr, gamma, entropy_coef, hidden_dim

#### ✅ SelfPlayTraining
- **对手池**: 保留最近10个历史版本
- **鲁棒性**: 避免过拟合到特定场景
- **检查点间隔**: 每50k步

#### ✅ ExplorationBonus
- **内在奖励**: 基于访问计数的探索奖励
- **公式**: bonus = coef / sqrt(count + 1)
- **离散化**: 状态四舍五入到0.01精度

#### ✅ AdaptiveBatchSizing
- **动态调整**: 根据loss方差自动调整batch size
- **范围**: 64 - 1024
- **策略**: 高方差→小batch，低方差→大batch

## 🧪 实验配置建议

### 实验1: 边特征GAT vs 标准GAT

```bash
# 标准GAT (baseline)
python scripts/train_gnn_mappo.py \
  --robots 5 --clots 3 --timesteps 500000 \
  --use-gat --exp-name "gat_standard"

# 边特征GAT
python scripts/train_advanced.py \
  --robots 5 --clots 3 --timesteps 500000 \
  --architecture edge_gat --exp-name "gat_edge_features"
```

**预期**: 边特征GAT在狭窄血管和高流速场景下性能更好

### 实验2: Transformer vs GAT

```bash
# Transformer (全对全注意力)
python scripts/train_advanced.py \
  --robots 8 --clots 5 --timesteps 500000 \
  --architecture transformer --exp-name "transformer"

# GAT (局部邻居)
python scripts/train_gnn_mappo.py \
  --robots 8 --clots 5 --timesteps 500000 \
  --use-gat --exp-name "gat"
```

**预期**: 
- Transformer: 更强的全局协调，但计算成本高
- GAT: 更高效，适合中小规模

### 实验3: 稀疏注意力 (可扩展性)

```bash
# 大规模集群
python scripts/train_advanced.py \
  --robots 20 --clots 10 --timesteps 1000000 \
  --architecture sparse_transformer \
  --k-neighbors 8 \
  --exp-name "sparse_attn_k8"
```

**预期**: O(Nk)复杂度，20+机器人仍可训练

### 实验4: 层次化GNN

```bash
python scripts/train_advanced.py \
  --robots 12 --clots 6 --timesteps 500000 \
  --architecture hierarchical_gnn \
  --exp-name "hierarchical"
```

**预期**: 更好的全局策略和局部执行

### 实验5: 自适应课程学习

```bash
# 固定课程 (baseline)
python scripts/train_gnn_mappo.py \
  --robots 5 --clots 3 --timesteps 500000 \
  --curriculum --exp-name "curriculum_fixed"

# 自适应课程
python scripts/train_advanced.py \
  --robots 5 --clots 3 --timesteps 500000 \
  --adaptive-curriculum \
  --exp-name "curriculum_adaptive"
```

**预期**: 自适应课程更快达到高成功率

### 实验6: 预训练效果

```bash
# 无预训练
python scripts/train_gnn_mappo.py \
  --robots 5 --clots 3 --timesteps 300000 \
  --exp-name "no_pretrain"

# 三阶段预训练
python scripts/train_advanced.py \
  --robots 5 --clots 3 --timesteps 300000 \
  --pretrain \
  --exp-name "with_pretrain"
```

**预期**: 预训练显著减少样本需求

### 实验7: PBT超参数优化

```bash
python scripts/train_pbt.py \
  --population-size 8 \
  --robots 5 --clots 3 --timesteps 500000 \
  --exp-name "pbt"
```

**预期**: 自动发现最优超参数组合

### 实验8: 探索奖励

```bash
# 无探索奖励
python scripts/train_gnn_mappo.py \
  --robots 5 --clots 3 --timesteps 500000 \
  --exp-name "no_exploration"

# 带探索奖励
python scripts/train_advanced.py \
  --robots 5 --clots 3 --timesteps 500000 \
  --exploration-bonus 0.01 \
  --exp-name "with_exploration"
```

**预期**: 探索奖励降低接触失败率

## 📊 评估指标

### 性能指标
- **成功率**: 所有血栓清除的比例
- **清除率**: 平均血栓清除百分比
- **接触失败率**: 从未接触血栓的episode比例
- **Episode回报**: 累积奖励

### 效率指标
- **样本效率**: 达到80%成功率所需步数
- **计算效率**: FPS (frames per second)
- **内存占用**: GPU显存使用

### 协作指标
- **分流度**: 机器人在血栓间的分布均匀度
- **图密度**: agent连接图的平均密度
- **碰撞率**: agent间碰撞频率

### 鲁棒性指标
- **泛化性能**: 在未见场景上的成功率
- **对抗鲁棒性**: 对随机扰动的稳定性

## 🎯 研究问题

### Q1: 边特征的重要性
**假设**: 显式建模相对速度和接近方向能提升协调质量  
**实验**: EdgeFeatureGAT vs 标准GAT  
**指标**: 碰撞率, 分流度

### Q2: 全对全 vs 局部注意力
**假设**: Transformer的全局视野在复杂分支场景中更优  
**实验**: Transformer vs GAT vs SparseTransformer  
**指标**: 成功率 (复杂血管), 计算效率

### Q3: 层次化推理的价值
**假设**: 分离局部和全局推理能提高决策质量  
**实验**: HierarchicalGNN vs FlatGNN  
**指标**: 全局策略质量 (如多血栓分配)

### Q4: 自适应课程 vs 固定课程
**假设**: 自适应调整难度比固定阈值更高效  
**实验**: AdaptiveCurriculum vs 固定阈值  
**指标**: 样本效率, 最终性能

### Q5: 预训练的迁移效果
**假设**: 基础技能预训练能加速复杂任务学习  
**实验**: 有/无预训练  
**指标**: 收敛速度, 初期接触率

### Q6: PBT的超参数发现
**假设**: PBT能自动发现优于手动调优的配置  
**实验**: PBT vs 网格搜索 vs 默认配置  
**指标**: 最优性能, 搜索效率

### Q7: 探索奖励的必要性
**假设**: 内在奖励能缓解稀疏奖励问题  
**实验**: 有/无探索奖励  
**指标**: 接触失败率, 探索覆盖度

## 🔬 消融研究矩阵

| 组件 | Baseline | Variant 1 | Variant 2 |
|-----|----------|-----------|-----------|
| **Attention** | GAT | Transformer | Sparse |
| **Edge** | No edge feat | Distance only | Full (8D) |
| **Hierarchy** | Flat | 2-level | 3-level |
| **Curriculum** | Fixed | Adaptive | Pre-train |
| **Communication** | Unlimited | Bottleneck-16 | Bottleneck-8 |
| **Graph Metrics** | No shaping | Static | Learned |

## 💡 未来方向

### 短期 (1-2周)
1. ✅ 实现所有模块 (已完成)
2. 🔄 编写高级训练脚本
3. 🔄 运行基础对比实验
4. 📊 可视化注意力权重

### 中期 (1-2月)
1. 大规模实验 (20+ agents)
2. 真实血管几何数据集
3. Sim-to-Real 迁移研究
4. 发表论文

### 长期 (3-6月)
1. 硬件实验验证
2. 与临床专家合作
3. 开源完整系统
4. 工业应用探索

## 📚 参考文献

### Transformer for MARL
- MAT: Wen et al., "Multi-Agent Transformer", NeurIPS 2021
- TarMAC: Das et al., "TarMAC: Targeted Multi-Agent Communication", ICML 2019

### 层次化多智能体
- HAMA: Yang et al., "Hierarchical Multi-Agent RL", AAMAS 2020
- QMIX: Rashid et al., "QMIX", ICML 2018

### 课程学习
- PAIRED: Dennis et al., "Emergent Complexity", NeurIPS 2020
- PLR: Jiang et al., "Prioritized Level Replay", ICLR 2021

### PBT
- Jaderberg et al., "Population Based Training", arxiv 2017

### 探索
- RND: Burda et al., "Random Network Distillation", ICLR 2019
- NGU: Badia et al., "Never Give Up", ICLR 2020
