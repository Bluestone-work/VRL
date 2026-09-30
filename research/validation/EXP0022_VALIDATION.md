# EXP_0022 验证交接记录 — 2026-09-28

当前阶段：**显式单位与流场工程验证完成，RL集成/成功率验证尚未完成**。

## 运行记录

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0-5,8-23 /home/wj/miniconda3/envs/v/bin/python -m pytest tests -q \
  --junitxml=research/validation/EXP0022_FULL_TESTS_20260928.xml
```

结果：286 passed、1 skipped，10.56秒；新MCA测试42项。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-5 \
  /home/wj/miniconda3/envs/v/bin/python -m scripts.validate_mca_physiology \
  --out research/validation/EXP0022_MCA_20260928_final
```

结果：exit 0；1,296流场、11,664可达性参数组合、3,888数值组合。
所有工程检查与资料hash检查通过；`training_ready=false`，`success_rate=null`。

最终目录包含配置快照、CSV全量结果、JSON摘要、中文报告、`COMPLETE.json`文件校验和。
资料快照保存于`research/runs/EXP0022_REFERENCES_20260928/`，不能把其中群体训练速度
写成当前机器人在血液中的实测推进速度。

## 后续继续的位置

- 读取`research/experiments/EXP_0022_MCA_PHYSIOLOGY.md`及最终报告。
- `environments/mca_physiology.py`已可独立调用，但旧单/向量RL环境尚未接入。
- 新环境需处理速度积分子步、出口离场及粒子同流场；全局子步上界高，先做局部自适应和收敛测试。
- 缺乏实测机器人/溶解/出口阻抗数据时，只能做明确标注的工程敏感性，不能宣称生理标定完成。
- 当前控制还是独立机器人速度的抽象执行器；共享磁场硬件约束未实现。
- 既有训练、checkpoint和成功定义保持原协议；本轮无正式训练任务启动。
