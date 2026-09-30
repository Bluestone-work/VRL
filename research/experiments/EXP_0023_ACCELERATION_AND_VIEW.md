# EXP23 — 编译积分、并行采样与原版三维渲染恢复

日期：2026-09-29。用户要求优化速度并增加场景可视化，随后指出此前渲染更好。当前主线仍为修改后的 MCA 物理任务、纯 direct-local MAPPO。

## 已执行

- 新增 `environments/mca_compiled.py`：完整输运/壁面投影/开放出口/接触溶解/压力反馈循环由 Numba 编译执行，float64，关闭 fastmath；保持原步长限制、边界条件、物理输入。参考环境代码保留未改。
- 缓存确定性名义解剖与初始压力场；逐次 reset 仍使用原种子生成粒子。随机状态和初始化观测逐元素一致性测试通过；非名义几何仍走原 reset。
- 新增 `scripts/train_mca_parallel.py`：每种子8个环境，6个 CPU 工作线程，编译积分释放GIL；GPU批量推理与PPO。42/44使用cuda:0，43使用cuda:1。每轮仍128个环境样本、5个优化epoch、batch128。
- **采样结构有明确变化**：原1环境×128时间步改为8环境×16时间步，GAE轨迹长度/随机动作顺序因此改变。不宣称与串行训练逐位相同或最终权重相同；这是已登记的执行阶段变化。
- 三个旧进程均在下一次8192步检查点附近迁移，新进程首轮更新与checkpoint成功后才终止旧进程。已保存父checkpoint及hash、迁移时状态、原日志；恢复全部模型/优化器状态及原环境的未结束回合，新增7个环境使用互不重复的后续episode种子。
- 新训练目录：`research/runs/EXP_0023_MCA_FAST_20260929b/`。原目录的 `active_run.json` 指向当前运行，监控器和场景回放自动跟随。

## 速度与验证

- 原运行约1.8环境步/秒/种子；编译单环境实跑约29步/秒；编译并行预检1024步约98步/秒。
- 三种子正式续训同时运行、桌面渲染开启时，首次稳定观测约93–104步/秒/种子，约50倍提升；300万步剩余预算外推约8–9小时。以实时状态为准，非保证完成时刻。
- 全套测试 **342 passed，1 skipped**，其中13项编译后端专项测试。并行checkpoint在1024步保存后恢复至1152步，与连续1152步的模型参数和环境状态完全一致。
- 配对轨迹：3 seeds × 3分辨率（0.1/0.05/0.025），位置/质量/奖励/离场时间全部通过，边索引、活跃掩码、出口编号及回合结束类型一致。最大位置偏差0.000418073 mm，质量偏差0.00000599889，奖励偏差0.0000120353；已登记配对位置容差0.001 mm，比原0.05 mm数值门槛严格50倍。
- **不是逐位相同积分**：初次诊断用的1e-7 mm/2e-6质量等机器级一致性断言失败；定位到端点处浮点舍入引起中点/单侧积分分支和接触采样不同。保留attempt1–7记录；最终以明确登记的数值容差、离散事件一致及多分辨率配对检验验收，不改原物理精度和0.05 mm收敛门槛。
- 以上验证覆盖当前名义任务与测试场景，不能推断未见解剖或任意长时间溶解反馈已经验证。

证据：`research/validation/EXP0023_FAST_FULL_20260929.xml`、`EXP0023_BACKEND_AUDIT_20260929/`、`EXP0023_PARALLEL_20260929_continuous/resume_verified.json`、新运行根目录 `STARTUP_VERIFIED.json` / `migration_status.json`。

## 三维场景

用户记忆中的优质渲染入口是 `watch_gui.py`，使用 `environments/pv_render.py` 的 PyVista/VTK，而非最早的PyBullet简易管段。旧渲染文档记录了PyBullet TinyRenderer透明墙体遮挡内部机器人的问题。

- 已新增并运行 `scripts/view_mca_vtk.py`，复用原 PVRenderer 的连续可变半径血管、分层透明、高光材质与相机；新GUI加入柔和渐变背景、补光、自动环绕和治疗区域聚焦。
- 保留 `scripts/view_mca_pybullet.py` 作为PyBullet选项；按用户对画质的反馈，默认展示恢复为原VTK渲染。
- 展示当前seed42检查点的独立确定性回放，检查点更新后自动加载；轨迹来自实际积分子步，无直线插值或另一个Bullet动力学任务。训练与渲染分开进程/CPU核，渲染不影响训练状态。
- 物理0.15秒回合按80倍慢放；蓝色机器人绘制为实际半径4倍以便观察，白色粒子3倍；金色为血栓靶点标记，不是假造血栓表面。墙体显示初始阻塞半径，不假称逐帧重建了血栓组织表面。
- 鼠标拖拽旋转、滚轮缩放；空格暂停，F聚焦治疗区域，R全景，O切换自动环绕。
- GUI日志、当前截图和回放来源：原运行目录 `vtk_view/`；监控网页 `http://127.0.0.1:8763` 同时展示该场景截图和训练曲线。

## 续接

先读 `research/runs/EXP_0023_MCA_FAST_20260929b/seed_*/status.json`，不要重复启动。
需要停止新进程时发送SIGTERM或创建对应seed目录下的STOP文件；训练器会完成当前rollout/PPO更新、原子保存checkpoint，再进入paused状态。恢复前移除STOP文件，核实旧PID已结束，使用相同源码执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 /home/wj/miniconda3/envs/v/bin/python -m scripts.train_mca_parallel --seed 42 --device cuda:0 --n-envs 8 --workers 6 --out research/runs/EXP_0023_MCA_FAST_20260929b/seed_42 --resume
```

物理输入和原任务可达性限制没有改变：1秒时限不足以清除初始总质量4，不能把此次速度提升解释为任务成功率突破。
