"""演示改进后的血管环境和螺旋机器人集群可视化。

使用方法：
    # MCA 中风场景（解剖学真实）
    python demo_improved_viz.py --scenario mca_stroke --episodes 2

    # 多层分支树（4代分支）
    python demo_improved_viz.py --scenario multilevel --episodes 2

    # 传统简单场景对比
    python demo_improved_viz.py --scenario bifurcation --episodes 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_marl_env import Vascular3DMARLEnv


def main() -> None:
    ap = argparse.ArgumentParser(description="改进的血管环境演示")
    ap.add_argument(
        "--scenario",
        default="mca_stroke",
        choices=["straight", "bifurcation", "anastomosis", "stenotic",
                 "multilevel", "mca_stroke"],
        help="血管场景：multilevel=多代分支树, mca_stroke=MCA中风解剖"
    )
    ap.add_argument("--robots", type=int, default=3, help="机器人数量")
    ap.add_argument("--clots", type=int, default=3, help="血栓数量")
    ap.add_argument("--horizon", type=int, default=300, help="最大步数")
    ap.add_argument("--episodes", type=int, default=3, help="演示回合数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=30.0, help="帧率")
    args = ap.parse_args()

    print("=" * 70)
    print(f"改进的血管环境演示")
    print("=" * 70)
    print(f"场景: {args.scenario}")
    print(f"机器人: {args.robots} 个（每个渲染为螺旋集群）")
    print(f"血栓: {args.clots} 个")
    print("-" * 70)

    if args.scenario == "multilevel":
        print("📐 多层分支血管树：")
        print("   - 3代分支（8个末端分支）")
        print("   - Murray定律半径递减")
        print("   - 扭曲度随机（模拟真实血管弯曲）")
    elif args.scenario == "mca_stroke":
        print("🧠 MCA中风解剖场景：")
        print("   - ICA虹吸段（S形弯曲，最难导航）")
        print("   - M1水平段（典型大血管闭塞位置）")
        print("   - M2上/下分支（策略必须选择分支）")
        print("   - 豆状核穿支小分支")

    print("-" * 70)
    print("🔬 微型机器人：螺旋丝状集群（人工细菌鞭毛，ABF）")
    print("   - 每个智能体 = 3根螺旋丝")
    print("   - 沿运动方向旋转（旋转-平移耦合）")
    print("   - 集群布局：玫瑰花瓣式分布")
    print("=" * 70)
    print()

    env = Vascular3DMARLEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        render_mode="human",
        randomize_scenario=False,  # 固定拓扑便于观察
        randomize_clots=True,
        use_pybullet=True,
        obs_mode="geometric",       # 使用增强观测
    )

    frame_budget = 1.0 / max(args.fps, 1e-6)

    for ep in range(1, args.episodes + 1):
        print(f"\n[Episode {ep}/{args.episodes}]")
        obs, info = env.reset(seed=args.seed + ep)
        env.render()

        ret, steps = 0.0, 0
        contact_steps = 0

        while True:
            t0 = time.time()
            # 随机动作演示（实际应用中会加载训练好的策略）
            action = env.action_space.sample() * 0.5  # 降低速度便于观察

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            ret += reward
            steps += 1
            if info["active_contacts"] > 0:
                contact_steps += 1

            if terminated or truncated:
                break

            sleep = frame_budget - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

        print(f"  回报: {ret:+8.2f}")
        print(f"  步数: {steps}/{args.horizon}")
        print(f"  血栓溶解率: {info['removal_rate']:.1%}")
        print(f"  成功: {'✓' if info['success'] else '✗'}")
        print(f"  接触步数: {contact_steps} ({100*contact_steps/steps:.1f}%)")
        print(f"  碰撞: 墙壁={info['wall_collisions']}, 机器人间={info['robot_collisions']}")

    env.close()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("\n关键改进：")
    print("  1. ✅ 血管结构：从3段简单管道 → 多代分支/解剖学真实")
    print("  2. ✅ 机器人模型：从单个球体 → 螺旋丝状集群（ABF）")
    print("  3. ✅ 控制方式：保持不变（每个智能体独立3D速度命令）")
    print("  4. ✅ 物理仿真：连续性方程、Murray定律、测地路由")
    print("=" * 70)


if __name__ == "__main__":
    main()
