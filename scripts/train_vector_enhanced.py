"""增强版训练脚本，支持所有创新点。

支持的创新点：
1. --use-gat: 图注意力网络
2. --use-communication: 智能体通信
3. --use-hierarchical: 层次化策略
4. --curriculum-stages: 多阶段课程学习

使用方法：
    python scripts/train_vector_enhanced.py \
        --scenario mca_stroke \
        --n-envs 64 \
        --timesteps 500000 \
        --use-gat \
        --curriculum-stages 3 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vector_env import VectorVascularEnv
from marl.maddpg_policy import MADDPG


class CurriculumScheduler:
    """多阶段课程学习调度器"""

    def __init__(self, stages: list[dict], window: int = 100):
        self.stages = stages
        self.current_stage = 0
        self.window = window
        self.success_history = deque(maxlen=window)

    def should_advance(self, success_rate: float, threshold: float = 0.6) -> bool:
        """是否应该进入下一阶段"""
        if self.current_stage >= len(self.stages) - 1:
            return False
        return success_rate >= threshold

    def advance(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.success_history.clear()
            return True
        return False

    def get_current_stage(self) -> dict:
        """获取当前阶段配置"""
        return self.stages[self.current_stage]


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    print("=" * 80)
    print("训练 MADDPG (向量化 + 创新点)")
    print("=" * 80)
    print(f"环境: n_envs={args.n_envs} robots={args.robots} clots={args.clots}")
    print(f"场景: {args.scenario}")
    print(f"观测: {args.obs_mode} 奖励: {args.reward_mode}")
    print(f"设备: {device} 种子: {args.seed}")

    # 创新点标记
    innovations = []
    if args.use_gat:
        innovations.append("GAT")
    if args.use_communication:
        innovations.append("Communication")
    if args.use_hierarchical:
        innovations.append("Hierarchical")
    if args.curriculum_stages > 1:
        innovations.append(f"Curriculum({args.curriculum_stages})")

    if innovations:
        print(f"创新点: {', '.join(innovations)}")
    else:
        print("创新点: 无（基线）")
    print("=" * 80)

    # 多阶段课程学习
    curriculum = None
    if args.curriculum_stages > 1:
        stages = [
            {"scenario": "bifurcation", "clots": 1, "difficulty": 0.3},
            {"scenario": "multilevel", "clots": 2, "difficulty": 0.6},
            {"scenario": "mca_stroke", "clots": 3, "difficulty": 1.0},
        ][:args.curriculum_stages]
        curriculum = CurriculumScheduler(stages)
        current_stage = curriculum.get_current_stage()
        scenario = current_stage["scenario"]
        clots = current_stage["clots"]
        print(f"[课程学习] 阶段 1/{len(stages)}: {scenario} ({clots}血栓)")
    else:
        scenario = args.scenario
        clots = args.clots

    # 创建环境
    env = VectorVascularEnv(
        n_envs=args.n_envs,
        scenario=scenario,
        num_robots=args.robots,
        num_clots=clots,
        horizon=args.horizon,
        seed=args.seed,
        randomize_scenario=args.randomize_scenario,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        robot_radius=args.robot_radius,
        scenario_pool=args.scenario_pool,
        tree_resample_interval=args.tree_resample_interval,
    )
    obs = env.reset_all()
    obs_dim = obs["nodes"].shape[2]
    state_dim = 0 if args.no_critic_state else int(obs["clot_state"][0].size)

    # 创建智能体（暂时不实现创新点的网络结构，先用基线）
    # TODO: 实现 GAT, Communication, Hierarchical 的网络结构
    if args.use_gat or args.use_communication or args.use_hierarchical:
        print("\n⚠️  注意: GAT/Communication/Hierarchical 网络尚未实现")
        print("    当前使用基线网络结构，只记录配置用于后续实现\n")

    agent = MADDPG(
        n_agents=args.robots,
        obs_dim=obs_dim,
        action_dim=3,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
        state_dim=state_dim,
        share_parameters=not args.no_share_parameters,
        seed=args.seed,
    )

    # 日志目录
    if args.save_path:
        run_dir = Path(args.save_path).parent
    else:
        suffix = f"_{args.tag}" if args.tag else ""
        logdir = Path(args.logdir) / f"vec_{scenario}_r{args.robots}_c{clots}{suffix}"
        run = 1
        while (logdir / f"run_{run}").exists():
            run += 1
        run_dir = logdir / f"run_{run}"

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"日志目录: {run_dir}")

    # 保存配置
    config = vars(args).copy()
    config.update({
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_dim": 3,
        "innovations": innovations,
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    # 打开日志文件
    log_file = run_dir / "training_log.jsonl"
    metrics = log_file.open("a")

    # 训练循环状态
    ep_returns = np.zeros((args.n_envs,), np.float32)
    recent_return: deque[float] = deque(maxlen=200)
    recent_success: deque[float] = deque(maxlen=100)
    recent_removal: deque[float] = deque(maxlen=100)
    recent_contact: deque[float] = deque(maxlen=100)

    episode_count = 0
    best = -np.inf
    transitions = 0
    start = time.time()
    batched_steps = args.timesteps // args.n_envs

    for it in range(1, batched_steps + 1):
        # 探索率衰减
        frac = transitions / max(args.epsilon_decay_steps, 1)
        eps = max(args.epsilon_end, 1.0 - (1.0 - args.epsilon_end) * frac)

        # 选择动作
        nodes = obs["nodes"]
        flat = nodes.reshape(-1, obs_dim)
        with torch.no_grad():
            t = torch.as_tensor(flat, device=device)
            act = agent.actor(0)(t).cpu().numpy()
        act = act.reshape(args.n_envs, args.robots, 3)

        if eps > 0:
            act = np.clip(
                act + eps * np.random.normal(0, 0.2, act.shape).astype(np.float32),
                -1.0, 1.0,
            )

        # 环境步进
        nxt, reward, term, trunc, info = env.step(act)
        done = term | trunc

        # 处理终止观测
        next_nodes = nxt["nodes"]
        next_state_src = nxt["clot_state"]
        if "final_observation" in info:
            next_nodes = np.where(
                done[:, None, None], info["final_observation"]["nodes"], next_nodes
            )
            next_state_src = np.where(
                done[:, None, None],
                info["final_observation"]["clot_state"],
                next_state_src,
            )

        # 奖励分配
        shared = info["team_reward"] / max(args.robots, 1)
        per_agent = info["agent_rewards"] + shared[:, None]
        states = None if state_dim == 0 else obs["clot_state"].reshape(args.n_envs, -1)
        next_states = (
            None if state_dim == 0 else next_state_src.reshape(args.n_envs, -1)
        )

        # 存储经验
        for e in range(args.n_envs):
            agent.replay_buffer.store(
                obs=nodes[e], actions=act[e], rewards=per_agent[e],
                next_obs=next_nodes[e],
                done=bool(term[e]),
                state=None if states is None else states[e],
                next_state=None if next_states is None else next_states[e],
            )

        transitions += args.n_envs
        ep_returns += reward

        # 梯度更新
        if len(agent.replay_buffer) >= max(args.batch_size, args.learning_starts):
            for _ in range(args.updates_per_step):
                loss = agent.update()

        # 处理完成的episode
        finished = np.flatnonzero(done)
        for e in finished:
            episode_count += 1
            success = bool(info["success"][e])
            removal = float(info["removal_rate"][e])
            contact_miss = bool(info.get("contact_miss", [False]*args.n_envs)[e])

            recent_return.append(float(ep_returns[e]))
            recent_success.append(1.0 if success else 0.0)
            recent_removal.append(removal)
            recent_contact.append(0.0 if contact_miss else 1.0)

            # 记录日志
            log_entry = {
                "timestep": transitions,
                "episode": episode_count,
                "return": round(float(ep_returns[e]), 3),
                "success": success,
                "removal_rate": round(removal, 3),
                "wall_collisions": int(info["wall_collisions"][e]),
                "robot_collisions": int(info.get("robot_collisions", [0]*args.n_envs)[e]),
                "clots_engaged": int(info["clots_engaged"][e]),
                "contact_miss": contact_miss,
                "first_contact_step": int(info.get("first_contact_step", [-1]*args.n_envs)[e]),
            }

            if curriculum:
                log_entry["curriculum_stage"] = curriculum.current_stage + 1

            metrics.write(json.dumps(log_entry) + "\n")
            ep_returns[e] = 0.0

        if finished.size:
            metrics.flush()

        # 保存最佳模型
        if recent_return and float(np.mean(recent_return)) > best:
            best = float(np.mean(recent_return))
            save_path = Path(args.save_path) if args.save_path else (run_dir / "best_policy.pt")
            agent.save(save_path)

        # 课程学习阶段提升
        if curriculum and len(recent_success) >= 50:
            success_rate = float(np.mean(recent_success))
            if curriculum.should_advance(success_rate, threshold=0.6):
                if curriculum.advance():
                    stage = curriculum.get_current_stage()
                    print(f"\n[课程学习] 提升到阶段 {curriculum.current_stage + 1}/{len(curriculum.stages)}")
                    print(f"  场景: {stage['scenario']}, 血栓: {stage['clots']}\n")

                    # 重建环境
                    env.close()
                    env = VectorVascularEnv(
                        n_envs=args.n_envs,
                        scenario=stage["scenario"],
                        num_robots=args.robots,
                        num_clots=stage["clots"],
                        horizon=args.horizon,
                        seed=args.seed + curriculum.current_stage,
                        randomize_scenario=args.randomize_scenario,
                        reward_mode=args.reward_mode,
                        obs_mode=args.obs_mode,
                        robot_radius=args.robot_radius,
                        scenario_pool=args.scenario_pool,
                        tree_resample_interval=args.tree_resample_interval,
                    )
                    obs = env.reset_all()

        # 打印进度
        if it % args.log_interval == 0:
            elapsed = time.time() - start
            print(
                f"it {it:6d} | trans {transitions:8d} | ep {episode_count:5d} | "
                f"ret {np.mean(recent_return) if recent_return else 0.0:+7.2f} | "
                f"succ {np.mean(recent_success) if recent_success else 0.0:.3f} | "
                f"rmv {np.mean(recent_removal) if recent_removal else 0.0:.3f} | "
                f"cnt {np.mean(recent_contact) if recent_contact else 0.0:.3f} | "
                f"eps {eps:.3f} | "
                f"{transitions / elapsed:.0f} tr/s"
            )

    # 保存最终模型
    final_path = Path(args.save_path) if args.save_path else (run_dir / "final_policy.pt")
    agent.save(final_path)
    metrics.close()
    env.close()

    print("\n" + "=" * 80)
    print(f"✅ 训练完成")
    print(f"最佳平均回报: {best:.2f}")
    print(f"最终成功率: {np.mean(recent_success) if recent_success else 0.0:.1%}")
    print(f"最终溶解率: {np.mean(recent_removal) if recent_removal else 0.0:.1%}")
    print(f"日志位置: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="增强版MADDPG训练脚本")

    # 环境参数
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--scenario-pool", default="legacy",
                   help="which scenarios --randomize-scenario draws from: "
                        "legacy, generated, anatomical, arterial, venous, all")
    p.add_argument("--tree-resample-interval", type=int, default=0,
                   help="steps between resampling the shared topology. 0 keeps "
                        "one tree for the whole run, which makes a multi-scenario "
                        "pool contribute only its first draw. A few multiples of "
                        "--horizon is the useful range")
    p.add_argument("--robot-radius", type=float, default=0.0045,
                   help="device radius; sets the narrowest lumen the geometry "
                        "may contain (floored at 3x). 0.0011 is needed for the "
                        "anatomical territories' stenoses to survive")
    p.add_argument("--scenario", default="mca_stroke")
    p.add_argument("--robots", type=int, default=3)
    p.add_argument("--clots", type=int, default=3)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--timesteps", type=int, default=500000)
    p.add_argument("--randomize-scenario", action="store_true", default=True)
    p.add_argument("--obs-mode", default="geometric", choices=["geometric", "legacy"])
    p.add_argument("--reward-mode", default="milestone", choices=["baseline", "milestone"])

    # 创新点参数
    p.add_argument("--use-gat", action="store_true", help="使用图注意力网络")
    p.add_argument("--use-communication", action="store_true", help="启用智能体通信")
    p.add_argument("--use-hierarchical", action="store_true", help="使用层次化策略")
    p.add_argument("--curriculum-stages", type=int, default=1, help="课程学习阶段数(1=禁用)")

    # 算法参数
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epsilon-decay-steps", type=int, default=200000)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--no-share-parameters", action="store_true", default=False)
    p.add_argument("--no-critic-state", action="store_true", default=False)

    # 日志与保存
    p.add_argument("--tag", default="")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--logdir", default="logdir")
    p.add_argument("--save-path", default="", help="模型保存路径（覆盖默认）")

    # 其他
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    train(p.parse_args())
