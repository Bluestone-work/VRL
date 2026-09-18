"""Render an episode to a GIF or MP4 you can actually read.

    python make_gif.py --policy checkpoints/run_2_best_policy.pt --out out.mp4
    python make_gif.py --scenario mca_stroke --out random.gif
    python make_gif.py --views triple --out three_views.mp4

The old version drove PyBullet's TinyRenderer, which does not composite
transparency: the translucent vessel wall wrote depth without blending, so the
swarm inside the lumen was hidden. A measured frame had 19,600 wall pixels, 13
robot pixels and zero pixels of the green "lysing" state. This one renders with
VTK (depth peeling, real order-independent transparency) and adds the HUD that
makes navigation quality visible rather than merely present.

Each frame carries:

  * the lumen as a translucent swept surface, tapering by Murray's law, with
    the centerline drawn faintly as the navigational reference,
  * one legible sphere per robot, coloured blue in transit, green while lysing
    a clot, amber on a wall hit this step,
  * clots sized by remaining mass, so lysis is visible as shrinkage,
  * a cyan geodesic from each robot to its assigned clot, following the vessel
    rather than cutting through walls -- a robot entering the wrong branch
    shows a line that doubles back through the junction,
  * a HUD: per-agent geodesic distance to target (the panel that answers "is
    it navigating?"), removal rate, and per-clot remaining mass.

`--views` renders several cameras into one frame. The top view is the one that
tells you which branch the swarm is in: these trees branch mostly within a
plane, so an oblique view stacks the branches along the line of sight while a
top view separates them.

    single        one fitted oblique (default)
    top_oblique   top + fitted oblique, stacked
    triple        full-width top, with front and side below
    quad          top, front, side, oblique in a 2x2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import ALL_SCENARIOS


def robot_states(env, contacting: np.ndarray, wall_hits: np.ndarray) -> list[str]:
    """Per-robot colour state. Wall contact wins, since it is the failure."""
    out = []
    for i in range(env.num_robots):
        if i < len(wall_hits) and wall_hits[i]:
            out.append("wall")
        elif contacting[i]:
            out.append("contact")
        else:
            out.append("transit")
    return out


def contact_mask(env) -> np.ndarray:
    """Which robots are within lysis range of a live clot.

    Uses the same radius the lysis model uses, so the colour in the frame
    cannot disagree with what the reward paid for.
    """
    n = env.num_robots
    if env.active_clots <= 0:
        return np.zeros((n,), dtype=bool)
    alive = env.clot_masses[: env.active_clots] > 0
    if not np.any(alive):
        return np.zeros((n,), dtype=bool)
    d = np.linalg.norm(
        env.robot_positions[:, None, :]
        - env.clot_positions[: env.active_clots][None, :, :], axis=2,
    )
    d[:, ~alive] = np.inf
    return d.min(axis=1) <= env.clot_contact_radius


def geodesic_routes(env, assigned: np.ndarray) -> list[np.ndarray]:
    """Vessel-following polyline from each robot to its assigned clot.

    Walks the `next_hop` field from `route_to`, which is a Dijkstra
    predecessor tree on the station graph, so the path is a true geodesic
    through the vessel -- including taking the correct side of a bifurcation.
    The robot's own position and the clot's are stitched onto the ends so the
    line starts and finishes on the actual bodies rather than on the nearest
    station.
    """
    routes: list[np.ndarray] = []
    if env.active_clots <= 0:
        return routes
    for i in range(env.num_robots):
        ci = int(assigned[i])
        if ci < 0 or ci >= env.active_clots or env.clot_masses[ci] <= 0:
            continue
        target = int(env.clot_stations[ci])
        _dist, next_hop = env._route(ci)
        s = int(env.robot_stations[i])
        chain = [s]
        # Bounded walk: a malformed predecessor field would otherwise spin.
        for _ in range(env.tree.n_stations + 2):
            if s == target:
                break
            nxt = int(next_hop[s])
            if nxt == s:
                break
            s = nxt
            chain.append(s)
        pts = [env.robot_positions[i]]
        pts.extend(env.tree.points[chain])
        pts.append(env.clot_positions[ci])
        routes.append(np.asarray(pts, dtype=np.float64))
    return routes


def header_text(env, step: int, info: dict, ret: float, reward: float,
                assigned: np.ndarray, states: list[str], geo: np.ndarray) -> str:
    """The monospace block at the top of the HUD."""
    lines = [
        f"{env.active_scenario}   step {step:3d}/{env.horizon}",
        f"reward   {reward:+8.3f}  return {ret:+8.2f}",
        f"team {info.get('team_reward', 0.0):+7.3f}  agent "
        f"{np.mean(info.get('agent_rewards', 0.0)):+7.3f}",
        f"lysis {info.get('removed_mass', 0.0):+.4f}  remaining "
        f"{info.get('remaining_mass', 0.0):.4f}",
        f"removal  {info.get('removal_rate', 0.0) * 100:5.1f} %"
        f"   clots left {info.get('active_clots', 0)}",
        f"engaged  {info.get('clots_engaged', 0)}"
        f"   contacts {info.get('active_contacts', 0)}",
        f"wall hits {info.get('wall_collisions', 0)}"
        f"   peer {info.get('robot_collisions', 0)}",
        "",
        "agent target   state    speed    geo",
    ]
    label = {"contact": "LYSING", "wall": "wall!", "transit": "transit"}
    for i in range(env.num_robots):
        ci = int(assigned[i]) if i < len(assigned) else -1
        tgt = f"clot {ci}" if ci >= 0 else "--"
        speed = float(np.linalg.norm(env.robot_velocities[i]) / max(env.max_speed, 1e-8))
        distance = float(geo[i]) if i < len(geo) and np.isfinite(geo[i]) else 0.0
        lines.append(f"  {i}    {tgt:7s}  {label[states[i]]:6s} "
                     f"{speed:6.3f}  {distance:6.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--controller", default="zero",
                    choices=["zero", "guided", "flow_guided", "flow_spread"],
                    help="use a deterministic controller when --policy is omitted")
    ap.add_argument("--out", default="episode.gif",
                    help=".gif or .mp4 (mp4 keeps full colour; GIF is 256)")
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--clots", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--robot-radius", type=float, default=0.0045,
                    help="must match training; use 0.0011 for anatomical runs")
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--wall-opacity", type=float, default=0.16,
                    help="0.15 to see the swarm clearly, 0.4 for more tissue")
    ap.add_argument("--no-routes", action="store_true",
                    help="hide the robot-to-clot geodesics")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--views", default="single",
                    help="camera set: single, top_oblique, triple, quad. "
                         "The top view is the one that shows which branch the "
                         "swarm is in; oblique views stack the branches along "
                         "the line of sight")
    ap.add_argument("--layout", default="bottom", choices=["bottom", "right"],
                    help="bottom gives the 3D view the full width; these vessel "
                         "trees are flat, so that is usually what you want")
    ap.add_argument("--spin", type=float, default=0.0,
                    help="camera degrees per frame; 0 holds still")
    ap.add_argument("--scenario", default="mca_stroke", choices=list(ALL_SCENARIOS))
    ap.add_argument("--reward-mode", default="baseline",
                    choices=["baseline", "milestone"])
    ap.add_argument("--obs-mode", default="geometric",
                    choices=["geometric", "legacy"])
    ap.add_argument("--hidden-dim", type=int, default=128)
    # Read the choices from the registry rather than restating them, so adding
    # an architecture does not silently leave the viewers unable to load it.
    from marl.mappo_advanced import ARCHITECTURES

    ap.add_argument("--architecture", default=None, choices=ARCHITECTURES,
                    help="only needed for checkpoints written without a meta block")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--initialization-mode", default="legacy",
                    choices=["legacy", "stratified", "random"])
    ap.add_argument("--task-allocator", default="none",
                    choices=["none", "hungarian", "learned"],
                    help="optional explicit clot-task allocator for the low-level controller")
    ap.add_argument("--task-checkpoint", default="",
                    help="high-level allocator checkpoint when --task-allocator=learned")
    ap.add_argument("--allocation-interval", type=int, default=20)
    args = ap.parse_args()

    from environments.pv_render import History, PVRenderer, PVStyle

    style = PVStyle(width=args.width, height=args.height,
                    wall_opacity=args.wall_opacity,
                    views=args.views,
                    hud_position=args.layout,
                    hud_width=0 if args.no_hud else 420,
                    hud_height=0 if args.no_hud else 250,
                    camera_azimuth_per_frame=args.spin)

    # use_pybullet stays off: the scene is built from the vessel tree directly,
    # so there is no reason to pay for a physics client that renders nothing.
    env = Vascular3DMARLEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        render_mode=None,
        randomize_scenario=False,
        randomize_clots=True,
        robot_radius=args.robot_radius,
        use_pybullet=False,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        initialization_mode=args.initialization_mode,
    )

    policy = None
    if args.policy:
        from marl.policy_loader import load_policy

        policy = load_policy(
            args.policy, env, device=args.device,
            architecture=args.architecture, hidden_dim=args.hidden_dim,
        )
    else:
        from marl.geometric_control import policy_action

        if args.controller == "zero":
            print("no --policy given: random actions")
        else:
            def controller_policy(obs_dict, env_ref):
                raw = np.zeros((env_ref.num_robots, 3), np.float32)
                return policy_action(raw, obs_dict, env_ref,
                                     mode=args.controller, residual_scale=0.0)

            policy = controller_policy
            print(f"no --policy given: deterministic controller={args.controller}")

    obs, _ = env.reset(seed=args.seed)

    allocator_model = None
    allocator_device = None
    if args.task_allocator == "learned":
        if not args.task_checkpoint:
            raise SystemExit("--task-checkpoint is required for --task-allocator=learned")
        import torch

        from marl.hierarchical_allocator import HierarchicalAllocator, allocator_features

        probe_assignments = np.full((env.num_robots,), -1, dtype=np.int32)
        feature_probe, _ = allocator_features(env, probe_assignments)
        allocator_device = torch.device(args.device)
        checkpoint = torch.load(
            args.task_checkpoint, map_location=allocator_device, weights_only=False
        )
        meta = checkpoint["meta"]
        allocator_model = HierarchicalAllocator(
            feature_probe.shape[-1], meta["slots"], meta["hidden_dim"]
        ).to(allocator_device)
        allocator_model.load_state_dict(checkpoint["model"])
        allocator_model.eval()

    task_assignments = np.full((env.num_robots,), -1, dtype=np.int32)

    renderer = PVRenderer(style)
    renderer.build(env.tree, env.num_robots, env.horizon,
                   clot_draw_radius=env.clot_contact_radius * 0.62)
    hist = History()

    frames: list[np.ndarray] = []
    ret, step = 0.0, 0
    info: dict = {"removal_rate": 0.0, "success": False}
    wall_hits = np.zeros((env.num_robots,), dtype=bool)

    while step < args.max_frames * args.stride:
        if args.task_allocator != "none" and (
            step == 0 or step % args.allocation_interval == 0
        ):
            if args.task_allocator == "hungarian":
                from marl.task_allocator import balanced_assignment

                task_assignments = balanced_assignment(env).assignments
            else:
                from marl.hierarchical_allocator import allocator_features, sample_allocation

                features, valid = allocator_features(env, task_assignments)
                task_assignments = sample_allocation(
                    allocator_model, features, valid, allocator_device,
                    deterministic=True,
                ).assignments
            env.set_task_assignments(task_assignments)
            obs = env._build_observation()
        action = env.action_space.sample() if policy is None else policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        step += 1

        assigned = env._assigned_clot()
        geo = env._geodesic_to_target(assigned)
        # Non-finite geodesics (no live clot reachable) would break the plot's
        # autoscaling, so they are dropped to zero for display only.
        geo_plot = np.where(np.isfinite(geo), geo, 0.0)
        alive = env.clot_masses[: env.active_clots]
        base = np.maximum(env.clot_initial_mass[: env.active_clots], 1e-8)
        hist.push(step, geo_plot, info.get("removal_rate", 0.0), alive, reward)

        if step % args.stride == 0 or terminated or truncated:
            contacting = contact_mask(env)
            # The env reports the count, not the per-robot flags, so recover
            # "which robot hit a wall" from its distance to the lumen axis.
            axis_pt, axis_r = env.tree._axis_point(env.robot_positions,
                                                   env.robot_stations)
            radial = np.linalg.norm(env.robot_positions - axis_pt, axis=1)
            wall_hits = radial >= (axis_r - env.robot_radius) * 0.999

            states = robot_states(env, contacting, wall_hits)
            routes = None if args.no_routes else geodesic_routes(env, assigned)
            renderer.update(
                env.robot_positions, states,
                env.clot_positions[: env.active_clots], alive / base,
                clot_base_radius=env.clot_contact_radius * 0.62,
                routes=routes,
            )
            frames.append(renderer.frame(
                hist, header_text(env, step, info, ret, reward, assigned, states, geo_plot)
            ))

        if terminated or truncated:
            break

    renderer.close()
    env.close()

    if not frames:
        print("no frames captured")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in (".mp4", ".webm", ".mov"):
        import imageio.v2 as imageio

        # Even dimensions: libx264 rejects odd ones.
        h, w = frames[0].shape[:2]
        crop = (h - h % 2, w - w % 2)
        imageio.mimwrite(out, [f[: crop[0], : crop[1]] for f in frames],
                         fps=args.fps, quality=8, macro_block_size=1)
    else:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(args.fps, 1)), loop=0)

    print(f"wrote {out} ({len(frames)} frames, {frames[0].shape[1]}x"
          f"{frames[0].shape[0]}, {out.stat().st_size / 1e6:.2f} MB)")
    print(f"final: removal {info['removal_rate']:.3f} success {info['success']}")


if __name__ == "__main__":
    main()
