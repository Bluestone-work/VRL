"""Watch the trained policy in an interactive 3D window (local machine).

    python watch_gui.py                                    # random actions
    python watch_gui.py --policy experiments/multi_seed/gat_r5_c3/seed_43/best_policy.pt \
                        --architecture gat --robots 5

Mouse: drag to orbit, scroll to zoom, shift+drag to pan, 'r' to reset the view.

This needs a real display. On a headless server use `make_gif.py` instead, which
renders the same scene off-screen to a GIF or MP4 with the instrument panel
attached.

The renderer is the VTK one in `environments/pv_render.py`, shared with
`make_gif.py`. PyBullet's viewer was replaced because its TinyRenderer does not
composite transparency: a robot inside a translucent lumen was hidden outright,
not dimmed.

Per-step state is printed to the terminal rather than drawn into the window, so
you can orbit freely while still seeing what the swarm is doing.
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
from environments.vessel_geometry import ALL_SCENARIOS
from marl.mappo_advanced import ARCHITECTURES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None,
                    help="path to a .pt checkpoint; omit for random actions")
    ap.add_argument("--robots", type=int, default=5)
    ap.add_argument("--clots", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--robot-radius", type=float, default=0.0045,
                    help="must match training; use 0.0011 for anatomical runs")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--scenario", default="mca_stroke", choices=list(ALL_SCENARIOS))
    ap.add_argument("--reward-mode", default="baseline",
                    choices=["baseline", "milestone"])
    ap.add_argument("--obs-mode", default="geometric",
                    choices=["geometric", "legacy"])
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--architecture", default=None, choices=ARCHITECTURES,
                    help="only needed for checkpoints written without a meta block")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=20.0, help="playback speed cap")
    ap.add_argument("--render-stride", type=int, default=1,
                    help="refresh the live window every N simulator steps")
    ap.add_argument("--wall-opacity", type=float, default=0.16)
    ap.add_argument("--no-routes", action="store_true",
                    help="hide the robot-to-clot geodesics")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.render_stride < 1:
        ap.error("--render-stride must be at least 1")

    # Import after argparse so --help works without a display, and before the
    # env so pyvista is initialised for an on-screen window.
    from environments.pv_render import PVRenderer, PVStyle
    from make_gif import contact_mask, geodesic_routes, robot_states

    env = Vascular3DMARLEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        render_mode=None,
        randomize_scenario=False,     # keep one topology so it is easier to watch
        randomize_clots=True,
        robot_radius=args.robot_radius,
        use_pybullet=False,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
    )

    policy = None
    if args.policy:
        from marl.policy_loader import load_policy

        policy = load_policy(args.policy, env, device=args.device,
                             architecture=args.architecture,
                             hidden_dim=args.hidden_dim)
    else:
        print("no --policy given: driving with random actions")

    # No HUD in the live window: the panel is a pasted raster, which would sit
    # on top of a view you are meant to be able to orbit. The same numbers go to
    # the terminal instead.
    style = PVStyle(width=1280, height=800, hud_width=0, hud_height=0,
                    wall_opacity=args.wall_opacity)
    renderer = PVRenderer(style, off_screen=False)

    frame_budget = 1.0 / max(args.fps, 1e-6)
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        # The tree is resampled per reset only when randomize_scenario is set,
        # but clots and robots always are, so the scene is rebuilt per episode.
        renderer.build(env.tree, env.num_robots, env.horizon,
                       clot_draw_radius=env.clot_contact_radius * 0.62)
        # build() replaces the plotter, so each episode needs a freshly
        # initialized interactor before pump() can process window events.
        renderer.show_interactive()

        ret, steps = 0.0, 0
        info: dict = {}
        while True:
            t0 = time.time()
            action = (env.action_space.sample() if policy is None
                      else policy(obs, env))
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            steps += 1

            render_due = (
                steps % args.render_stride == 0 or terminated or truncated
            )
            status_due = steps % 10 == 0
            if render_due or status_due:
                assigned = env._assigned_clot()
                contacting = contact_mask(env)
                axis_pt, axis_r = env.tree._axis_point(
                    env.robot_positions, env.robot_stations
                )
                radial = np.linalg.norm(env.robot_positions - axis_pt, axis=1)
                wall_hits = radial >= (axis_r - env.robot_radius) * 0.999

            if render_due:
                alive = env.clot_masses[: env.active_clots]
                base = np.maximum(
                    env.clot_initial_mass[: env.active_clots], 1e-8
                )
                renderer.update(
                    env.robot_positions,
                    robot_states(env, contacting, wall_hits),
                    env.clot_positions[: env.active_clots], alive / base,
                    clot_base_radius=env.clot_contact_radius * 0.62,
                    routes=(
                        None if args.no_routes
                        else geodesic_routes(env, assigned)
                    ),
                )
                renderer.pump()

            if status_due:
                print(f"  ep {ep} step {steps:3d} | return {ret:+7.2f} | "
                      f"removal {info['removal_rate']:.3f} | "
                      f"lysing {int(contacting.sum())} | "
                      f"wall {info['wall_collisions']}", end="\r", flush=True)

            if terminated or truncated:
                break
            sleep = frame_budget - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

        print(
            f"ep {ep}: return {ret:+8.2f} | steps {steps:3d} | "
            f"removal {info['removal_rate']:.3f} | success {info['success']} | "
            f"wall_hits {info['wall_collisions']}"
        )

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
