"""Re-evaluate the navigation methods on the HCC multifocal portal construct.

The controller implementations are imported unchanged.  Only the anatomy,
scene seed, flow and dynamic-obstacle count are replaced.  Legacy controllers
run through ``LysisEpisode`` with dissolution disabled; the reported task is
target navigation, so the metrics are comparable with the IPPO navigation
environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# The workstation has two recorded unstable cores (6/7); pin evaluation to
# the remaining affinity set before NumPy/Torch initialize their kernels.
if hasattr(os, 'sched_getaffinity'):
    _allowed = os.sched_getaffinity(0) - {6, 7}
    if _allowed:
        os.sched_setaffinity(0, _allowed)
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.vessel_sdf import VesselSDF, project_local_commands
from scripts.benchmark_lysis import NAV_GOAL_TOLERANCE_MM, navigation_outcome

ANATOMY = "hcc_portal_multifocal"
DIFF = (
    ("easy", 4, .025),
    ("medium", 8, .05),
    ("hard", 16, .10),
    ("stress", 32, .15),
)
PREDICTORS = [str(ROOT.parent.parent / "VRL_archive_20261016/research/EXP0090_20261011/predictor" / f"predictor_s{i}.pt") for i in range(3)]
TURBO = ROOT.parent.parent / "VRL_archive_20261016/research/EXP0093_20261011/turbo_hard_s0/policy.pt"
# The v1 run stopped at iteration 258/1000 (132 k environment steps) with no
# difficulty rotation and is not a finished policy.  v2 completed 1000
# iterations / 512 k environment steps with the four-difficulty rotation, so it
# is the registered HCC checkpoint for ours_hcc.
OURS_MODEL = ROOT / "research/runs/HCC_OURS_S41_500K_20261019_v2/policy.pt"
IPPO_MODELS = {
    "IPPO_S41": ROOT / "research/runs/EXP0141_IPPO_SEMANTIC_REDUCED_S41_20261016/final_model.zip",
    "IPPO_S42": ROOT / "research/runs/EXP0142_IPPO_SEMANTIC_REDUCED_S42_20261016/final_model.zip",
    "IPPO_S43": ROOT / "research/runs/EXP0143_IPPO_SEMANTIC_REDUCED_S43_20261016/final_model.zip",
}


def run_classic(method, difficulty, seed, horizon=300.0, centerline_match=False):
    from scripts.benchmark_lysis import LysisEpisode, SettleGuard
    from marl import factorial as FA
    from marl.deployable_sensing import DeployableConfig

    _, obstacles, flow = DIFF[difficulty]
    # Keep sensing and all controller logic at their registered defaults.  The
    # navigation study fixes latency at zero so difficulty isolates flow and
    # obstacle density, matching the IPPO protocol.
    kw = dict(horizon=horizon, sensing="image", navigation=True, centerline_match=centerline_match,
              sense_cfg=DeployableConfig(latency_steps=0),
              flow_inlet_mm_s=flow, dynamic_obstacle_count=obstacles)
    if method == "ours_hcc":
        from marl.predictive_nav_rl import NavigationActor, RLNavigator
        from marl.deployable_sensing import DeployableConfig
        payload = torch.load(OURS_MODEL, map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        net = NavigationActor(int(cfg.get("width", 64)), float(cfg.get("proposal_bias", .2)))
        net.load_state_dict(payload["state"])
        ep = LysisEpisode(3, ANATOMY, seed, tpg=True, **kw)
        controller = RLNavigator(ep, net=net, stochastic=False,
                                 predictors=cfg.get("predictors"), uncertainty_fallback=False)
        tag = method
    elif method in ("ours", "online", "mppi", "cbf", "turbo"):
        FA.PRED.update(ensemble=PREDICTORS, fallback_std=None,
                       weights=None, turbo=str(TURBO))
        local_name = {"ours": "ours", "online": "online", "mppi": "mppi",
                      "cbf": "cbf", "turbo": "turbo"}[method]
        ep = FA.make("A_TPG", 3, ANATOMY, seed, **kw)
        controller = FA.local(local_name, ep)
        tag = method
    elif method == "pursuit":
        ep = LysisEpisode(3, ANATOMY, seed, tpg=True, **kw)
        controller = SettleGuard(ep)
        tag = method
    elif method == "stpg":
        from marl.lysis_baselines import SwitchableTPG
        ep = LysisEpisode(3, ANATOMY, seed, tpg=True, **kw)
        controller = SettleGuard(ep)
        tag = method
    elif method == "pac":
        from marl.lysis_baselines import PacNMPC
        ep = LysisEpisode(3, ANATOMY, seed, tpg=False, fallback="park", **kw)
        controller = PacNMPC(ep)
        tag = method
    else:
        raise ValueError(method)

    reached = np.zeros(ep.n, dtype=bool)
    reach_times = np.full(ep.n, np.nan)
    wall_after_reach = 0.
    start_t = time.monotonic()
    try:
        sdf = VesselSDF(ep.sensor)
        est = ep.observe()
        targets = ep.plan_targets_now(est)
        switcher = SwitchableTPG(ep.coord) if method == "stpg" and ep.coord is not None else None
        while True:
            if switcher is not None:
                switcher.update(ep.env.elapsed_s)
            rule = ep.ctl.act(targets, est)
            hold = ep.hold(est)
            local = controller(ep, est, targets, rule, hold)
            local, _ = project_local_commands(local, est, targets, hold, ep.ctl.frames(est), sdf, ep.ctl.body)
            done, outcome = ep.step(est, local, hold)
            wall_after_reach += float(outcome["wall"][reached].sum())
            newly = ~reached & (ep.navigation_distances() <= NAV_GOAL_TOLERANCE_MM)
            reach_times[newly] = ep.env.elapsed_s
            reached |= newly
            if done or reached.all() or not ep.env.active[:ep.n].all():
                break
            est = ep.observe()
        row = ep.row(tag, wall_after_reach_s=wall_after_reach)
        if method == "pac":
            row["pac_bound_fail_rate"] = controller.bound_fail / max(controller.calls, 1)
        if method == "stpg" and switcher is not None:
            row.update(stpg_switches=switcher.switches, stpg_solves=switcher.solves,
                       stpg_infeasible=switcher.infeasible)
        return _finish_row(row, ep, method, difficulty, seed, reached, reach_times, start_t)
    finally:
        ep.close()


def _finish_row(row, ep, method, difficulty, seed, reached, reach_times, start_t):
    row = dict(row)
    success = bool(np.all(reached) and row["lost"] == 0)
    row.update(navigation_outcome(success, row))
    row.update(accepted_seed=ep.manifest["accepted_seed"], navigation_targets=ep.navigation_targets.tolist(),
               centerline_match=ep.sensor.cam.centerline_match, anatomy=ANATOMY, difficulty=int(difficulty),
               difficulty_name=DIFF[difficulty][0], dynamic_obstacle_count=DIFF[difficulty][1],
               nominal_flow_inlet_mm_s=DIFF[difficulty][2], navigation_reached=int(reached.sum()),
               navigation_time_s=(float(np.nanmax(reach_times)) if success else None),
               strict_navigation_success=row["strict_success"],
               collision_free_navigation=row["collision_free_success"],
               elapsed_wall_s=float(time.monotonic()-start_t) if start_t else None)
    return row


def run_ippo(label, difficulty, seed, horizon=300.0, centerline_match=False):
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import IPPOAgentVecEnv
    from scripts.train_sb3_ippo import load_navigation_ppo
    _, obstacles, flow = DIFF[difficulty]
    physical = CurriculumVascularEnv(seed=seed, clusters=3, horizon=horizon,
                                     curriculum=False, reward_mode="navigation",
                                     zero_latency=True, direct_action=True, centerline_match=centerline_match)
    physical.train_anatomies = [ANATOMY]
    # Match the classical SettleGuard safety layer.  Without this, IPPO is
    # executed with a different wall/near-target intervention and the result
    # mixes policy quality with wrapper semantics.
    model = load_navigation_ppo(str(IPPO_MODELS[label]), device="cpu")
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced", safety_parity=True,
                          flow_feature="flow" in model.observation_space.spaces, exit_mask=False)
    start = time.monotonic()
    try:
        model.set_env(env)
        obs = env.reset(options=dict(anatomy=ANATOMY, difficulty=int(difficulty),
                                     scene_seed=int(seed), flow=float(flow), latency=0,
                                     dynamic_obstacle_count=int(obstacles), scenario_family="uniform"))
        # SB3 auto-resets on done. Capture identities from the evaluated episode.
        evaluated_scene = physical.scene.copy()
        targets = physical.nav_targets.tolist()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            if bool(done[0]):
                info = infos[0]
                metrics = dict(info.get("episode_metrics", {}))
                return dict(method=label, anatomy=ANATOMY, seed=int(seed), difficulty=int(difficulty),
                            centerline_match=centerline_match, accepted_seed=evaluated_scene["accepted_seed"],
                            scenario_hash=evaluated_scene["scenario_hash"], navigation_targets=targets,
                            task_success=metrics["task_success"], strict_success=metrics["strict_success"],
                            termination=metrics.get("termination", metrics.get("termination_reason")),
                            integration_failure=bool(info.get("integration_failure")),
                            difficulty_name=DIFF[difficulty][0], dynamic_obstacle_count=int(obstacles),
                            nominal_flow_inlet_mm_s=float(flow), navigation_success=bool(info.get("navigation_success", False)),
                            strict_navigation_success=bool(metrics.get("strict_success", False)),
                            navigation_reached=int(info.get("navigation_reached", metrics.get("navigation_reached", 0))),
                            wall_after_reach_s=metrics.get("wall_after_reach_s"),
                            navigation_time_s=(float(metrics.get("elapsed_s", 0.))
                                               if info.get("navigation_success") else None),
                            wall_contact_s=float(metrics.get("wall_contact_s", np.asarray(info.get("agent_wall_contact_s", 0.)).sum())),
                            lost=int(metrics.get("lost", np.asarray(info.get("agent_lost", 0.)).sum())),
                            particle_collision_events=float(metrics.get("particle_collision_events", np.asarray(info.get("agent_particle_collision_events", 0.)).sum())),
                            particle_contact_s=float(metrics.get("particle_contact_s", np.asarray(info.get("agent_particle_contact_s", 0.)).sum())),
                            collision_free_success=metrics["collision_free_success"],
                            collision_free_navigation=metrics["collision_free_success"],
                            elapsed_wall_s=float(time.monotonic()-start))
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--centerline-match", action="store_true", help="use the same closest-centerline matcher for every method")
    ap.add_argument("--horizon", type=float, default=300.)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--repeat-offset", type=int, default=0,
                    help="shift the paired seed block so parallel shards cover disjoint repeats")
    ap.add_argument("--methods", default="ours,online,pursuit,pac,mppi,cbf,turbo,stpg,IPPO_S41,IPPO_S42,IPPO_S43")
    ap.add_argument("--ippo-root", type=Path, default=None,
                    help="directory containing HCC_IPPO_S41/S42/S43 run folders; use retrained checkpoints")
    ap.add_argument("--ours-model", type=Path, default=None,
                    help="HCC-trained predictive selector policy.pt for method ours_hcc")
    ap.add_argument("--ippo-model", action="append", default=[], metavar="LABEL=PATH",
                    help="override one IPPO label with an explicit .zip, e.g. "
                         "IPPO_S41=research/runs/HCC_MIX_S41_500K_20261028/final_model.zip; "
                         "repeatable, applied after --ippo-root")
    args = ap.parse_args()
    global IPPO_MODELS
    if args.repeats < 1 or args.repeat_offset < 0 or args.horizon <= 0:
        ap.error("repeats and horizon must be positive and repeat-offset nonnegative")
    if args.ippo_root is not None:
        IPPO_MODELS = {}
        for s in (41, 42, 43):
            v2 = args.ippo_root / f"HCC_IPPO_S{s}_500K_20261019_v2/final_model.zip"
            v1 = args.ippo_root / f"HCC_IPPO_S{s}_500K_20261019/final_model.zip"
            IPPO_MODELS[f"IPPO_S{s}"] = v2 if v2.exists() else v1
    for spec in args.ippo_model:
        label, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--ippo-model expects LABEL=PATH, got {spec!r}")
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"IPPO override not found: {target}")
        IPPO_MODELS[label.strip()] = target
    if args.ours_model is not None:
        global OURS_MODEL
        OURS_MODEL = args.ours_model
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods or len(set(methods)) != len(methods):
        ap.error("specify one or more distinct methods")
    for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
        for d in range(4):
            seed = 1910000000 + repeat*100 + d
            for method in methods:
                started = time.monotonic()
                try:
                    row = (run_ippo(method, d, seed, args.horizon, args.centerline_match) if method in IPPO_MODELS or method.startswith("IPPO_")
                           else run_classic(method, d, seed, args.horizon, args.centerline_match))
                    row["repeat"] = repeat
                    rows.append(row)
                    print(json.dumps({"method": method, "difficulty": d,
                                      "success": row.get("navigation_success"),
                                      "strict": row.get("strict_navigation_success"),
                                      "wall_s": round(row.get("wall_contact_s", 0.), 3),
                                      "seconds": round(time.monotonic()-started, 2)}, ensure_ascii=False), flush=True)
                except Exception as exc:
                    import traceback
                    rows.append(dict(method=method, anatomy=ANATOMY, difficulty=d, repeat=repeat,
                                     seed=seed, error=repr(exc), trace=traceback.format_exc()))
                    print(json.dumps({"method": method, "difficulty": d, "error": repr(exc)}), flush=True)
    result = dict(protocol="HCC navigation with frozen goals and post-step geodesic arrival scoring",
                  navigation_goal_tolerance_mm=NAV_GOAL_TOLERANCE_MM, ippo_exit_mask=False, centerline_match=args.centerline_match,
                  anatomy=ANATOMY, source_image="/home/wj/桌面/新的构造.jpg", horizon_s=args.horizon,
                  repeats=args.repeats, difficulties=[dict(index=i, name=n, dynamic_obstacles=o, flow=f)
                                                      for i,(n,o,f) in enumerate(DIFF)],
                  methods=methods, predictor_checkpoints=[str(p) for p in PREDICTORS],
                  turbo_checkpoint=str(TURBO), ippo_checkpoints={k:str(v) for k,v in IPPO_MODELS.items()},
                  ours_checkpoint=str(OURS_MODEL),
                  ours_checkpoint_sha256=(hashlib.sha256(Path(OURS_MODEL).read_bytes()).hexdigest()
                                          if Path(OURS_MODEL).exists() else None),
                  repeat_offset=args.repeat_offset,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["difficulty"])].append(r)
    summary = {}
    for (method, level), group in sorted(by.items()):
        valid = [r for r in group if "error" not in r and not r.get("integration_failure")]
        complete = len(valid) == len(group)
        summary[f"{method}|{level}"] = dict(
            episodes=len(valid), requested=len(group), errors=len(group)-len(valid), complete=complete,
            navigation_success_rate=float(np.mean([r["navigation_success"] for r in valid])) if complete else None,
            strict_navigation_success_rate=float(np.mean([r["strict_navigation_success"] for r in valid])) if complete else None,
            wall_contact_s=float(np.mean([r["wall_contact_s"] for r in valid])) if complete else None,
            lost_rate=float(np.mean([r["lost"] > 0 for r in valid])) if complete else None,
            collision_episode_rate=float(np.mean([r["particle_collision_events"] > 0 for r in valid])) if complete else None,
            mean_navigation_time_s=(float(np.mean([r["navigation_time_s"] for r in valid if r["navigation_success"]]))
                                    if complete and any(r["navigation_success"] for r in valid) else None))
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(not group["complete"] for group in summary.values()):
        raise SystemExit("Incomplete HCC evaluation. See error rows in the output.")


if __name__ == "__main__":
    main()
