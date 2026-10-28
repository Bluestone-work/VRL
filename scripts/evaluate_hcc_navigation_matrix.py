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


def _target_metric(ep, reached, reach_times, start_t):
    est = ep.observe()
    tgt = ep.plan_targets_now(est)
    for i, c in enumerate(np.asarray(tgt, dtype=int)):
        if reached[i] or c < 0:
            continue
        if np.linalg.norm(ep.env.clot_positions_mm[c] - est.pos[i]) <= .30:
            reached[i] = True
            reach_times[i] = float(ep.env.elapsed_s)
    return est, tgt


def run_classic(method, difficulty, seed, horizon=300.0):
    from scripts.benchmark_lysis import LysisEpisode, SettleGuard, WallGuard
    from marl import factorial as FA
    from marl.deployable_sensing import DeployableConfig

    _, obstacles, flow = DIFF[difficulty]
    # Keep sensing and all controller logic at their registered defaults.  The
    # navigation study fixes latency at zero so difficulty isolates flow and
    # obstacle density, matching the IPPO protocol.
    kw = dict(horizon=horizon, sensing="image",
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
        switcher = SwitchableTPG(ep.coord) if ep.coord is not None else None
        tag = method
    elif method == "pac":
        from marl.lysis_baselines import PacNMPC
        ep = LysisEpisode(3, ANATOMY, seed, tpg=False, fallback="park", **kw)
        controller = PacNMPC(ep)
        tag = method
    else:
        raise ValueError(method)

    # Do not let the legacy lysis event terminate the navigation episode.
    from dataclasses import replace
    ep.env.config = replace(ep.env.config, lysis_mass_per_s=0.)
    reached = np.zeros(ep.n, dtype=bool)
    reach_times = np.full(ep.n, np.nan)
    start_t = time.monotonic()
    done = False
    try:
        while True:
            est, tgt = _target_metric(ep, reached, reach_times, start_t)
            if method == "stpg" and switcher is not None:
                switcher.update(ep.env.elapsed_s)
            rule = ep.ctl.act(tgt, est)
            hold = ep.hold(est)
            done, _ = ep.step(est, controller(ep, est, tgt, rule, hold), hold)
            # Count a target reached by the just-executed command before a
            # finite-horizon terminal is handled.
            try:
                _target_metric(ep, reached, reach_times, start_t)
            except Exception:
                pass
            if np.all(reached) and not np.any(~ep.env.active[:ep.n]):
                break
            if done:
                break
        row = ep.row(tag)
        if method == "pac":
            row["pac_bound_fail_rate"] = controller.bound_fail / max(controller.calls, 1)
        if method == "stpg" and switcher is not None:
            row.update(stpg_switches=switcher.switches, stpg_solves=switcher.solves,
                       stpg_infeasible=switcher.infeasible)
        return _finish_row(row, ep, method, difficulty, seed, reached, reach_times, start_t)
    finally:
        ep.close()


def _finish_row(row, ep, method, difficulty, seed, reached=None, reach_times=None, start_t=0.):
    if reached is None:
        # For the STPG/PAC wrappers, derive reached status from the last
        # recorded target estimate by replaying one observation.
        reached = np.zeros(ep.n, dtype=bool)
        reach_times = np.full(ep.n, np.nan)
        try:
            est = ep.observe(); tgt = ep.plan_targets_now(est)
            for i, c in enumerate(np.asarray(tgt, dtype=int)):
                if c >= 0 and np.linalg.norm(ep.env.clot_positions_mm[c] - est.pos[i]) <= .30:
                    reached[i] = True
                    reach_times[i] = float(ep.env.elapsed_s)
        except Exception:
            pass
    row = dict(row)
    row.update(anatomy=ANATOMY, difficulty=int(difficulty),
               difficulty_name=DIFF[difficulty][0], dynamic_obstacle_count=DIFF[difficulty][1],
               nominal_flow_inlet_mm_s=DIFF[difficulty][2], navigation_reached=int(reached.sum()),
               navigation_success=bool(np.all(reached) and int(row.get("lost", 0)) == 0),
               navigation_time_s=(float(np.nanmax(reach_times)) if np.all(reached) else None),
               strict_navigation_success=bool(np.all(reached) and int(row.get("lost", 0)) == 0
                                              and float(row.get("wall_contact_s", 0.)) < 1.),
               collision_free_navigation=bool(np.all(reached) and int(row.get("lost", 0)) == 0
                                              and int(row.get("particle_collision_events", 0)) == 0),
               elapsed_wall_s=float(time.monotonic()-start_t) if start_t else None)
    return row


def run_ippo(label, difficulty, seed, horizon=300.0):
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import IPPOAgentVecEnv
    from stable_baselines3 import PPO
    _, obstacles, flow = DIFF[difficulty]
    physical = CurriculumVascularEnv(seed=seed, clusters=3, horizon=horizon,
                                     curriculum=False, reward_mode="navigation",
                                     zero_latency=True, direct_action=True)
    physical.train_anatomies = [ANATOMY]
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced")
    model = PPO.load(str(IPPO_MODELS[label]), env=env, device="cpu")
    start = time.monotonic()
    try:
        obs = env.reset(options=dict(anatomy=ANATOMY, difficulty=int(difficulty),
                                     scene_seed=int(seed), flow=float(flow), latency=0,
                                     dynamic_obstacle_count=int(obstacles), scenario_family="uniform"))
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            if bool(done[0]):
                info = infos[0]
                metrics = dict(info.get("episode_metrics", {}))
                return dict(method=label, anatomy=ANATOMY, seed=int(seed), difficulty=int(difficulty),
                            difficulty_name=DIFF[difficulty][0], dynamic_obstacle_count=int(obstacles),
                            nominal_flow_inlet_mm_s=float(flow), navigation_success=bool(info.get("navigation_success", False)),
                            strict_navigation_success=bool(metrics.get("strict_success", False)),
                            navigation_reached=int(info.get("navigation_reached", metrics.get("navigation_reached", 0))),
                            navigation_time_s=(float(metrics.get("elapsed_s", 0.))
                                               if info.get("navigation_success") else None),
                            wall_contact_s=float(metrics.get("wall_contact_s", np.asarray(info.get("agent_wall_contact_s", 0.)).sum())),
                            lost=int(metrics.get("lost", np.asarray(info.get("agent_lost", 0.)).sum())),
                            particle_collision_events=float(metrics.get("particle_collision_events", np.asarray(info.get("agent_particle_collision_events", 0.)).sum())),
                            particle_contact_s=float(metrics.get("particle_contact_s", np.asarray(info.get("agent_particle_contact_s", 0.)).sum())),
                            collision_free_navigation=bool(info.get("navigation_success", False) and float(metrics.get("particle_collision_events", 0.)) == 0),
                            elapsed_wall_s=float(time.monotonic()-start))
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=float, default=300.)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--repeat-offset", type=int, default=0,
                    help="shift the paired seed block so parallel shards cover disjoint repeats")
    ap.add_argument("--methods", default="ours,online,pursuit,pac,mppi,cbf,turbo,stpg,IPPO_S41,IPPO_S42,IPPO_S43")
    ap.add_argument("--ippo-root", type=Path, default=None,
                    help="directory containing HCC_IPPO_S41/S42/S43 run folders; use retrained checkpoints")
    ap.add_argument("--ours-model", type=Path, default=None,
                    help="HCC-trained predictive selector policy.pt for method ours_hcc")
    args = ap.parse_args()
    if args.ippo_root is not None:
        global IPPO_MODELS
        IPPO_MODELS = {}
        for s in (41, 42, 43):
            v2 = args.ippo_root / f"HCC_IPPO_S{s}_500K_20261019_v2/final_model.zip"
            v1 = args.ippo_root / f"HCC_IPPO_S{s}_500K_20261019/final_model.zip"
            IPPO_MODELS[f"IPPO_S{s}"] = v2 if v2.exists() else v1
    if args.ours_model is not None:
        global OURS_MODEL
        OURS_MODEL = args.ours_model
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
        for d in range(4):
            seed = 1910000000 + repeat*100 + d
            for method in methods:
                started = time.monotonic()
                try:
                    row = (run_ippo(method, d, seed, args.horizon) if method.startswith("IPPO_")
                           else run_classic(method, d, seed, args.horizon))
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
    result = dict(protocol="hcc_portal_multifocal navigation; algorithms unchanged; zero-latency paired evaluation",
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
        if "error" not in r: by[(r["method"], r["difficulty"])].append(r)
    summary = {f"{m}|{d}": {"episodes":len(v),
                              "navigation_success_rate":float(np.mean([x["navigation_success"] for x in v])),
                              "strict_navigation_success_rate":float(np.mean([x["strict_navigation_success"] for x in v])),
                              "wall_contact_s":float(np.mean([x.get("wall_contact_s",0.) for x in v])),
                              "lost_rate":float(np.mean([x.get("lost",0)>0 for x in v])),
                              "collision_episode_rate":float(np.mean([x.get("particle_collision_events",0)>0 for x in v])),
                              "mean_navigation_time_s":float(np.mean([x["navigation_time_s"] for x in v if x.get("navigation_time_s") is not None])) if any(x.get("navigation_time_s") is not None for x in v) else None}
               for (m,d),v in sorted(by.items())}
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
