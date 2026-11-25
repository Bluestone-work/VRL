"""Compare the frozen HCC development layouts at legacy and reference healthy inlet flow."""
import argparse
import json
from pathlib import Path

import numpy as np

from environments.mca_physiology import optimistic_wall_upstream_margin
from scripts.train_sb3_curriculum import CurriculumVascularEnv


def sections(sim, positions, edges):
    solution, transport = sim.solution, sim.transport
    _, radius, radial, _ = transport.coordinates(positions, edges, solution)
    downstream = transport.ends[edges, 1]
    flux = solution["station_inflow_mm3_s"][downstream]
    mean = flux / (np.pi * radius**2)
    local = np.linalg.norm(transport.velocity_mm_s(positions, edges, solution), axis=1)
    np.testing.assert_allclose(local, 2 * mean * np.maximum(1 - (radial / radius)**2, 0))
    rows = []
    for i in range(len(edges)):
        margin = optimistic_wall_upstream_margin(mean[i], radius[i], sim.config.robot_radius_mm,
                                                 sim.config.robot_speed_mm_s)
        rows.append(dict(edge=int(edges[i]), radius_mm=radius[i], radial_mm=radial[i], q_ml_min=.06*flux[i],
                         mean_mm_s=mean[i], center_mm_s=2*mean[i], local_mm_s=local[i],
                         optimistic_wall_margin_mm_s=margin))
    return rows


def audit(scene):
    env = CurriculumVascularEnv(seed=1041, clusters=3, curriculum=False, reward_mode="navigation",
                                zero_latency=True, direct_action=True, centerline_match=True, fixed_scenes=[scene])
    env.train_anatomies = [scene["anatomy"]]
    try:
        env.reset()
        sim = env.ep.env
        root = sim.flow_model.root
        healthy = sim.flow_model.solve()
        actual = sim.solution
        goals = sim.clot_positions_mm[env.nav_targets]
        if "healthy_inlet_ml_min" in scene:
            np.testing.assert_allclose(healthy["inlet_flow_ml_min"], scene["healthy_inlet_ml_min"])
        assert actual["maximum_conservation_residual_mm3_s"] < 1e-8
        return dict(seed=scene["seed"], layout_hash=env.scene["scenario_hash"],
                    targets=env.nav_targets.tolist(), inlet_diameter_mm=2*actual["radius_mm"][root],
                    healthy_q_ml_min=healthy["inlet_flow_ml_min"], inlet_q_ml_min=actual["inlet_flow_ml_min"],
                    inlet_mean_mm_s=actual["mean_speed_mm_s"][root], active_speed_mm_s=sim.config.robot_speed_mm_s,
                    starts=sections(sim, sim.positions_mm[:3], sim.edges[:3]),
                    goals=sections(sim, goals, sim.transport.nearest_edges(goals)))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes-out", type=Path, required=True)
    parser.add_argument("--healthy-inlet", type=float, default=638.6, help="healthy inlet Q in mL/min")
    parser.add_argument("--active-speed", type=float, help="zero-flow active speed in mm/s")
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())["rows"]
    scenes = [p["scene"] for p in pairs if not p["ippo"]["navigation_success"]]
    reference = []
    for scene in scenes:
        spec = dict(scene, healthy_inlet_ml_min=args.healthy_inlet)
        if args.active_speed is not None:
            spec["active_speed_mm_s"] = args.active_speed
        reference.append(spec)
    legacy, calibrated = [audit(s) for s in scenes], [audit(s) for s in reference]
    for left, right in zip(legacy, calibrated):
        assert left["layout_hash"] == right["layout_hash"] and left["targets"] == right["targets"]
    report = dict(reference=("Pazahr et al. PMID 24169068, ten healthy fasting volunteers, mean portal "
                             "flow 638.6 +/- 202.3 mL/min" if args.healthy_inlet == 638.6 else "user-selected healthy flow"),
                  assumptions="Synthetic geometry, steady pressure after occlusion, model terminal resistances. "
                              "Wall margin assumes full active speed at smallest geometrically accessible gap.",
                  legacy=legacy, reference_healthy_inlet=calibrated)
    for path, value in ((args.out, report), (args.scenes_out, reference)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
