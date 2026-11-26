"""Diagnostics for one executed interval. Distances and commands use mm and seconds."""
import numpy as np


def executed_step(ep, est, sdf, hold, requested, guarded, projected, targets):
    n = ep.n
    return {
        "elapsed_s": float(ep.env.elapsed_s),
        "requested_local_velocity": np.asarray(requested).tolist(),
        "wallguard_settle_local_velocity": np.asarray(guarded).tolist(),
        "sdf_projected_local_velocity": np.asarray(projected).tolist(),
        "tpg_hold": np.asarray(hold, bool).tolist(),
        "final_sent_command_world": ep.sent_world.tolist(),
        "position_mm_world": ep.env.positions_mm[:n].tolist(),
        "estimated_body_clearance_mm_before": [
            float(sdf(est.pos[i], int(est.edge[i]))[0] - ep.ctl.body) for i in range(n)
        ],
        "distance_to_target_mm": ep.env._target_distances()[np.arange(n), targets].tolist(),
        "wall_contact_s_step": np.asarray(ep.info["wall_contact_s"], float).tolist(),
    }
