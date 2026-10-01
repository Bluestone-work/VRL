"""Read-only audit of observation scale and redundancy; no training, no env changes.

States come from deterministic rollouts of one policy on the shared development
layouts. At each sampled state the 76-feature prefix and the 36-feature routed
target block are the env's own observation; the four candidate 96-feature
obstacle blocks are recomputed from the same state with the module functions
the env uses. Samples within an episode are correlated diagnostic samples.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_obstacle_forecast import (bound_trajectory_features, forecast_particles,
                                                linear_particle_predictions, trajectory_features)
from environments.mca_physical_env import TARGET_SLOT_DIMS, DynamicsConfig
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_compiled import reset_with_valid_particles
from scripts.train_mca_physical import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASE_NAMES = (['pos_x', 'pos_y', 'pos_z', 'vel_t', 'vel_n', 'vel_b', 'tan_x', 'tan_y', 'tan_z',
               'radial_t', 'radial_n', 'radial_b', 'wall_clearance', 'radius', 'agent_mask',
               'target_geodesic', 'target_dir_t', 'target_dir_n', 'target_dir_b', 'target_mass',
               'time_left', 'flow_t', 'flow_n', 'flow_b', 'near_p_t', 'near_p_n', 'near_p_b',
               'near_pv_t', 'near_pv_n', 'near_pv_b', 'near_clearance', 'near_present',
               'robot_over_radius', 'stenosis_ratio', 'log_flow', 'alive_fraction'] +
              [f'risk{k}_{f}' for k in range(4) for f in
               ('rp_t', 'rp_n', 'rp_b', 'rv_t', 'rv_n', 'rv_b', 'clear', 'tmin', 'cmin', 'present')])
TARGET_NAMES = ([f'slot{j}_{f}' for j in range(4) for f in
                 ('dir_t', 'dir_n', 'dir_b', 'geodesic', 'mass', 'mine', 'shared', 'alive')] +
                ['sum_alive', 'sum_my_geodesic', 'sum_valid', 'sum_mass'])


def obstacle_blocks(env):
    """The four 96-feature variants exactly as env._observation builds them."""
    n = env.num_robots
    pos = env.positions_mm[:n]
    frame = np.stack((env.tree.tangents[env.robot_stations], env.tree.normals[env.robot_stations],
                      env.tree.binormals[env.robot_stations]), axis=1)
    ids = np.flatnonzero(env.active[n:])+n
    if not len(ids):
        return None
    velocities = env.transport.velocity_mm_s(env.positions_mm[ids], env.edges[ids], env.solution)
    c = env.config
    common = (pos, env.velocity_mm_s, frame, env.positions_mm[ids], velocities)
    tail = (c.robot_radius_mm, c.particle_radius_mm, c.robot_speed_mm_s, c.particle_safety_margin_mm)
    route = trajectory_features(*common, *forecast_particles(env, ids), *tail)
    linear_pred = linear_particle_predictions(env.positions_mm[ids], velocities)
    linear = trajectory_features(*common, *linear_pred, *tail)
    anchored = trajectory_features(*common, *linear_pred, *tail,
                                   reference_velocities=np.zeros_like(env.velocity_mm_s))
    return dict(route_raw=route, route_bounded=bound_trajectory_features(route),
                linear_bounded=bound_trajectory_features(linear),
                anchored_bounded=bound_trajectory_features(anchored),
                linear_raw=linear, anchored_raw=anchored)


def stats(x):
    a = np.abs(x)
    return dict(mean=float(x.mean()), std=float(x.std()), p95_abs=float(np.quantile(a, .95)),
                max_abs=float(a.max()), distinct=int(min(len(np.unique(x)), 1000)))


def r2(features, target):
    """Least-squares R^2 of target columns from features (+bias); held-out half."""
    half = len(features)//2
    x = np.column_stack((features, np.ones(len(features))))
    coef, *_ = np.linalg.lstsq(x[:half], target[:half], rcond=None)
    residual = target[half:]-x[half:]@coef
    total = ((target[half:]-target[:half].mean(0))**2).sum(0)
    keep = total > 1e-12
    return float(1-(residual[:, keep]**2).sum()/total[keep].sum()) if keep.any() else None


def effective_rank(x, fraction):
    x = x[:, x.std(0) > 1e-9]
    z = (x-x.mean(0))/x.std(0)
    s = np.linalg.svd(z, compute_uv=False)**2
    return int(np.searchsorted(np.cumsum(s)/s.sum(), fraction)+1), int(x.shape[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layouts', type=int, default=40)
    parser.add_argument('--stride', type=int, default=20)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    protocol = json.loads((ROOT/'configs/experiments/EXP_0034_ROUTED_PURE_RL.json').read_text())
    checkpoint = ROOT/'research/runs/EXP0034_MASKED_PREFLIGHT_20260930/policy_16384.pt'
    env = CompiledMCAPhysicalEnv(DynamicsConfig.from_json(ROOT/protocol['physics_config']))
    agent = make_physical_agent(env, seed=43, hidden_dim=protocol['hidden_dim'])
    agent.load(checkpoint, load_optimizers=False); agent.actor.eval(); agent.critic.eval()
    base, target, blocks = [], [], {}
    for index in range(args.layouts):
        reset_with_valid_particles(env, protocol['validation_seed_base']+index)
        obs = env._observation(); step = 0
        while not env._done:
            if step % args.stride == 0:
                variants = obstacle_blocks(env)
                if variants is not None:
                    alive = env.agent_mask.astype(bool)
                    base.append(obs['nodes'][alive, :76]); target.append(obs['nodes'][alive, 76:])
                    for key, value in variants.items():
                        blocks.setdefault(key, []).append(value[alive])
            # The masked checkpoint has zero weights on the appended block, so
            # routed-target values cannot influence these actions.
            _, _, _, action, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
            obs, _, _, _, _ = env.step(action); step += 1
        print(f'layout {index}: {step} steps, {sum(len(b) for b in base)} rows', flush=True)
    base = np.concatenate(base).astype(np.float64); target = np.concatenate(target).astype(np.float64)
    blocks = {k: np.concatenate(v).astype(np.float64) for k, v in blocks.items()}
    rows = len(base)
    report = dict(rows=rows, layouts=args.layouts, stride_steps=args.stride, policy=str(checkpoint.relative_to(ROOT)),
                  policy_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  scope='Deterministic rollouts of one near-parent policy (seed 43); correlated samples; alive robot rows only.')
    report['base_stats'] = {name: stats(base[:, i]) for i, name in enumerate(BASE_NAMES)}
    report['target_stats'] = {name: stats(target[:, i]) for i, name in enumerate(TARGET_NAMES)}
    report['block_stats'] = {k: dict(p95_abs=float(np.quantile(np.abs(v), .95)), max_abs=float(np.abs(v).max()))
                             for k, v in blocks.items()}
    # Exact identities from the code.
    radius_mm = base[:, 13]*1.5
    checks = dict(
        agent_mask_always_one=bool(np.all(base[:, 14] == 1)),
        robot_over_radius_from_radius_max_err=float(np.abs(base[:, 32]-0.08/np.maximum(radius_mm, 1e-12)).max()),
        log_flow_from_flow_max_err=float(np.abs(base[:, 34]-np.log1p(np.linalg.norm(base[:, 21:24], axis=1))).max()),
        near_present_equals_risk0_present=float(np.mean(base[:, 31] == base[:, 45])))
    # Is the nearest particle one of the four risk-ranked slots?
    near = base[:, 24:27]
    slots = np.stack([base[:, 36+10*k:39+10*k] for k in range(4)], 1)
    match = np.linalg.norm(slots-near[:, None], axis=-1).min(1) < 1e-4
    checks['nearest_particle_in_risk_top4'] = float(match[base[:, 31] == 1].mean())
    checks['nearest_particle_is_risk_slot0'] = float((np.linalg.norm(slots[:, 0]-near, axis=1) < 1e-4)[base[:, 31] == 1].mean())
    # Nearest target block (15-19) vs routed slots.
    slot = lambda j, f: target[:, TARGET_SLOT_DIMS*j+f]
    geo = np.stack([np.where(slot(j, 7) > 0, slot(j, 3), np.inf) for j in range(4)], 1)
    live = np.isfinite(geo).any(1)
    checks['target_geodesic_equals_min_slot_geodesic_max_err'] = float(np.abs(base[live, 15]-geo[live].min(1)).max())
    checks['slot_alive_equals_mass_positive'] = float(np.mean([np.mean((slot(j, 7) > 0) == (slot(j, 4) > 0)) for j in range(4)]))
    nearest = geo[live].argmin(1)
    route_dir = np.stack([target[live][np.arange(live.sum()), TARGET_SLOT_DIMS*nearest+f] for f in range(3)], 1)
    checks['straight_vs_route_bearing_cosine_mean'] = float(np.mean(np.sum(route_dir*base[live, 16:19], 1)))
    checks['target_mass_equals_nearest_slot_mass_max_err'] = float(np.abs(
        base[live, 19]-target[live][np.arange(live.sum()), TARGET_SLOT_DIMS*nearest+4]).max())
    report['identity_checks'] = checks
    # Linear reconstructability: how much of each 96 block is explained by the base 76.
    report['r2_from_base76'] = {k: r2(base, v) for k, v in blocks.items()}
    report['r2_route_raw_from_base76_plus_linear_raw'] = r2(np.column_stack((base, blocks['linear_raw'])), blocks['route_raw'])
    report['route_minus_linear_raw_abs'] = dict(
        mean=float(np.abs(blocks['route_raw']-blocks['linear_raw']).mean()),
        p95=float(np.quantile(np.abs(blocks['route_raw']-blocks['linear_raw']), .95)))
    report['r2_target_block_from_base76'] = r2(base, target)
    # Near-duplicate pairs and effective dimension.
    full = np.column_stack((base, target))
    names = BASE_NAMES+TARGET_NAMES
    varying = np.flatnonzero(full.std(0) > 1e-9)
    corr = np.corrcoef(full[:, varying].T)
    pairs = [(names[varying[i]], names[varying[j]], round(float(corr[i, j]), 4))
             for i in range(len(varying)) for j in range(i+1, len(varying)) if abs(corr[i, j]) > .95]
    report['constant_features'] = [names[i] for i in range(full.shape[1]) if full[:, i].std() <= 1e-9]
    report['correlated_pairs_abs_r_gt_0.95'] = pairs
    report['effective_rank'] = {name: {str(f): effective_rank(x, f) for f in (.95, .99)}
                                for name, x in (('base76', base), ('base76+target36', full),
                                                ('base76+route_bounded96', np.column_stack((base, blocks['route_bounded']))))}
    np.savez_compressed(args.out/'samples.npz', base=base, target=target, **blocks)
    atomic_json(args.out/'summary.json', report)
    print(json.dumps({k: report[k] for k in ('rows', 'identity_checks', 'r2_from_base76', 'r2_route_raw_from_base76_plus_linear_raw',
                                             'route_minus_linear_raw_abs', 'r2_target_block_from_base76', 'constant_features',
                                             'correlated_pairs_abs_r_gt_0.95', 'effective_rank', 'block_stats')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
