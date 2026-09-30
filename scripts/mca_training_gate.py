"""Necessary feasibility checks before claiming all-clot-success training.

This gate can prove that some tasks are impossible. Passing it is NOT proof of
reachability, hardware calibration, training convergence, or policy success.
"""
import numpy as np
import hashlib
import json
from pathlib import Path


def task_feasibility(env):
    config = env.config
    points = env.clot_positions_mm
    if len(points) > 1:
        distances = np.linalg.norm(points[:, None]-points[None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        disjoint = bool(distances.min() > 2*config.contact_distance_mm)
    else:
        disjoint = True
    initial_mass = float(env.initial_mass.sum())
    capacity = config.num_robots*config.lysis_mass_per_s*config.episode_duration_s
    single_target = config.contact_model == 'stenosis_surface'
    impossible = (disjoint or single_target) and capacity < initial_mass-1e-12
    return dict(all_clear_proven_impossible=impossible, disjoint_contact_regions=disjoint,
                at_most_one_contact_target_per_robot=single_target,
                initial_mass=initial_mass, optimistic_removal_capacity=capacity,
                optimistic_minimum_time_s=initial_mass/(config.num_robots*config.lysis_mass_per_s)
                if config.lysis_mass_per_s > 0 else None,
                sufficient_for_training_readiness=False,
                reason='Insufficient total lysis capacity within the episode' if impossible
                else 'Mass/time necessary bound passes; transport and sustained contact still require validation')


def require_all_clear_capacity(env, *, allow_unreachable_baseline=False):
    report = task_feasibility(env)
    if report['all_clear_proven_impossible'] and not allow_unreachable_baseline:
        raise ValueError(
            'All-clot success is provably unreachable: optimistic removal '
            f"{report['optimistic_removal_capacity']:.6g} < initial mass {report['initial_mass']:.6g}. "
            'Validate task duration and physical reachability before success-target training. '
            '--allow-unreachable-baseline is only for explicitly labelled engineering diagnostics.')
    return report


def require_feasibility_certificate(protocol, root):
    """Bind constructive reachability evidence to the actual surface-task code.

    Engineering override flags never bypass this new-task launch condition.
    """
    root=Path(root)
    for field,minimum,backend in [('feasibility_certificate',20,'compiled'),
                                  ('reference_certificate',1,'reference')]:
        if not protocol.get(field):raise ValueError(f'Missing {field}; validate before training')
        report=json.loads((root/protocol[field]).read_text())
        if not report.get('passed') or report.get('config')!=protocol['physics_config']:
            raise ValueError('Invalid feasibility certificate or wrong physics configuration')
        rows=report.get('episodes',[])
        if len({r['seed'] for r in rows})<minimum or any(
                not r['success'] or r['backend']!=backend or r['lost_robots'] or
                r['max_action_norm']>1+1e-12 or any(m!=0 for m in r['remaining_masses'])
                for r in rows):
            raise ValueError('Incomplete or failed constructive feasibility evidence')
        hashes=report.get('source_sha256',{})
        if protocol['physics_config'] not in hashes or 'environments/mca_physical_env.py' not in hashes:
            raise ValueError('Missing certificate source hashes')
        for name,digest in hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Stale feasibility certificate: {name}')
        if protocol.get('require_randomization_boundaries'):
            config=json.loads((root/protocol['physics_config']).read_text())
            for multiplier in (config['inlet_flow_multiplier_min'],config['inlet_flow_multiplier_max']):
                boundary=[r for r in report.get('boundary_episodes',[]) if r.get('boundary_multiplier')==multiplier]
                if len({r['seed'] for r in boundary})<(5 if backend=='compiled' else 1) or any(
                        not r['success'] or r['backend']!=backend or r['lost_robots'] or
                        r['max_action_norm']>1+1e-12 or any(m!=0 for m in r['remaining_masses']) or
                        r['reset']['flow_multiplier']!=multiplier for r in boundary):
                    raise ValueError('Missing or failed flow-boundary feasibility evidence')
            if backend=='compiled' and {r['reset'].get('particle_layout') for r in rows} != {
                    'length_uniform','branch_balanced','branch_clustered'}:
                raise ValueError('Feasibility evidence must cover all three particle layout modes')
        if protocol.get('require_random_clot_evidence'):
            if any(r['reset'].get('clot_initialization')!='random_branches' for r in rows) or len({
                    tuple(r['reset'].get('clot_stations',[])) for r in rows})!=len(rows):
                raise ValueError('Feasibility evidence requires independently randomized clot layouts')
    return dict(constructive_feasibility_validated=True,policy_success_rate=None,
                clinical_or_hardware_validated=False)
