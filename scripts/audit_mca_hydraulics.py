"""Independent nodal-pressure audit of the unchanged MCA engineering model.

The production solvers reduce a resistance tree recursively. Here Kirchhoff's
equations are assembled directly and solved by a sparse linear solver after
collapsing only coincident (zero-resistance) junction nodes. Also compare the
trapezoidal resistance to the analytic integral for linearly tapered radii.
This is an implementation audit, NOT an assertion of physiological validity.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from environments.mca_compiled import CompiledMCAPhysicalEnv, solve_flow
from environments.mca_physical_env import DynamicsConfig
from scripts.train_mca_physical import atomic_json, reset_with_valid_particles

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/experiments/EXP_0024_MCA_DISTRIBUTED_DYNAMICS.json'


def nodal_flows(model, radii, *, exact_taper=False):
    """Solve pressures independently; do not call model._resistance/solve."""
    count = len(radii)
    parents = np.arange(count)
    def group(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return int(i)
    for node in model.order[1:]:
        if model.length[node] == 0:
            parents[group(node)] = group(model.parent[node])
    groups = np.array([group(i) for i in range(count)])
    unique, inverse = np.unique(groups, return_inverse=True)
    size = len(unique)
    root = int(inverse[model.root])
    known = {root: model.driving_pressure}
    edges = []
    sinks = []
    for node in model.order[1:]:
        length = model.length[node]
        if not length:
            continue
        first = model.parent[node]
        a, b = radii[first], radii[node]
        if min(a, b) <= 0:
            raise ValueError('This audit requires positive lumen radii')
        resistance = (length*(a*a+a*b+b*b)/(3*a**3*b**3) if exact_taper else
                      length*.5*(a**-4+b**-4))
        edges.append((int(inverse[first]), int(inverse[node]), 1/resistance, node))
    for node in model.order:
        if not model.children[node]:
            resistance = model.terminal_resistance[node]
            if resistance:
                sinks.append((int(inverse[node]), 1/resistance))
            else:
                known[int(inverse[node])] = 0.
    rows, cols, values = [], [], []
    for a, b, conductance, _ in edges:
        rows.extend([a, b, a, b]);cols.extend([a, b, b, a])
        values.extend([conductance, conductance, -conductance, -conductance])
    for node, conductance in sinks:
        rows.append(node);cols.append(node);values.append(conductance)
    matrix = coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()
    boundary = np.array(sorted(known), dtype=int)
    unknown = np.array([i for i in range(size) if i not in known], dtype=int)
    pressure = np.zeros(size)
    pressure[boundary] = [known[i] for i in boundary]
    pressure[unknown] = spsolve(matrix[unknown][:, unknown],
                               -matrix[unknown][:, boundary] @ pressure[boundary])
    edge_flow = {node: float(conductance*(pressure[a]-pressure[b]))
                 for a, b, conductance, node in edges}
    inlet_flow = float((matrix @ pressure)[root])
    return inlet_flow, edge_flow, pressure


def run(out):
    env = CompiledMCAPhysicalEnv(DynamicsConfig.from_json(CONFIG))
    reset_with_valid_particles(env, 810100000)
    model = env.flow_model
    _, hyd = env._compiled_arguments()
    cases = [(f'all_mass_{fraction:g}', np.full(env.num_clots, fraction))
             for fraction in (0., .25, .5, .75, 1.)]
    for clot in range(env.num_clots):
        for fraction in (0., .25, .5, .75):
            mass = np.ones(env.num_clots);mass[clot] = fraction
            cases.append((f'clot_{clot}_mass_{fraction:g}', mass))
    records = []
    for name, masses in cases:
        radii = env._radii(masses)
        reference = model.solve(radii)
        compiled, _ = solve_flow(radii, hyd)
        inlet, edge_flow, _ = nodal_flows(model, radii)
        exact_inlet, _, _ = nodal_flows(model, radii, exact_taper=True)
        nodes = np.array(sorted(edge_flow))
        predicted = np.array([edge_flow[i] for i in nodes])
        ref_flow = reference['station_inflow_mm3_s']
        error = float(np.max(np.abs(predicted-ref_flow[nodes])))
        rel_error = error/max(float(np.max(np.abs(ref_flow))), 1e-30)
        compiled_error = float(np.max(np.abs(compiled-ref_flow)))
        record = dict(case=name, masses=masses.tolist(),
                      inlet_flow_ml_min=reference['inlet_flow_ml_min'],
                      independent_inlet_flow_ml_min=inlet*60/1000,
                      maximum_node_flow_error_mm3_s=error, relative_error=rel_error,
                      compiled_flow_error_mm3_s=compiled_error,
                      exact_taper_inlet_difference_percent=100*(exact_inlet/inlet-1),
                      passed=rel_error<1e-8 and compiled_error<1e-8)
        records.append(record)
    # Point-target contact and lumen surface are distinct in the current model.
    contacts = []
    for j, station in enumerate(env.clot_stations):
        point = env.clot_positions_mm[j]
        edge = int(env.transport.nearest_edges(point[None])[0])
        _, r, _, _ = env.transport.coordinates(point[None], np.array([edge]), env.solution)
        radius = float(r[0]);body = env.config.robot_radius_mm
        flux = env.solution['station_inflow_mm3_s'][env.transport.ends[edge, 1]]
        mean = float(flux/(np.pi*radius**2))
        # At the clot centre cross-section, the furthest allowed centre from
        # the vessel axis that still counts as a point-target contact.
        contact_offset = min(env.config.contact_distance_mm, radius-body)
        contact_flow = 2*mean*(1-(contact_offset/radius)**2)
        contacts.append(dict(clot=j, lumen_radius_mm=radius,
                             contact_centre_radius_mm=contact_offset,
                             minimum_contact_cross_section_flow_mm_s=contact_flow,
                             propulsion_mm_s=env.config.robot_speed_mm_s,
                             minimum_gap_between_robot_surface_and_lumen_wall_mm=max(0.,radius-contact_offset-body),
                             point_contact_can_touch_lumen_wall=radius-contact_offset-body<=1e-12))
    summary = dict(passed=all(r['passed'] for r in records), cases=records,
                   contact_geometry=contacts, physical_inputs_changed=False,
                   policy_training_started=False, policy_success_rate=None,
                   scope='Numerical implementation and point-contact semantics only; no calibration claim')
    sources = [Path(__file__), CONFIG, ROOT/'environments/mca_physiology.py',
               ROOT/'environments/mca_compiled.py', ROOT/'environments/mca_physical_env.py']
    atomic_json(out/'protocol.json', dict(method='independent nodal sparse pressure solve',
                cases=len(cases), relative_flow_tolerance=1e-8,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    atomic_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    run(out)
