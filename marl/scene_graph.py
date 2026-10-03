"""Topology-aware scene graph for learned navigation (no route controller inside).

The graph is built only from the vessel geometry, the solved flow field and the
physical state of robots, clots and particles. It never reads the environment's
route tables, next-hop lookahead, target allocation or the routed observation
columns: those belong to the teacher. tests/test_scene_graph.py enforces this by
making every route/allocation accessor raise while the graph is built.

Nodes
  vessel   centreline stations, subsampled to about one per 2 mm; junctions,
           leaves and the inlet are always kept
  robot    every robot slot (inactive robots are masked)
  clot     every clot (cleared clots are masked)
  particle every particle slot (exited particles are masked)

Topology enters through pairwise relations between the stations nodes are
attached to: geodesic distance along the vessel graph, vessel adjacency and the
downstream/upstream relation of the rooted tree. Everything else is relative
geometry, so the representation has no fixed node count and no fixed anatomy.

`extract_scene(env)` returns plain numpy arrays (static part cached per tree);
`collate_scenes` pads a list of scenes into batched torch tensors. The dataset
stores exactly the extracted arrays, so training and inference share one path.
"""
from __future__ import annotations

import numpy as np

VESSEL, ROBOT, CLOT, PARTICLE = range(4)
VESSEL_SPACING_MM = 2.0
VESSEL_DIM, ROBOT_DIM, CLOT_DIM, PARTICLE_DIM = 9, 13, 3, 4


def _station_tables(env):
    """Station geodesic matrix (mm), depth from the inlet and vessel node subsample. Pure geometry."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    tree = env.tree
    n = tree.n_stations
    rows, cols, weights = [], [], []
    for i, neighbours in enumerate(tree.station_graph):
        for j, w in neighbours:
            rows.append(i); cols.append(j); weights.append(w)
    scale = float(tree.physical_mm_per_unit)
    graph = csr_matrix((np.asarray(weights)*scale, (rows, cols)), shape=(n, n))
    geodesic = shortest_path(graph, directed=False).astype(np.float32)
    degree = np.array([len(x) for x in tree.station_graph])
    inlet = int(tree.inlet_station)
    keep = degree != 2
    keep[inlet] = True
    for branch in tree.branches:
        ids = np.arange(branch.start, branch.stop+1)
        arc = geodesic[branch.start, ids]
        keep[ids[np.diff(np.floor(arc/VESSEL_SPACING_MM), prepend=-1) > 0]] = True
        keep[branch.stop] = True
    kept = np.flatnonzero(keep)
    # Vessel edges: kept stations joined by a chain of non-kept stations.
    index = -np.ones(n, np.int64); index[kept] = np.arange(len(kept))
    adjacency = np.zeros((len(kept), len(kept)), bool)
    for a, s in enumerate(kept):
        stack, seen = [s], {s}
        while stack:
            u = stack.pop()
            for v, _ in tree.station_graph[u]:
                if v in seen:
                    continue
                seen.add(v)
                if keep[v]:
                    adjacency[a, index[v]] = True
                else:
                    stack.append(v)
    depth = geodesic[inlet]
    return dict(geodesic=geodesic, depth=depth.astype(np.float32), kept=kept.astype(np.int64),
                vessel_adjacency=adjacency, degree=degree, total_length_mm=np.float32(depth.max()))


def _static(env):
    key = id(env.tree)
    cache = getattr(env, '_scene_graph_static', None)
    if cache is None or cache[0] != key:
        cache = (key, _station_tables(env))
        env._scene_graph_static = cache
    return cache[1]


def extract_scene(env):
    """Arrays describing the current scene. Reads geometry and physical state only."""
    s = _static(env)
    tree, t, cfg = env.tree, env.transport, env.config
    n = env.num_robots
    kept = s['kept']
    healthy = np.asarray(env.flow_model.healthy_radius_mm, np.float64)
    radius = np.asarray(env.solution['radius_mm'], np.float64)
    speed = np.asarray(env.solution['mean_speed_mm_s'], np.float64)
    deg = s['degree'][kept]
    vessel = np.column_stack((
        np.log(healthy[kept]), radius[kept]/np.maximum(healthy[kept], 1e-9),
        np.log1p(speed[kept]/cfg.robot_speed_mm_s), deg == 1, deg == 2, deg >= 3,
        kept == int(tree.inlet_station), s['depth'][kept]/s['total_length_mm'],
        np.full(len(kept), float(tree.territory.retrograde)))).astype(np.float32)
    pos = env.positions_mm.astype(np.float64)
    active = env.active.astype(bool)
    frames = np.stack((tree.tangents, tree.normals, tree.binormals), axis=1).astype(np.float64)
    robot_station = np.asarray(env.robot_stations, np.int64)
    axis, lumen, radial, _ = t.coordinates(pos[:n], env.edges[:n], env.solution)
    flow = t.velocity_mm_s(pos[:n], env.edges[:n], env.solution)
    local = lambda frame, v: np.einsum('nij,nj->ni', frame, v)
    rframe = frames[robot_station]
    robot = np.column_stack((
        local(rframe, env.velocity_mm_s)/cfg.robot_speed_mm_s,
        local(rframe, (pos[:n]-axis)/np.maximum(lumen[:, None], 1e-9)),
        (lumen-radial-cfg.robot_radius_mm)/np.maximum(lumen, 1e-9),
        np.log(np.maximum(lumen, 1e-6)),
        local(rframe, flow)/cfg.robot_speed_mm_s,
        np.full(n, max(0., 1-env.elapsed_s/cfg.episode_duration_s)),
        np.full(n, active[:n].mean()))).astype(np.float32)
    mass = env.masses/np.maximum(env.initial_mass, 1e-12)
    clot = np.column_stack((mass, mass > 0, np.log(np.maximum(radius[env.clot_stations], 1e-6)))).astype(np.float32)
    pid = np.arange(n, len(pos))
    p_station = t.ends[env.edges[pid], 0].astype(np.int64)
    p_vel = t.velocity_mm_s(pos[pid], env.edges[pid], env.solution)
    particle = np.column_stack((np.log1p(np.linalg.norm(p_vel, axis=1)/cfg.robot_speed_mm_s),
                                np.full(len(pid), cfg.particle_radius_mm/cfg.robot_radius_mm),
                                active[pid], np.ones(len(pid)))).astype(np.float32)
    return dict(
        anatomy=str(tree.scenario), tree_key=np.int64(hash(tree.points.tobytes()) & 0x7fffffffffff),
        geodesic=s['geodesic'], depth=s['depth'], kept=kept, vessel_adjacency=s['vessel_adjacency'],
        station_points=t.points.astype(np.float32), station_frames=frames.astype(np.float32),
        vessel=vessel, robot=robot, clot=clot, particle=particle,
        robot_pos=pos[:n].astype(np.float32), robot_vel=env.velocity_mm_s.astype(np.float32),
        robot_station=robot_station, robot_active=active[:n],
        clot_pos=env.clot_positions_mm.astype(np.float32), clot_station=np.asarray(env.clot_stations, np.int64),
        clot_alive=mass > 0,
        particle_pos=pos[pid].astype(np.float32), particle_vel=p_vel.astype(np.float32),
        particle_station=p_station, particle_active=active[pid],
        robot_radius=np.float32(cfg.robot_radius_mm), particle_radius=np.float32(cfg.particle_radius_mm),
        robot_speed=np.float32(cfg.robot_speed_mm_s))


STATIC_KEYS = ('geodesic', 'depth', 'kept', 'vessel_adjacency', 'station_points', 'station_frames')
