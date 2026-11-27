"""Paired planar scheduling on two fixed crossing corridors with parked goal occupancy."""
import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marl.tpg_coordinator import TPGCoordinator

GRID = 0.2
DIAMETER = 0.6
HORIZON = 30.0


def corridors(seed, conflict):
    rng = np.random.default_rng(seed)
    # Vary arrival offsets continuously within predeclared conflict strata.
    x = rng.uniform(4.2, 6.2)
    y = x + (rng.uniform(-0.2, 0.2) if conflict else rng.uniform(1.6, 3.2))
    u = GRID * np.arange(61)
    return [np.column_stack((u-x, 0*u)), np.column_stack((0*u, u-y))]


def write_inputs(routes, path, template):
    graph = ET.Element("graphml")
    road = ET.SubElement(graph, "graph", edgedefault="directed")
    task = ET.Element("root")
    offset = 0
    for route in routes:
        for j, (x,y) in enumerate(route):
            node = ET.SubElement(road, "node", id=f"n{offset+j}")
            ET.SubElement(node, "data").text = f"{x:.12f},{y:.12f}"
        for j in range(len(route)-1):
            a, b = f"n{offset+j}", f"n{offset+j+1}"
            ET.SubElement(road, "edge", source=a, target=b)
            ET.SubElement(road, "edge", source=b, target=a)
        ET.SubElement(task, "agent", start_id=str(offset), goal_id=str(offset+len(route)-1))
        offset += len(route)
    config = ET.parse(template)
    config.find("./algorithm/agent_size").text = str(DIAMETER/2)
    ET.ElementTree(graph).write(path/"map.xml")
    ET.ElementTree(task).write(path/"task.xml")
    config.write(path/"config.xml")


def run_ccbs(routes, path, binary):
    try:
        run = subprocess.run([str(binary.resolve()), "map.xml", "task.xml", "config.xml"],
                             cwd=path, capture_output=True, text=True, timeout=45, check=True)
    except subprocess.TimeoutExpired:
        return None, "process_timeout"
    (path/"solver.txt").write_text(run.stdout + run.stderr)
    found = re.search(r"So(?:u)?lution found:\s*(true|false)", run.stdout)
    if not found:
        raise ValueError(f"unrecognized CCBS status in {path}")
    if found[1] == "false":
        return None, "no_solution"
    agents = ET.parse(path/"task_log.xml").findall("./log/agent")
    if sorted(int(a.attrib["number"]) for a in agents) != list(range(len(routes))):
        raise ValueError("CCBS returned an incomplete set of paths")
    tracks = []
    for route, agent in zip(routes, sorted(agents, key=lambda a: int(a.attrib["number"]))):
        track = [[0., *route[0]]]
        for part in agent.findall("./path/section"):
            d = part.attrib
            start = [float(d["start_i"]), float(d["start_j"])]
            end = [float(d["goal_i"]), float(d["goal_j"])]
            dt = float(d["duration"])
            length = np.linalg.norm(np.subtract(end, start))
            if not np.allclose(track[-1][1:], start) or dt < 0 or (length > 1e-8 and abs(length-dt)>1e-5):
                raise ValueError("discontinuous or non-unit-speed CCBS path")
            if dt > 0:
                track.append([track[-1][0]+dt, *end])
        if not np.allclose(track[-1][1:], route[-1]):
            raise ValueError("CCBS path misses its assigned goal")
        tracks.append(np.array(track))
    return tracks, "completed"


def execute(routes, enabled):
    xyz = np.column_stack((np.concatenate(routes), np.zeros(sum(map(len,routes)))))
    n = len(routes[0])
    starts, goals = [0,n], [n-1,2*n-1]
    env = SimpleNamespace(transport=SimpleNamespace(points=xyz), robot_stations=starts,
                          clot_stations=goals, clot_positions_mm=xyz[goals],
                          config=SimpleNamespace(lysis_mass_per_s=1.), initial_mass=np.ones(2))
    tpg = TPGCoordinator(env, [[0],[1]], lambda a,b: list(range(b,a-1,-1)),
                         d_min_mm=DIAMETER, buffer_mm=0., step_mm=GRID, dwell_s=0.) if enabled else None
    k = np.zeros(2, dtype=int)
    records = [[[0., *r[0]]] for r in routes]
    for tick in range(1, round(HORIZON/GRID)+1):
        pos = xyz[np.array(starts)+k]
        hold = tpg.gate(pos, np.ones(2, bool)) if enabled else np.zeros(2, bool)
        for i, route in enumerate(routes):
            if k[i] < len(route)-1:
                if not hold[i]:
                    k[i] += 1
                records[i].append([tick*GRID, *route[k[i]]])
        if np.all(k == n-1):
            return list(map(np.array,records)), "completed"
    return list(map(np.array,records)), "horizon"


def score(tracks):
    # Linear interpolation also parks the early finisher at its endpoint.
    times = np.unique(np.concatenate([p[:,0] for p in tracks]))
    xy = [np.array([np.interp(times,p[:,0],p[:,j]) for j in (1,2)]).T for p in tracks]
    delta = xy[0]-xy[1]
    mins = float(np.linalg.norm(delta[0]))
    seconds = 0.
    for r, v, dt in zip(delta[:-1], np.diff(delta, axis=0), np.diff(times)):
        vv = float(v @ v)
        t = np.clip(-(r@v)/vv,0,1) if vv else 0.
        mins = min(mins,float(np.linalg.norm(r+t*v)))
        if vv < 1e-24:
            seconds += dt * (np.linalg.norm(r) < DIAMETER-1e-9)
            continue
        rv = float(r@v)
        disc = rv**2-vv*(r@r-DIAMETER**2)
        if disc > 0:
            lo, hi = (-rv-np.sqrt(disc))/vv, (-rv+np.sqrt(disc))/vv
            seconds += dt*max(0.,min(1.,hi)-max(0.,lo))
    return dict(min_separation=float(mins), spacing_pair_s=float(seconds),
                elapsed_s=float(times[-1]), safe=bool(mins >= DIAMETER-1e-6))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=20, help="number of high/low pairs")
    a = p.parse_args()
    if a.pairs < 1:
        p.error("pairs must be positive")
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in range(a.pairs):
        for conflict in (False, True):
            routes = corridors(seed, conflict)
            path = a.out / f"s{seed}_{int(conflict)}"
            path.mkdir()
            write_inputs(routes, path, a.config)
            for method in ("uncoordinated", "vrl_tpg", "ccbs"):
                tracks, status = (run_ccbs(routes,path,a.binary) if method=="ccbs"
                                  else execute(routes,method=="vrl_tpg"))
                metrics = score(tracks) if tracks is not None else {}
                if status == "completed" and metrics["elapsed_s"] > HORIZON+1e-8:
                    status = "late_completion"
                rows.append(dict(seed=seed, conflict=conflict, method=method, status=status,
                                 safe_completion=status=="completed" and metrics.get("safe",False),
                                 capped_time_s=metrics["elapsed_s"] if status=="completed" else HORIZON,
                                 **metrics))
    (a.out/"paired_coordination.json").write_text(json.dumps(dict(
        revision="b8f45e166cf39ee7e8e06f0bfe4fbab0dee68932", diameter=DIAMETER,
        grid=GRID, speed=1., horizon=HORIZON, episodes=rows), indent=2)+"\n")


if __name__ == "__main__":
    main()
