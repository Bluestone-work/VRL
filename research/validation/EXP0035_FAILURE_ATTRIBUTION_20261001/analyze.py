"""Failure attribution over traced EXP35 evaluation episodes (read-only).

usage: analyze.py TRACE_ROOT OUT_JSON
Distances are geodesic along the centreline (mm) unless stated. Contact needs
euclidean AND geodesic <= 0.12 mm (localized_point model).
"""
import json, sys
from pathlib import Path
import numpy as np

CONTACT = .12
DT = .1
root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob('seed_*/ep_*.npz')):
    z = np.load(f)
    seed_train = int(f.parent.name.split('_')[1]); seed = int(f.stem.split('_')[1])
    t, mass, geo, euc, act, vel, route, assign = (z[k] for k in ('t', 'mass', 'geo', 'euc', 'act', 'vel', 'route', 'assign'))
    active, branch, contact, wall, flow, pcont = (z[k] for k in ('active', 'branch', 'contact', 'wall', 'flow', 'pcontact'))
    T, n = active.shape; k = mass.shape[1]
    geo = np.where(geo < 0, np.inf, geo)
    # geodesic distance is masked to inf for cleared clots; recompute "clot alive" from mass
    final = z['final_mass']; success = bool(z['success'])
    mass_next = np.vstack([mass[1:], final[None]])
    clear_step = np.array([int(np.argmax(mass_next[:, j] <= 0)) if np.any(mass_next[:, j] <= 0) else -1 for j in range(k)])
    dm = mass.sum(1)-mass_next.sum(1)
    progress_steps = np.flatnonzero(dm > 1e-12)
    last_progress = int(progress_steps[-1]) if len(progress_steps) else -1
    n_cleared = int((final <= 0).sum())
    speed = np.linalg.norm(vel, axis=2)
    cmd = np.linalg.norm(act, axis=2)
    # cosine between executed action and route bearing to own assigned target
    ra = np.take_along_axis(route, np.clip(assign, 0, k-1)[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0]
    cos_route = (act*ra).sum(2)/np.maximum(cmd*np.linalg.norm(ra, axis=2), 1e-9)
    valid = active & (cmd > 1e-9) & (np.linalg.norm(ra, axis=2) > .5) & (assign >= 0)
    # geodesic progress toward assigned target per step
    ga = np.take_along_axis(geo, np.clip(assign, 0, k-1)[:, :, None], axis=2)[:, :, 0]
    ga_next = np.vstack([ga[1:], ga[-1:]])
    dprog = np.where(np.isfinite(ga) & np.isfinite(ga_next), ga-ga_next, np.nan)  # mm closer per step
    row = dict(train_seed=seed_train, seed=seed, success=success, elapsed=float(z['elapsed']), lost=int(z['lost']),
               n_cleared=n_cleared, removal=float(z['removal']), last_progress_s=(last_progress+1)*DT if last_progress >= 0 else 0.,
               clear_times=sorted(((clear_step[clear_step >= 0]+1)*DT).tolist()),
               cos_route_mean=float(np.nanmean(np.where(valid, cos_route, np.nan))),
               wall_frac=float((wall.sum()/max(active.sum()*DT, 1e-9))),
               particle_contact_s=float(pcont.sum()), mean_speed=float(speed[active].mean()),
               stop_frac=float(((cmd < 1e-9) & active).sum()/max(active.sum(), 1)))
    # per-clot participation: robots with contact while that clot lost mass
    part = []
    for j in range(k):
        loss_steps = np.flatnonzero((mass[:, j]-mass_next[:, j]) > 1e-12)
        if len(loss_steps):
            near = (euc[loss_steps, :, j] <= CONTACT+1e-9) & (contact[loss_steps] > 0)
            part.append(int(near.any(0).sum()))
    row['robots_per_clot'] = part
    row['clear_duration_s'] = []
    for j in range(k):
        loss_steps = np.flatnonzero((mass[:, j]-mass_next[:, j]) > 1e-12)
        if clear_step[j] >= 0 and len(loss_steps):
            row['clear_duration_s'].append(float((clear_step[j]-loss_steps[0]+1)*DT))
    # approach behaviour: in the final 2 mm geodesic before first contact of each cleared clot
    approach = []
    for j in range(k):
        loss_steps = np.flatnonzero((mass[:, j]-mass_next[:, j]) > 1e-12)
        if not len(loss_steps):
            continue
        s0 = loss_steps[0]
        d = geo[:s0, :, j]
        dn = np.vstack([d[1:], geo[s0:s0+1, :, j]]) if s0 else d
        win = np.isfinite(d) & (d < 2.) & active[:s0]
        if win.any():
            approach.append(float(np.mean(((d-dn)/DT)[win])))
    row['approach_speed_cleared'] = approach
    if not success:
        rem = np.flatnonzero(final > 0)
        detail = []
        for j in rem:
            dj = np.where(active, geo[:, :, j], np.inf)  # inf after clearance impossible (alive)
            ej = np.where(active, euc[:, :, j], np.inf)
            closest = dj.min(1)
            after = slice(max(last_progress+1, 0), T)
            touched = float(mass[-1, j] if False else final[j]) < 1-1e-9
            on_branch = (branch == z['clot_branch'][j]) & active
            # robots assigned to this clot in the dead window
            assigned_dead = (assign[after] == j) & active[after]
            win = closest[after]
            # oscillation: path/net displacement of the closest robot over the dead window
            idx = np.argmin(np.where(active[after], geo[after, :, j], np.inf), axis=1)
            pos = z['pos'][after][np.arange(len(idx)), idx]
            path = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() if len(pos) > 1 else 0.
            net = np.linalg.norm(pos[-1]-pos[0]) if len(pos) > 1 else 0.
            # geodesic progress of assigned robots toward this clot in the dead window
            prog = dprog[after][assigned_dead] if assigned_dead.any() else np.array([np.nan])
            cr = cos_route[after][assigned_dead & valid[after]] if (assigned_dead & valid[after]).any() else np.array([np.nan])
            flow_ratio = np.linalg.norm(flow[after], axis=2)[assigned_dead] if assigned_dead.any() else np.array([np.nan])
            v_assigned = speed[after][assigned_dead] if assigned_dead.any() else np.array([np.nan])
            detail.append(dict(clot=int(j), mass_left=float(final[j]), touched=bool(touched),
                               min_geo_ever=float(closest.min()), min_geo_dead=float(win.min()) if len(win) else None,
                               med_geo_dead=float(np.median(win)) if len(win) else None,
                               min_euc_ever=float(ej.min()),
                               ever_on_branch=bool(on_branch.any()),
                               frac_time_robot_on_branch=float(on_branch.any(1).mean()),
                               n_assigned_dead=float(assigned_dead.sum(1).mean()) if len(win) else 0.,
                               dead_path_mm=float(path), dead_net_mm=float(net),
                               dead_progress_mm_per_s=float(np.nanmean(prog)/DT),
                               dead_frac_receding=float(np.nanmean(prog < -1e-6)) if np.isfinite(prog).any() else None,
                               dead_cos_route=float(np.nanmean(cr)),
                               dead_speed=float(np.nanmean(v_assigned)),
                               dead_flow=float(np.nanmean(flow_ratio)),
                               dead_wall_frac=float(np.where(assigned_dead, wall[after], 0).sum()/max(assigned_dead.sum()*DT, 1e-9)) if len(win) else None))
        row['remaining'] = detail
    rows.append(row)
Path(sys.argv[2]).write_text(json.dumps(rows))
print(len(rows), 'episodes')
