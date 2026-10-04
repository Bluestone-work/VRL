"""Diagnostic truth/measurement residuals, never fed back to a controller."""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.tracked_learning_episode import TrackedLearningEpisode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=int, required=True)
    parser.add_argument('--policy', choices=('joint', 'memory'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    episode = TrackedLearningEpisode(args.scene)
    errors, at_contact, contact_seen_safe = [], 0, 0
    detected = total = 0
    while not episode.done:
        e, p = episode.env, episode.packet
        active = np.flatnonzero(p.active)
        axis, radius, radial, _ = e.transport.coordinates(e.positions_mm[:3], e.edges[:3], e.solution)
        clearance = (radius-radial-e.config.robot_radius_mm)/radius
        for i in active:
            time_s, pos, vel = episode.sensor.processor.tracks[('robot', int(i))]
            predicted = pos+(e.elapsed_s-time_s)*vel
            errors.append((np.linalg.norm(predicted-e.positions_mm[i]),
                np.linalg.norm(p.navigation[i, 3:6]*e.config.robot_speed_mm_s-e.velocity_mm_s[i]),
                float(p.navigation[i, 9]-clearance[i]), float(e.elapsed_s-time_s)))
        detected += len(active); total += 3
        observed_clearance = p.navigation[:, 9].copy()
        observed_active = p.active.copy()
        choice = episode.library.conventional_choice(p, episode.valid, episode.details, memory=args.policy=='memory')
        episode.step(choice)
        touching = np.asarray(episode.info['wall_contact_s']) > 0
        at_contact += int((touching & observed_active).sum())
        contact_seen_safe += int((touching & observed_active & (observed_clearance > .08)).sum())
    a = np.asarray(errors)
    summary = dict(diagnostic_only=True, truth_to_controller=False, scene_seed=args.scene,
        policy=args.policy, frame_samples=len(a), detected_fraction=detected/total,
        mean_position_error_mm=float(a[:,0].mean()), p95_position_error_mm=float(np.quantile(a[:,0],.95)),
        mean_velocity_error_mm_s=float(a[:,1].mean()), p95_velocity_error_mm_s=float(np.quantile(a[:,1],.95)),
        mean_clearance_overestimate_fraction=float(a[:,2].mean()),
        p95_clearance_overestimate_fraction=float(np.quantile(a[:,2],.95)),
        mean_track_age_s=float(a[:,3].mean()), contact_agent_steps=at_contact,
        contact_agent_steps_with_reported_clearance_above_margin=contact_seen_safe,
        result=episode.result(args.policy))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(summary,f,indent=2)
    episode.close()
    print(json.dumps({k:v for k,v in summary.items() if k!='result'},indent=2))


if __name__ == '__main__': main()
