"""Read-only residual instrumentation; never routes truth back into scheduling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.run_option_learning import make_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=int,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'request.json').write_text(json.dumps(dict(scene_seed=args.scene,
        diagnostic_only=True,truth_to_controller=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    episode=make_episode(args.scene)
    original=episode.prepare
    positions=[];velocities=[];pair_errors=[];pair_velocity_errors=[]
    missing=frames=unsafe_but_reported_safe=0
    last=-1.
    def record():
        nonlocal missing,frames,last,unsafe_but_reported_safe
        env,p=episode.env,episode.packet
        if env.elapsed_s==last:return
        last=float(env.elapsed_s);frames+=1
        missing+=int((~p.active).sum())
        for i in np.flatnonzero(p.active):
            stamp,position,velocity=episode.sensor.processor.tracks[('robot',int(i))]
            predicted=position+(env.elapsed_s-stamp)*velocity
            positions.append(float(np.linalg.norm(predicted-env.positions_mm[i])))
            velocities.append(float(np.linalg.norm(p.navigation[i,3:6]-env.velocity_mm_s[i])))
            for j in range(i+1,episode.cfg.clusters):
                if not p.peer_visible[i,j]:continue
                true_distance=float(np.linalg.norm(env.positions_mm[j]-env.positions_mm[i]))
                reported=float(np.linalg.norm(p.peer_relative_mm[i,j]))
                pair_errors.append(reported-true_distance)
                pair_velocity_errors.append(float(np.linalg.norm(p.peer_relative_velocity_mm_s[i,j]-
                    (env.velocity_mm_s[j]-env.velocity_mm_s[i]))))
                unsafe_but_reported_safe+=int(true_distance<2. and reported>=2.)
    def instrumented_prepare():
        original();record()
    episode.prepare=instrumented_prepare
    record()
    try:
        while not episode.done:
            episode.step_option(episode.conventional_options('memory'))
        result=episode.result('memory')
        reference=json.loads(args.reference.read_text().splitlines()[0])
        equivalent=result['final_state_hash']==reference['final_state_hash']
        summarize=lambda values:dict(count=len(values),mean=float(np.mean(values)),
            p95=float(np.quantile(values,.95)),maximum=float(np.max(values))) if values else dict(count=0)
        summary=dict(diagnostic_only=True,truth_to_controller=False,
            scene_seed=args.scene,policy='memory',frames=frames,missing_track_agent_frames=missing,
            position_error_mm=summarize(positions),reported_velocity_error_mm_s=summarize(velocities),
            pair_distance_overestimate_mm=summarize(pair_errors),
            reported_pair_velocity_error_mm_s=summarize(pair_velocity_errors),
            visible_pair_frames_true_below_2_reported_above_2=unsafe_but_reported_safe,
            same_final_state_as_reference=equivalent,reference=str(args.reference),
            note='One scene; correlations are not causal identification. Environment velocity is the reported physical-step velocity, not a hardware truth measurement.',
            result=result)
        (args.out/'diagnostic.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='result'},indent=2))
        if not equivalent:raise RuntimeError('Instrumented replay changed final state; do not attribute residuals to reference')
    finally:episode.close()


if __name__=='__main__':main()
