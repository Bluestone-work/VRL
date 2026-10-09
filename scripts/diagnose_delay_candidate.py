"""Single-candidate delay-aware pursuit screening.

The candidate uses only delayed deployable estimates and route geometry. It
extrapolates the measured position by latency * dt * measured velocity before
target allocation and route command generation. The physics step still uses
the real environment state and the original observation estimate.
"""
import argparse, json, multiprocessing as mp, traceback
from pathlib import Path
import numpy as np

ANATOMIES=json.loads(Path('configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']['train']

def run(spec):
    anatomy,n,seed,lat,flow,method=spec
    ep=None
    try:
        from scripts.benchmark_lysis import LysisEpisode, WallGuard, SettleGuard
        from marl.deployable_sensing import DeployableConfig, Estimate
        ep=LysisEpisode(n, anatomy, seed, flow_inlet_mm_s=flow,
                        sense_cfg=DeployableConfig(latency_steps=lat))
        guard=WallGuard(ep); entries={}; departures=0; prev_near={}; near_time=0.; eligible=0.
        while True:
            sensed=ep.observe()
            control_est=sensed
            if method=='predictive':
                # All values are from the deployable estimate. Extrapolation is
                # causal and bounded to avoid runaway dead reckoning.
                horizon=float(lat)*float(ep.env.config.control_dt_s)
                pos=sensed.pos+np.clip(sensed.vel,-3.,3.)*horizon
                control_est=Estimate(pos=pos,vel=sensed.vel,edge=sensed.edge,station=sensed.station,
                                     active=sensed.active,particles=sensed.particles,peers_rel=sensed.peers_rel,
                                     peers_vis=sensed.peers_vis,clot_alive=sensed.clot_alive)
            tgt=ep.plan_targets_now(control_est); rule=ep.ctl.act(tgt,control_est); hold=ep.hold(control_est)
            local=SettleGuard(ep)(ep,control_est,tgt,rule,hold) if method=='settle' else guard(ep,control_est,tgt,rule,hold)
            now=float(ep.env.elapsed_s); near=np.zeros(n,bool)
            for i,t in enumerate(tgt):
                if t<0 or not sensed.active[i]: continue
                d=float(np.linalg.norm(ep.env.clot_positions_mm[t]-sensed.pos[i])); near[i]=d<.3
                key=(i,int(t)); entries.setdefault(key,0)
                if near[i]: entries[key]+=1
                if prev_near.get(key,False) and not near[i]: departures+=1
                prev_near[key]=bool(near[i])
            done,info=ep.step(sensed,local,hold)
            dt=float(ep.env.elapsed_s)-now; eligible += float(((np.asarray(tgt)>=0)&sensed.active).sum())*dt; near_time += float(near.sum())*dt
            if done: break
        row=ep.row(method)
        row.update(anatomy_split='train',latency_steps=lat,flow_inlet_mm_s=flow,
                   first_neighborhood_entry_s=None if not entries else min((k for k,v in entries.items() if v),default=(None,None))[0] if any(entries.values()) else None,
                   neighborhood_residence_ratio=near_time/max(eligible,1e-9),departures_after_entry=departures,
                   targets_cleared=int(np.sum(ep.env.masses<=0)),failure_stage='success' if row['strict_success'] else 'incomplete')
        ep.close(); return row
    except Exception:
        return dict(error=traceback.format_exc(),method=method,anatomy=anatomy,clusters=n,seed=seed,latency_steps=lat)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--flow',type=float,default=.05);p.add_argument('--methods',default='predictive,settle,no_settle');a=p.parse_args()
    jobs=[(an,n,2600000000+ANATOMIES.index(an)*100000+30,lat,a.flow,m) for an in ANATOMIES for n in (1,2,3) for lat in (1,2,3) for m in a.methods.split(',')]
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,candidate='position plus latency times measured velocity',purpose='development-only candidate screening'),indent=2))
    with mp.get_context('spawn').Pool(a.workers) as pool,(a.out/'episodes.jsonl').open('w') as f:
        for i,row in enumerate(pool.imap_unordered(run,jobs),1):
            f.write(json.dumps(row)+'\n');f.flush();print(f'{i}/{len(jobs)}',flush=True)
if __name__=='__main__':main()
