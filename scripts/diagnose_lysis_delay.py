"""Step traces for no-settle cruising under matched sensing delay."""
import argparse,json,multiprocessing as mp,traceback
from pathlib import Path
import numpy as np

ANATOMIES=json.loads(Path('configs/evaluation_splits.json').read_text())['anatomy_order']

def one(spec):
    an,n,seed,lat,flow,method=spec
    try:
        from scripts.benchmark_lysis import LysisEpisode,WallGuard,SettleGuard,AdaptiveSettleGuard
        from marl.deployable_sensing import DeployableConfig
        ep=LysisEpisode(n,an,seed,flow_inlet_mm_s=flow,sense_cfg=DeployableConfig(latency_steps=lat))
        ctl=WallGuard(ep); rows=[]; entered={}; left=0; cleared=set(); prev_targets=None
        while True:
            est=ep.observe(); tgt=ep.plan_targets_now(est); rule=ep.ctl.act(tgt,est); hold=ep.hold(est)
            before=ep.prev_local.copy()
            local=ctl(ep,est,tgt,rule,hold)
            if method=='settle': local=SettleGuard(ep)(ep,est,tgt,rule,hold)
            elif method=='adaptive_settle': local=AdaptiveSettleGuard(ep)(ep,est,tgt,rule,hold)
            elif method=='damped':
                # Causal, deployable velocity feedback; no truth or flow labels.
                vlocal=np.einsum('nji,nj->ni',ep.ctl.frames(est),est.vel)
                local=np.clip(local-0.35*vlocal/max(ep.env.config.robot_speed_mm_s,1e-9),-1,1)
                local[~est.active]=0.
            for i,t in enumerate(tgt):
                if t<0:continue
                d=float(np.linalg.norm(ep.env.clot_positions_mm[t]-est.pos[i])); near=d<.3
                key=(i,int(t)); entered.setdefault(key,[])
                if near: entered[key].append(float(ep.env.elapsed_s))
                if key in entered and entered[key] and not near and prev_targets is not None: left+=1
            done,info=ep.step(est,local,hold)
            rows.append(dict(anatomy=an,n=n,seed=seed,latency_steps=lat,flow=flow,method=method,t=float(ep.env.elapsed_s),
                             positions=est.pos.tolist(),estimated_velocity=est.vel.tolist(),targets=np.asarray(tgt).tolist(),
                             carrots=ep.ctl.carrot.tolist(),route_progress=ep.ctl.prog.tolist(),
                             rule_local=rule.tolist(),sent_local=local.tolist(),previous_sent_local=before.tolist(),
                             safety_delta=(local-rule).tolist(),active=est.active.tolist(),clot_alive=ep.env.masses.tolist(),
                             wall_contact=np.asarray(info.get('wall_contact_s',np.zeros(n))).tolist(),
                             lost=int(info.get('lost_robots',0))))
            prev_targets=tgt.copy()
            if done:break
        row=dict(anatomy=an,n=n,seed=seed,latency_steps=lat,flow=flow,method=method,steps=len(rows),
                 termination=ep.info.get('termination_reason') if ep.info else None,
                 first_entry={f'{i}:{t}':min(v) for (i,t),v in entered.items() if v},
                 target_entries={f'{i}:{t}':len(v) for (i,t),v in entered.items()},
                 departures=left,targets_cleared=int(np.sum(ep.env.masses<=0)),lost=int(ep.info.get('lost_robots',0) if ep.info else 0))
        ep.close();return dict(summary=row,trace=rows)
    except Exception:return dict(error=traceback.format_exc(),anatomy=an,n=n,seed=seed,latency_steps=lat)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--flow',type=float,default=.05);p.add_argument('--methods',default='no_settle,settle,adaptive_settle,damped');a=p.parse_args()
    # Development-only seeds, disjoint from the accidentally accessed 2700M offset 11.
    jobs=[(an,n,2600000000+ANATOMIES.index(an)*100000+20,lat,a.flow,m) for an in ANATOMIES for n in (1,2,3) for lat in (1,2,3) for m in a.methods.split(',')]
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,scene_seed_offset=20,flow=a.flow,purpose='delay mechanism development diagnostic'),indent=2))
    with mp.get_context('spawn').Pool(a.workers) as pool,(a.out/'summaries.jsonl').open('w') as sf,(a.out/'traces.jsonl').open('w') as tf:
      for i,x in enumerate(pool.imap_unordered(one,jobs),1):
        sf.write(json.dumps(x.get('summary',x)));sf.write('\n');sf.flush()
        for t in x.get('trace',[]):tf.write(json.dumps(t)+'\n')
        tf.flush();print(f'{i}/{len(jobs)}',flush=True)
    print('complete')
if __name__=='__main__':main()
