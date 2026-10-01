import json, sys, numpy as np, torch
from pathlib import Path
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent
from scripts.train_mca_compiled import reset_with_valid_particles, physical_policy_action
from scripts.validate_mca_surface_task import witness_action
protocol=json.loads(Path(sys.argv[1]).read_text()); cfg=DynamicsConfig.from_json(Path(protocol['physics_config']))
payload=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
agent=make_physical_agent(CompiledMCAPhysicalEnv(cfg),seed=payload['meta']['training_seed'],hidden_dim=protocol['hidden_dim'],device='cpu',ppo=protocol.get('ppo'))
agent.load(sys.argv[2],load_optimizers=False); agent.actor.eval()
R=[]
for seed in map(int,sys.argv[3].split(',')):
    env=MCAPhysicalEnv(cfg); obs,_=reset_with_valid_particles(env,seed)
    while True:
        live=np.flatnonzero(env.masses>0)
        _,_,_,ex,_,_=physical_policy_action(agent,env,obs,deterministic=True)
        if len(live)==1:
            w=witness_action(env,np.full(5,live[0]))
            a=np.asarray(ex);act=env.active[:5]
            c=(a*w).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(w,axis=1),1e-9)
            n=env.num_robots;P=env.positions_mm[n:][env.active[n:]]
            pd=np.linalg.norm(env.positions_mm[:n,None]-P[None],axis=-1).min(1) if len(P) else np.full(n,9.)
            geo=env._target_distances()[:,live[0]]
            # which robot is assigned vs surplus
            asg=env._assigned_targets()
            # distance to the nearest other robot
            rr=np.linalg.norm(env.positions_mm[:n,None]-env.positions_mm[None,:n],axis=-1)+np.eye(n)*9
            for i in np.flatnonzero(act):
                R.append((c[i],pd[i],geo[i],np.linalg.norm(a[i]),rr[i].min()))
        obs,r,term,trunc,info=env.step(ex)
        if term or trunc: break
R=np.array(R);neg=R[:,0]<0
print(json.dumps(dict(n=len(R),frac_neg=float(neg.mean()),
  particle_dist_neg=float(np.median(R[neg,1])),particle_dist_pos=float(np.median(R[~neg,1])),
  frac_particle_within_0p3_neg=float((R[neg,1]<.3).mean()),frac_particle_within_0p3_pos=float((R[~neg,1]<.3).mean()),
  geo_neg=float(np.median(R[neg,2])),geo_pos=float(np.median(R[~neg,2])),
  norm_neg=float(np.median(R[neg,3])),norm_pos=float(np.median(R[~neg,3])),
  robot_dist_neg=float(np.median(R[neg,4])),robot_dist_pos=float(np.median(R[~neg,4])),
  cos_by_geo_bins={f'{lo}-{hi}':round(float(R[(R[:,2]>=lo)&(R[:,2]<hi),0].mean()),3) if ((R[:,2]>=lo)&(R[:,2]<hi)).any() else None for lo,hi in [(0,1),(1,3),(3,10),(10,30),(30,100)]})))
