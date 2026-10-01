import json, sys, numpy as np, torch
from pathlib import Path
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent
from scripts.train_mca_compiled import reset_with_valid_particles, physical_policy_action
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
            n=env.num_robots;j=live[0];act=env.active[:n]
            axis,_,_,_=env.transport.coordinates(env.positions_mm[:n],env.edges[:n],env.solution)
            route=env._route_directions(axis)[:,j]
            eu=env.clot_positions_mm[j]-env.positions_mm[:n];eu/=np.linalg.norm(eu,axis=1,keepdims=True)
            a=np.asarray(ex);an=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)
            geo=env._target_distances()[:,j]
            for i in np.flatnonzero(act):
                R.append(((an[i]*route[i]).sum(),(an[i]*eu[i]).sum(),(route[i]*eu[i]).sum(),geo[i]))
        obs,r,term,trunc,info=env.step(ex)
        if term or trunc: break
R=np.array(R);far=R[:,3]>0.7
dis=far&(R[:,2]<0.5)
print(json.dumps(dict(n=len(R),cos_action_route=round(R[far,0].mean(),3),cos_action_euclid=round(R[far,1].mean(),3),
  route_euclid_disagree_frac=round(dis.sum()/far.sum(),3),
  when_disagree_cos_route=round(R[dis,0].mean(),3) if dis.any() else None,when_disagree_cos_euclid=round(R[dis,1].mean(),3) if dis.any() else None)))
