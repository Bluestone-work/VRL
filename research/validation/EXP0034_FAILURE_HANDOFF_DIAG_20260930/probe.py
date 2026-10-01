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
env=MCAPhysicalEnv(cfg); obs,_=reset_with_valid_particles(env,int(sys.argv[3])); k=int(sys.argv[4])
print('lysis',cfg.lysis_mass_per_s,'contact',cfg.contact_distance_mm,cfg.contact_model,'dt',cfg.__dict__.get('dt_s'))
best=9
while True:
    _,_,_,ex,_,_=physical_policy_action(agent,env,obs,deterministic=True)
    obs,r,term,trunc,info=env.step(ex)
    eu=np.linalg.norm(env.positions_mm[:5]-env.clot_positions_mm[k],axis=1); geo=env._target_distances()[:,k] if env.masses[k]>0 else np.zeros(5)
    i=int(np.argmin(eu))
    if eu[i]<0.3 and env.steps%1==0 and eu[i]<best+0.05:
        best=min(best,eu[i]); print(round(env.elapsed_s,2),'robot',i,'eu',round(eu[i],4),'geo',round(float(geo[i]),4),'mass',round(env.masses[k],4),'edge',env.edges[i],'active',env.active[i])
    if term or trunc: break
