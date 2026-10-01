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
for seed in map(int,sys.argv[3].split(',')):
    env=MCAPhysicalEnv(cfg); obs,_=reset_with_valid_particles(env,seed)
    S=[];O=[];gd=[];edges=[]
    while True:
        _,_,_,ex,_,_=physical_policy_action(agent,env,obs,deterministic=True)
        one=(env.masses>0).sum()==1
        obs,r,term,trunc,info=env.step(ex)
        if one and (env.masses>0).sum()==1:
            a=env.active[:5]; S.append(info['shaping_rewards'][a].mean()); O.append((info['agent_rewards']-info['shaping_rewards'])[a].mean())
            gd.append(env._target_distances()[a].min(axis=1).mean()); edges.append(tuple(env.edges[:5]))
        if term or trunc: break
    if S:
        S,O=np.array(S),np.array(O)
        print(json.dumps(dict(seed=seed,success=bool(info['success']),steps=len(S),shaping_sum=round(S.sum(),3),other_sum=round(O.sum(),3),
           other_p5=round(float(np.percentile(O,5)),4),mean_geo_start=round(gd[0],2),mean_geo_end=round(gd[-1],2),
           distinct_edges_last300=len(set(edges[-300:])))),flush=True)
