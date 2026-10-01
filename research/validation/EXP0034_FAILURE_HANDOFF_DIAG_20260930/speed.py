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
    A={1:[],2:[],3:[],4:[]};V={1:[],2:[],3:[],4:[]};W={1:[],2:[],3:[],4:[]}
    while True:
        k=int((env.masses>0).sum())
        _,_,_,ex,_,_=physical_policy_action(agent,env,obs,deterministic=True)
        before=env._target_distances().min(axis=1)
        obs,r,term,trunc,info=env.step(ex)
        a=env.active[:5]
        if k in A and (env.masses>0).sum()==k:
            A[k].extend(np.linalg.norm(np.asarray(ex),axis=1)[a]); V[k].extend(np.linalg.norm(env.velocity_mm_s,axis=1)[a])
            W[k].extend(((before-env._target_distances().min(axis=1))/cfg.control_dt_s)[a])
        if term or trunc: break
    print(seed,info['success'],{k:(len(A[k])//5,round(float(np.mean(A[k])),3),round(float(np.mean(V[k])),3),round(float(np.mean(W[k])),3)) for k in A if A[k]},flush=True)
