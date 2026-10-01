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
mode=sys.argv[4]
for seed in map(int,sys.argv[3].split(',')):
    env=CompiledMCAPhysicalEnv(cfg); obs,_=reset_with_valid_particles(env,seed)
    while True:
        _,_,_,ex,_,_=physical_policy_action(agent,env,obs,deterministic=True)
        ex=np.asarray(ex,float)
        if mode=='unit': ex=ex/np.maximum(np.linalg.norm(ex,axis=1,keepdims=True),1e-9)
        obs,r,term,trunc,info=env.step(ex)
        if term or trunc: break
    print(json.dumps(dict(seed=seed,mode=mode,success=bool(info['success']),cf=bool(info['collision_free_success']),t=round(env.elapsed_s,1),ev=int(info['episode_particle_collision_events']))),flush=True)
