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
for seed in map(int,sys.argv[3].split(',')):
    env=MCAPhysicalEnv(cfg); obs,_=reset_with_valid_particles(env,seed)
    cos=[];flowr=[];handoff=None
    while True:
        live=np.flatnonzero(env.masses>0)
        w=witness_action(env,np.argmin(env._target_distances(),axis=1).copy()) if len(live) else None
        if env.elapsed_s<float(sys.argv[4]):
            _,_,_,ex,_,_=physical_policy_action(agent,env,obs,deterministic=True)
            if len(live)==1 and w is not None:
                a=np.asarray(ex);act=env.active[:5]
                c=(a*w).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(w,axis=1),1e-9)
                cos.extend(c[act].tolist())
                fl=env.transport.velocity_mm_s(env.positions_mm[:5],env.edges[:5],env.solution)
                flowr.extend((np.linalg.norm(fl,axis=1)/cfg.robot_speed_mm_s)[act].tolist())
        else:
            if handoff is None: handoff=dict(t=env.elapsed_s,remaining=live.tolist())
            ex=w
        obs,r,term,trunc,info=env.step(ex)
        if term or trunc: break
    print(json.dumps(dict(seed=seed,success=bool(info['success']),t=round(env.elapsed_s,1),handoff=handoff,
        cos_policy_vs_route_lastphase=dict(mean=round(float(np.mean(cos)),3) if cos else None,frac_neg=round(float(np.mean(np.array(cos)<0)),3) if cos else None,n=len(cos)),
        flow_over_speed=dict(mean=round(float(np.mean(flowr)),3) if flowr else None,p95=round(float(np.percentile(flowr,95)),3) if flowr else None))),flush=True)
