"""PyBullet GUI of actual MCA checkpoint rollout (visualization only).

Render physical substep samples from the compiled environment. No Bullet
physics step changes the training dynamics. Reload the latest policy between
replays; slow playback is explicitly labelled and independent of training.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pybullet as p
import torch
from PIL import Image
from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_physical import reset_with_valid_particles, atomic_json

ROOT=Path(__file__).resolve().parents[1]


def collect(checkpoint,seed):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=payload['training_state']
    saved_env=state['envs'][0] if 'envs' in state else state['env']
    env=CompiledMCAPhysicalEnv(saved_env.config)
    env.trace_dt_s=.05 if env.config.episode_duration_s>10 else .001
    obs,_=reset_with_valid_particles(env,seed)
    agent=make_physical_agent(env,seed=42,hidden_dim=128,device='cpu')
    if payload['meta'].get('observation_schema')!=env.observation_schema or payload['meta'].get('action_semantics')!='direct_local_frenet':
        raise ValueError('Checkpoint is not a physical direct-local policy')
    agent.load(checkpoint,load_optimizers=False);agent.actor.eval();agent.critic.eval()
    frames=[(0.,env.positions_mm.copy(),env.active.copy(),env.masses.copy())]
    initial_radius=env.solution['radius_mm'].copy()
    while True:
        _,_,_,action,_,_=physical_policy_action(agent,env,obs,deterministic=True)
        obs,_,term,trunc,info=env.step(action)
        trace=env.last_trace
        for i in range(len(trace['time_s'])):
            frames.append((float(trace['time_s'][i]),trace['positions_mm'][i].copy(),trace['active'][i].copy(),trace['masses'][i].copy()))
        if term or trunc: break
    return env,initial_radius,frames,payload['training_state']['transitions'],info


def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True)
    a.add_argument('--headless',action='store_true');a.add_argument('--once',action='store_true')
    a.add_argument('--slowdown',type=float,default=60.);a.add_argument('--seed',type=int,default=900000000)
    args=a.parse_args();torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint);checkpoint_anchor=checkpoint
    client=p.connect(p.DIRECT if args.headless else p.GUI,options='--background_color_red=0.035 --background_color_green=0.055 --background_color_blue=0.09')
    if client<0: raise RuntimeError('PyBullet GUI connection failed')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0);p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
    text_id=-1;camera_ready=False;last_mtime=None
    robot_colors=[[.2,.8,1,1],[.2,1,.6,1],[1,.8,.15,1],[.8,.4,1,1],[1,.45,.2,1]]
    try:
        while p.isConnected():
            pointer=checkpoint_anchor.parent.parent/'active_run.json'
            if pointer.exists():
                checkpoint=Path(json.loads(pointer.read_text())['path'])/checkpoint_anchor.parent.name/checkpoint_anchor.name
            current=checkpoint.stat().st_mtime_ns
            if current!=last_mtime:
                env,radii,frames,steps,info=collect(checkpoint,args.seed);last_mtime=current
                np.savez_compressed(out/'replay.npz',time_s=[f[0] for f in frames],positions_mm=[f[1] for f in frames],active=[f[2] for f in frames],masses=[f[3] for f in frames])
                atomic_json(out/'replay.json',dict(checkpoint=str(checkpoint),checkpoint_steps=steps,episode_seed=args.seed,
                    frame_count=len(frames),physical_duration_s=frames[-1][0],playback_slowdown=args.slowdown,
                    visualization_only=True,robot_radius_mm=env.config.robot_radius_mm,
                    particle_radius_mm=env.config.particle_radius_mm,clot_markers='red location markers; NOT resolved clot surface geometry',
                    tracer_marker_scale=3,robot_halo_scale=4,success=info['success'],removed_mass=float(env.initial_mass.sum()-env.masses.sum())))
                p.resetSimulation();p.setGravity(0,0,0)
                for e,(u,v) in enumerate(env.transport.ends):
                    start=env.transport.points[u];end=env.transport.points[v];delta=end-start;length=np.linalg.norm(delta);direction=delta/length
                    # Translucent cylinder shows local lumen; centerline is a
                    # bright guide. Underlying geometry remains piecewise tubes.
                    z=np.array([0.,0.,1.]);cross=np.cross(z,direction);dot=float(np.dot(z,direction))
                    q=np.r_[cross,1+dot];q=q/np.linalg.norm(q) if dot>-.999999 else np.array([1.,0.,0.,0.])
                    radius=float((radii[u]+radii[v])/2)
                    shape=p.createVisualShape(p.GEOM_CYLINDER,radius=radius,length=float(length),rgbaColor=[.2,.55,.72,.12])
                    p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=((start+end)/2).tolist(),baseOrientation=q.tolist())
                    p.addUserDebugLine(start,end,[.18,.45,.6],1)
                bodies=[];halos=[]
                for i in range(len(env.positions_mm)):
                    color=robot_colors[i] if i<env.num_robots else [.95,.95,1,1]
                    radius=env.body_radius[i]*(1 if i<env.num_robots else 3)
                    shape=p.createVisualShape(p.GEOM_SPHERE,radius=float(radius),rgbaColor=color)
                    bodies.append(p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=frames[0][1][i]))
                    if i<env.num_robots:
                        halo=p.createVisualShape(p.GEOM_SPHERE,radius=float(radius*4),rgbaColor=color[:3]+[.25])
                        halos.append(p.createMultiBody(baseMass=0,baseVisualShapeIndex=halo,basePosition=frames[0][1][i]))
                clots=[]
                for i,pos in enumerate(env.clot_positions_mm):
                    shape=p.createVisualShape(p.GEOM_SPHERE,radius=.35,rgbaColor=[1,.12,.18,.85])
                    clots.append(p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=pos))
                    p.addUserDebugText(f'C{i+1}',pos+[0,0,.6],[1,.35,.35],textSize=1.2)
                points=env.transport.points;center=(points.min(0)+points.max(0))/2;span=float(np.linalg.norm(np.ptp(points,axis=0)))
                if not camera_ready:
                    p.resetDebugVisualizerCamera(span*1.05,35,-55,center);camera_ready=True
                view=p.computeViewMatrixFromYawPitchRoll(center,span*1.05,35,-55,0,2)
                projection=p.computeProjectionMatrixFOV(48,16/10,.01,span*5)
                text_id=-1
            last_positions=None;trail=[]
            for k,(t,pos,active,mass) in enumerate(frames):
                if not p.isConnected(): break
                for i,body in enumerate(bodies):
                    p.resetBasePositionAndOrientation(body,pos[i],[0,0,0,1])
                    color=robot_colors[i] if i<env.num_robots else [.95,.95,1,1]
                    p.changeVisualShape(body,-1,rgbaColor=color if active[i] else [.45,.45,.45,.35])
                    if i<env.num_robots:
                        p.resetBasePositionAndOrientation(halos[i],pos[i],[0,0,0,1])
                        p.changeVisualShape(halos[i],-1,rgbaColor=color[:3]+[.25 if active[i] else .03])
                        if last_positions is not None:
                            trail.append(p.addUserDebugLine(last_positions[i],pos[i],color[:3],2))
                for i,body in enumerate(clots):
                    p.changeVisualShape(body,-1,rgbaColor=[1,.12,.18,max(.05,float(mass[i])*.85)])
                label=f'EXP23  |  checkpoint {steps:,} steps  |  t={t:.3f}s  |  robots {int(active[:env.num_robots].sum())}/{env.num_robots}  |  removed {(4-mass.sum())/4*100:.5f}%  |  replay {args.slowdown:g}x slower'
                text_id=p.addUserDebugText(label,center+[0,0,span*.4],[1,1,1],textSize=1.1,replaceItemUniqueId=text_id)
                last_positions=pos
                if k==min(30,len(frames)-1):
                    image=p.getCameraImage(1280,800,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER)
                    Image.fromarray(np.asarray(image[2],np.uint8).reshape(800,1280,4)).save(out/'scene.png')
                if not args.headless and k+1<len(frames):
                    time.sleep(min(.2,max(.005,(frames[k+1][0]-t)*args.slowdown)))
            if args.once or args.headless: break
            time.sleep(1)
            for line in trail: p.removeUserDebugItem(line)
    finally:
        if p.isConnected(): p.disconnect()


if __name__=='__main__': main()
