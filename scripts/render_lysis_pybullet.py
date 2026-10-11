"""PyBullet-style snapshot renderer for the current lysis environment.

Visualization only: physics and training never call this module.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pybullet as p

def quat(z, d):
    z=np.asarray(z,float); d=np.asarray(d,float); d=d/max(np.linalg.norm(d),1e-12)
    c=np.cross(z,d); w=1+float(z@d)
    q=np.r_[c,w]
    return (q/max(np.linalg.norm(q),1e-12)).tolist() if w>1e-8 else [1,0,0,0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--anatomy',default='mca_m1_lvo'); ap.add_argument('--clusters',type=int,default=3); ap.add_argument('--seed',type=int,default=2710100); ap.add_argument('--steps',default='40,160,320')
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    from scripts.benchmark_lysis import LysisEpisode, WallGuard
    ep=LysisEpisode(a.clusters,a.anatomy,a.seed,sensing='image',tpg=True)
    client=p.connect(p.DIRECT); p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=client); p.setGravity(0,0,0,physicsClientId=client)
    env=ep.env; pts=env.transport.points.astype(float); rad=np.asarray(env.flow_model.healthy_radius_mm,float)
    for u,v in env.transport.ends:
        s,e=pts[u],pts[v]; d=e-s; L=np.linalg.norm(d); shape=p.createVisualShape(p.GEOM_CYLINDER,radius=float((rad[u]+rad[v])*.5),length=float(L),rgbaColor=[.2,.55,.72,.18],physicsClientId=client)
        p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=((s+e)/2).tolist(),baseOrientation=quat([0,0,1],d),physicsClientId=client)
    bodies=[]; halos=[]
    for i in range(a.clusters):
        col=[[.2,.8,1,1],[.2,1,.6,1],[1,.8,.15,1]][i%3]
        sh=p.createVisualShape(p.GEOM_SPHERE,radius=float(env.body_radius[i]*5),rgbaColor=col,physicsClientId=client)
        bodies.append(p.createMultiBody(baseMass=0,baseVisualShapeIndex=sh,basePosition=env.positions_mm[i].tolist(),physicsClientId=client))
        hs=p.createVisualShape(p.GEOM_SPHERE,radius=float(env.body_radius[i]*9),rgbaColor=col[:3]+[.28],physicsClientId=client)
        halos.append(p.createMultiBody(baseMass=0,baseVisualShapeIndex=hs,basePosition=env.positions_mm[i].tolist(),physicsClientId=client))
    clot_b=[]
    for q in env.clot_positions_mm:
        sh=p.createVisualShape(p.GEOM_SPHERE,radius=.65,rgbaColor=[1,.05,.08,1],physicsClientId=client); clot_b.append(p.createMultiBody(baseMass=0,baseVisualShapeIndex=sh,basePosition=q.tolist(),physicsClientId=client))
    want={int(x) for x in a.steps.split(',') if x}; fig=p.computeViewMatrixFromYawPitchRoll(((pts.min(0)+pts.max(0))/2).tolist(),float(np.linalg.norm(np.ptp(pts,axis=0))*1.05),35,-55,0,2); proj=p.computeProjectionMatrixFOV(48,1.6,.01,float(np.linalg.norm(np.ptp(pts,axis=0))*5))
    ctl=WallGuard(ep)
    for k in range(1,max(want)+1):
        est=ep.observe(); tgt=ep.plan_targets_now(est); rule=ep.ctl.act(tgt,est); hold=ep.hold(est); local=ctl(ep,est,tgt,rule,hold); done,_=ep.step(est,local,hold)
        if k in want:
            for i,b in enumerate(bodies):
                p.resetBasePositionAndOrientation(b,env.positions_mm[i].tolist(),[0,0,0,1],physicsClientId=client)
                p.resetBasePositionAndOrientation(halos[i],env.positions_mm[i].tolist(),[0,0,0,1],physicsClientId=client)
            im=p.getCameraImage(1280,800,viewMatrix=fig,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER,physicsClientId=client)[2]
            Image.fromarray(np.asarray(im,dtype=np.uint8).reshape(800,1280,4)).save(out/f'pybullet_lysis_{a.anatomy}_N{a.clusters}_step{k:04d}.png')
        if done: break
    p.disconnect(client)
    print('saved',out)
if __name__=='__main__': main()
