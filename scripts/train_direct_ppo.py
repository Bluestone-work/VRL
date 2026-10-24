"""Continuous Gaussian PPO with the same 3-D direct route-frame action as SAC/TD3."""
import argparse,json,os,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from marl.image_flow_sac import ControlEpisode,ImageActor,ImageFlowController,observation_tensors
ROOT=Path(__file__).resolve().parents[1]

class ValueNet(nn.Module):
    def __init__(self): super().__init__(); from marl.image_flow_sac import Encoder; self.encoder=Encoder(); self.head=nn.Sequential(nn.Linear(160,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,*x): return self.head(self.encoder(*x)).squeeze(-1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--env-steps',type=int,default=1000000);p.add_argument('--seed',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--rollout',type=int,default=256);p.add_argument('--dynamic-obstacle-count',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--device',default='cpu');p.add_argument('--curriculum',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);np.random.seed(a.seed);torch.manual_seed(a.seed)
 actor=ImageActor('direct',zero_init=False).to(a.device);value=ValueNet().to(a.device);ao=torch.optim.Adam(actor.parameters(),lr=a.lr);vo=torch.optim.Adam(value.parameters(),lr=a.lr);split=json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_holdout_v1']['train'];cfg=dict(algorithm='PPO direct continuous 3-D route-frame action from random initialization',action_mode='direct',zero_init=False,dynamic_obstacle_count=a.dynamic_obstacle_count,action_scales=[1.,1.,1.],training_anatomies=split,workers=a.workers,rollout=a.rollout,env_steps=a.env_steps)
 (a.out/'config.json').write_text(json.dumps(cfg,indent=2));steps=0;iteration=0;start=time.time();env_steps_per= a.rollout*a.workers;log=(a.out/'training.jsonl').open('w');episodes=(a.out/'episodes.jsonl').open('w');
 while steps<a.env_steps:
  an=str(np.random.choice(split));n=int(np.random.randint(1,4));flow=float(np.random.choice([.025,.05,.1]));delay=int(np.random.randint(1,3));obstacles=a.dynamic_obstacle_count
  if a.curriculum:
   frac=steps/max(a.env_steps,1)
   if frac<.2:n,flow,delay,obstacles=1,.05,1,0
   elif frac<.5:n,flow,delay,obstacles=1,float(np.random.choice([.025,.05])),1,4
  ep=ControlEpisode(n,an,int(3000000000+steps),flow_inlet_mm_s=flow,sense_cfg=__import__('marl.deployable_sensing',fromlist=['DeployableConfig']).DeployableConfig(latency_steps=delay),dynamic_obstacle_count=obstacles);ctl=ImageFlowController(ep,actor=actor,stochastic=True,config={'station_keeping':False,'obstacle_features':True}); obs=[];acts=[];oldlp=[];rews=[];vals=[];dones=[]
  for _ in range(a.rollout):
   est=ep.observe();tgt=ep.plan_targets_now(est);rule=ep.ctl.act(tgt,est);hold=ep.hold(est);rows=ctl.observe(ep,est,tgt,rule,hold); tens=observation_tensors(rows,a.device)
   with torch.no_grad():act,lp=actor.sample(*tens);v=value(*tens)
   local=ctl.execute(ep,est,act.numpy());done,out=ep.step(est,local,hold);rew=np.full(n,10.*out['removed']-2.*np.sum(out['wall'])-10.*np.sum(out['lost'])-.01*ep.info['step_duration_s']-2.*np.sum(ep.info['particle_contact_s']),np.float32)
   for i in range(n): obs.append({k:rows[i][k] for k in rows[i]});acts.append(act[i].cpu().numpy());oldlp.append(float(lp[i]));vals.append(float(v[i]));rews.append(float(rew[i]));dones.append(done)
   steps+=n
   if done: episodes.write(json.dumps(ep.row('PPO_training',**ctl.summary()))+'\n');episodes.flush();ep.close();break
  if ep.env.active.any():ep.close()
  if len(obs)>=n:
   # Time-major GAE. The previous implementation used one-step reward-V and
   # mixed agents across the recursion, which was not a valid PPO estimator.
   T=len(obs)//n; tens=observation_tensors(obs,a.device);A=torch.tensor(np.asarray(acts),device=a.device);old=torch.tensor(oldlp,device=a.device);V=torch.tensor(vals,device=a.device).reshape(T,n);R=torch.tensor(rews,device=a.device).reshape(T,n);D=torch.tensor(dones,device=a.device,dtype=torch.float32).reshape(T,n)
   with torch.no_grad():
    next_v=torch.zeros_like(V);next_v[:-1]=V[1:];
    delta=R+0.995*(1-D)*next_v-V;gae=torch.zeros(n,device=a.device);adv_t=[]
    for j in range(T-1,-1,-1): gae=delta[j]+0.995*.95*(1-D[j])*gae;adv_t.append(gae)
    adv=torch.stack(adv_t[::-1]);ret=adv+V
   flat_adv=adv.reshape(-1);flat_ret=ret.reshape(-1);flat_old=old;flat_A=A
   flat_adv=(flat_adv-flat_adv.mean())/(flat_adv.std()+1e-6)
   for _ in range(4):
    out=actor.head(actor.encoder(*tens));dist=torch.distributions.Normal(out[:,:3],out[:,3:].clamp(-5,0).exp());z=torch.atanh(flat_A.clamp(-.999,.999));lp=(dist.log_prob(z)-torch.log(1-flat_A.square()+1e-6)).sum(-1);ratio=torch.exp(lp-flat_old);vpred=value(*tens);policy_loss=-torch.minimum(ratio*flat_adv,ratio.clamp(.8,1.2)*flat_adv).mean();value_loss=.5*(vpred-flat_ret).square().mean();entropy=dist.entropy().sum(-1).mean();loss=policy_loss+value_loss-.001*entropy;ao.zero_grad();vo.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);nn.utils.clip_grad_norm_(value.parameters(),5.);ao.step();vo.step()
  iteration+=1;row=dict(iteration=iteration,environment_steps=steps,loss=float(loss),actor_parameter_change_l2=float(sum(x.square().sum() for x in actor.parameters()).sqrt()),minutes=(time.time()-start)/60);log.write(json.dumps(row)+'\n');log.flush();
  if iteration%100==0:torch.save(dict(actor={k:v.cpu() for k,v in actor.state_dict().items()},config=cfg,iterations=iteration,environment_steps=steps),a.out/f'policy_steps{steps}.pt')
 torch.save(dict(actor={k:v.cpu() for k,v in actor.state_dict().items()},config=cfg,iterations=iteration,environment_steps=steps),a.out/'policy.pt');(a.out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(iterations=iteration,environment_steps=steps)));log.close();episodes.close()
if __name__=='__main__':main()
