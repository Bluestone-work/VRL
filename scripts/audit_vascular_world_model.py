"""One-step prediction audit on fixed behavior trajectories and new dev scenes.

This audits prediction, not control performance. Never used as policy input.
Risk scores are not assumed calibrated, especially under weighted BCE training.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from marl.vascular_predictive_rl import PredictivePolicy
from scripts.vascular_predictive_study import PredictiveEpisode
from scripts.vascular_option_study import ROOT,DEV_BASE,sha


def main():
    os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7});torch.set_num_threads(1)
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--decisions',type=int,default=80);a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu');p=PredictivePolicy(True);p.load_state_dict(ck['state'],strict=True);p.eval()
    order=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_order']
    state_errors=[];persist_errors=[];risks=[];labels=[];reward_errors=[];zero_reward_errors=[];scenes=[]
    for anatomy in ['mca_m1_lvo','ica_terminus_t','renal_artery']:
        seed=DEV_BASE+order.index(anatomy)*10000+200
        ep=PredictiveEpisode(anatomy,3,seed,wall_weight=ck['config']['wall_weight'],particle_weight=ck['config']['particle_weight'])
        rng=np.random.default_rng([61,order.index(anatomy)])
        scenes.append(dict(anatomy=anatomy,seed=seed,initial_state_hash=ep.initial_state_hash))
        for step in range(a.decisions):
            history,active=ep.observation()
            # Fixed behavior independent of checkpoint, model prediction or truth.
            action=np.where(rng.uniform(size=3)<.2,rng.integers(0,9,size=3),0)
            with torch.no_grad():
                _,_,pred=p(torch.from_numpy(history[None]),torch.from_numpy(active[None]),return_model=True)
                pred=pred[:,0].numpy()[:,np.arange(3),action]
            (next_history,next_active),reward,done=ep.step(action)
            valid=(active*next_active)>0;target=next_history[:,-1,:24]
            mean=pred.mean(0)
            state_errors.extend(np.mean((mean[:,:24]-target)**2,axis=-1)[valid].tolist())
            persist_errors.extend(np.mean((history[:,-1,:24]-target)**2,axis=-1)[valid].tolist())
            probs=(1/(1+np.exp(-np.clip(pred[:,:,25:],-40,40)))).mean(0)
            risks.extend(probs[valid].tolist());labels.extend(ep.hazards[valid].tolist())
            reward_errors.extend(((mean[:,24]-reward/10.)**2)[valid].tolist())
            zero_reward_errors.extend(((reward/10.)**2)[valid].tolist())
            if done:break
        ep.env.close()
    risk=np.asarray(risks);label=np.asarray(labels)
    report=dict(checkpoint_sha256=sha(a.checkpoint),protocol='fixed_behavior_80_percent_pursuit_20_percent_uniform',
                inference_uses_simulator_lookahead=False,development_scenes=scenes,agent_transitions=len(labels),
                next_state_mse=float(np.mean(state_errors)),persistence_mse=float(np.mean(persist_errors)),
                reward_mse=float(np.mean(reward_errors)),zero_reward_mse=float(np.mean(zero_reward_errors)),
                hazard_names=['wall','particle','spacing'],hazard_brier=np.mean((risk-label)**2,axis=0).tolist(),
                zero_risk_brier=np.mean(label**2,axis=0).tolist(),positive_counts=label.sum(0).tolist(),
                predicted_risk_mean=risk.mean(0).tolist(),
                interpretation='Development prediction diagnostic only; zero-positive hazards do not demonstrate useful avoidance prediction.')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
