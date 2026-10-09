import numpy as np
import torch
from marl.vascular_option_rl import OptionPolicy,OBS_DIM,HISTORY
from marl.vascular_predictive_rl import PredictivePolicy,model_loss
from scripts.vascular_predictive_study import PredictiveEpisode,gae
from scripts.vascular_option_study import Episode,DEV_BASE


def test_warmstart_exact_base_actor_and_critic():
    torch.manual_seed(61);base=OptionPolicy(True,True);p=PredictivePolicy(True)
    p.load_parent(dict(config=dict(memory=True,central=True),state=base.state_dict()))
    x=torch.randn(2,3,HISTORY,OBS_DIM);m=torch.ones(2,3)
    a,v=base(x,m);b,w=p(x,m)
    torch.testing.assert_close(a.logits,b.logits);torch.testing.assert_close(v,w)


def test_predictor_can_learn_but_actor_cannot_manipulate_predictor():
    torch.manual_seed(1);p=PredictivePolicy(True)
    x=torch.randn(2,3,HISTORY,OBS_DIM);m=torch.ones(2,3)
    with torch.no_grad():p.world_gate.weight.normal_()
    dist,_,pred=p(x,m,return_model=True);(-dist.log_prob(torch.zeros(2,3,dtype=torch.long)).mean()).backward()
    assert all(v.grad is None for model in p.models for v in model.parameters())
    assert p.world_gate.weight.grad.abs().sum()>0
    p.zero_grad();_,_,pred=p(x,m,return_model=True)
    loss,_=model_loss(pred,torch.zeros(2,3,dtype=torch.long),torch.zeros(2,3,OBS_DIM),
                     torch.ones(2,3),torch.ones(2,3,3),m,bootstrap=False)
    loss.backward();assert all(model[0].weight.grad.abs().sum()>0 for model in p.models)


def test_outcome_instrumentation_preserves_physics_and_metrics():
    a=Episode('mca_m1_lvo',3,DEV_BASE,horizon=1.)
    b=PredictiveEpisode('mca_m1_lvo',3,DEV_BASE,horizon=1.)
    try:
        while not a.done:
            oa,ra,da=a.step(np.zeros(3,int));ob,rb,db=b.step(np.zeros(3,int))
            np.testing.assert_array_equal(oa[0],ob[0]);np.testing.assert_array_equal(ra,rb)
        assert a.metrics()==b.metrics()
    finally:a.env.close();b.env.close()


def test_agent_death_stops_bootstrap_without_stopping_peer():
    r=np.ones((2,1,2));v=np.zeros_like(r);d=np.array([[[1.,0.]],[[0.,0.]]])
    a,_=gae(r,v,d,np.ones((1,2))*2,gamma=1.,lam=1.)
    np.testing.assert_allclose(a[:,0],[[1,4],[3,3]])
