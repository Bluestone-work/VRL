import numpy as np
import torch
from marl.multicluster import MultiClusterConfig, ClusterObservation
from marl.local_maneuver_learning import FEATURE_DIM
from marl.temporal_candidate_learning import HISTORY, CANDIDATE_DIM, INPUT_DIM
from marl.memory_prior_learning import MemoryRecommendationLibrary, MemoryPriorActorCritic
from scripts.tracked_learning_episode import TrackingManeuverLibrary


def observation():
    n=2
    nav=np.zeros((n,111),np.float32)
    nav[:,11]=1.;nav[:,12]=.1;nav[:,13]=1.;nav[:,16]=1.
    nav[:,35]=1.;nav[:,36]=1.;nav[:,39]=1.;nav[:,42]=1.
    nav[:,45]=1.;nav[:,46]=-1.;nav[:,49]=-1.;nav[:,52]=1.
    nav[:,9]=1.
    return ClusterObservation(nav,np.zeros((n,4),int),np.zeros((n,n,3)),
        np.zeros((n,n,3)),np.zeros((n,n),bool),np.ones(n,bool))


def test_cached_recommendation_preserves_memory_timing_and_output():
    cfg=MultiClusterConfig(clusters=2)
    old,new=TrackingManeuverLibrary(cfg),MemoryRecommendationLibrary(cfg)
    p=observation()
    used=set()
    for _ in range(100):
        _,oa,ov,od=old.prepare(p);_,na,nv,nd=new.prepare(p)
        oc=old.conventional_choice(p,ov,od,memory=True)
        nc=new.conventional_choice(p,nv,nd,memory=True)
        assert np.array_equal(oc,nc) and np.array_equal(oa,na)
        assert np.array_equal(nc,new.conventional_choice(p,nv,nd,memory=True))
        old.commit(oc,od);new.commit(nc,nd);used.update(oc)
    assert len(used)>1, 'Exercise both ordinary navigation and escape phases'


def test_untrained_model_follows_the_given_measured_memory_recommendation():
    x=torch.zeros(3,INPUT_DIM)
    c=x[:,HISTORY*FEATURE_DIM:].reshape(3,10,CANDIDATE_DIM)
    recommended=torch.tensor([0,8,5]);c[torch.arange(3),recommended,5]=1.
    model=MemoryPriorActorCritic()
    d=model.distribution(x,torch.ones(3,10,dtype=torch.bool))
    assert torch.equal(d.logits.argmax(-1),recommended)
    assert torch.all(d.probs[torch.arange(3),recommended]>.94)
