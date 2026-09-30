from dataclasses import replace
import torch
import pytest
from environments.mca_physical_env import MCAPhysicalEnv,DynamicsConfig
from marl.mca_physical_policy import make_physical_agent,initialize_expanded_obstacle_policy


def agents():
    config=replace(DynamicsConfig(),contact_model='localized_point')
    old=make_physical_agent(MCAPhysicalEnv(config),hidden_dim=32)
    new=make_physical_agent(MCAPhysicalEnv(replace(config,obstacle_observation='predictive_four')),hidden_dim=32)
    checkpoint=dict(meta=old.meta,actor=old.actor.state_dict(),critic=old.critic.state_dict())
    return old,new,checkpoint


def test_expansion_preserves_original_weights_and_initially_ignores_new_inputs():
    old,new,checkpoint=agents();keys=initialize_expanded_obstacle_policy(new,checkpoint)
    assert len(keys)==2 and new.meta['observation_schema']=='mca_point_obstacles_76_v4'
    for name,net in (('actor',new.actor),('critic',new.critic)):
        for key,value in net.state_dict().items():
            if f'{name}.{key}' in keys:
                torch.testing.assert_close(value[:,:36],checkpoint[name][key])
                assert torch.count_nonzero(value[:,36:])==0
            else:torch.testing.assert_close(value,checkpoint[name][key])
    x=torch.randn(5,36);extra=torch.randn(5,40)
    for a,b in ((old.actor.encoder.input_proj,new.actor.encoder.input_proj),
                (old.critic.inner.obs_encoder.input_proj,new.critic.inner.obs_encoder.input_proj)):
        torch.testing.assert_close(a(x),b(torch.cat((x,extra),dim=1)))


def test_wrong_schema_or_architecture_cannot_be_silently_expanded():
    _,new,checkpoint=agents();checkpoint['meta']['observation_schema']='mca_surface_36_v2'
    with pytest.raises(ValueError,match='explicit'):initialize_expanded_obstacle_policy(new,checkpoint)
    _,new,checkpoint=agents();checkpoint['meta']['hidden_dim']=128
    with pytest.raises(ValueError,match='hidden_dim'):initialize_expanded_obstacle_policy(new,checkpoint)
