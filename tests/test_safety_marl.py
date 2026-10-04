"""Safety-objective follow-up must not silently change observations or physics."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from scripts.run_safety_marl import effective_protocol, base_episode, BaseNetwork
import scripts.run_safety_marl as adapter
import scripts.run_measured_marl as trainer


def test_reward_override_leaves_rule_trajectory_and_input_packets_exactly_equal():
    p=effective_protocol('safety_reward')
    a=base_episode(p['preflight_scene_base']+100,duration=4.)
    b=base_episode(p['preflight_scene_base']+100,duration=4.)
    b.protocol={**b.protocol,'reward':{**b.protocol['reward'],**p['safety_reward_override']}}
    try:
        assert a.manifest['scenario_hash']==b.manifest['scenario_hash']
        while not a.done:
            assert a.needs_decision==b.needs_decision
            if a.needs_decision:
                a.choose_priority(a.conventional_priority());b.choose_priority(b.conventional_priority())
            for key,val in trainer.observation(a).items():np.testing.assert_array_equal(val,trainer.observation(b)[key])
            before_active=a.env.active[:3].copy();before_spacing=a.spacing.pair_violation_s
            ar,adone,_=a.step_control(a.conventional_low())
            br,bdone,_=b.step_control(b.conventional_low())
            assert adone==bdone
            difference=(-52*float(np.sum(a.info['wall_contact_s']))/(3*4)
                -52*float(np.sum(a.info['particle_contact_s']))/(3*4)
                -536*(a.spacing.pair_violation_s-before_spacing)/(3*4)
                -4*float(np.sum(before_active&~a.env.active[:3]))/3)
            assert np.isclose(br-ar,difference,atol=1e-12)
        assert a.result('rule')['final_state_hash']==b.result('rule')['final_state_hash']
    finally:a.close();b.close()


def test_adapter_loads_identical_parent_and_records_fresh_optimizer(monkeypatch,tmp_path):
    torch.manual_seed(21);parent=BaseNetwork('r_mappo',16)
    payload=dict(model=parent.state_dict())
    provenance=dict(parent_checkpoint='test_parent.pt',parent_checkpoint_sha256='test',parent_control_steps=32768,fresh_optimizer=True,exact_resume=False)
    monkeypatch.setattr(adapter,'checked_parent',lambda *args:(payload,provenance))
    for field in ('PROTOCOL','MeasuredMARL','make_episode','source_hashes','atomic_json','update'):
        monkeypatch.setattr(trainer,field,getattr(trainer,field))
    args=SimpleNamespace(mode='train',regime='safety_reward',variant='r_mappo',seed=42,parent_policy=False,checkpoint=None,steps=16384)
    _,record=adapter.configure(args)
    model=trainer.MeasuredMARL('r_mappo',16)
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k])
    trainer.atomic_json(tmp_path/'manifest.json',dict(source='test'))
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['cumulative_requested_controls']==49152
    assert manifest['fresh_optimizer'] is True and manifest['exact_resume'] is False
    assert record['parent_control_steps']==32768


def test_critic_scale_cannot_shrink_actor_clipped_gradient():
    torch.manual_seed(29);a=BaseNetwork('r_mappo',16)
    torch.manual_seed(29);b=BaseNetwork('r_mappo',16)
    for net,scale in ((a,1.),(b,1000.)):
        for param in net.low_actor.parameters():param.grad=torch.ones_like(param)*.01
        for param in net.low_critic.parameters():param.grad=torch.ones_like(param)*scale
        adapter.independent_gradient_clip(net,'low')
    for x,y in zip(a.low_actor.parameters(),b.low_actor.parameters()):
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
