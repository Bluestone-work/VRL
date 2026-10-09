import copy
import pytest
from scripts.run_vascular_adaptive_research import paired,next_recipe,seed_scene_interval


def row(i,safe):
    return dict(anatomy='a',seed=i,clusters=3,sensing='noise',scenario_hash=f's{i}',initial_state_hash=f'i{i}',
                cluster_safe_success=safe,removal=1.,removal_auc=.8,wall_contact_s=.1,particle_events=0,
                spacing=dict(spacing_violation_pair_s=0.))


def test_hash_mismatch_and_missing_scenes_are_rejected():
    b=[row(0,False),row(1,True)];c=copy.deepcopy(b);c[0]['initial_state_hash']='different'
    with pytest.raises(ValueError,match='hashes'):paired(c,b)
    with pytest.raises(ValueError,match='Incomplete'):paired(b[:1],b)


def test_safety_gain_cannot_hide_wall_or_removal_regression():
    b=[row(0,False),row(1,False)];c=[row(0,True),row(1,True)]
    assert paired(c,b)['promising']
    c[0]['wall_contact_s']=10.
    assert not paired(c,b)['promising']


def test_negative_round_changes_next_recipe_and_seed_ci_keeps_pairing():
    recipe=dict(lr=1e-4,gamma=.995,entropy=.01,world_weight=.1,model_warmup=8,wall_weight=12.,
                particle_weight=12.,updates=120,parent_kind='final')
    b=[row(0,True),row(1,True)];c=copy.deepcopy(b);c[0]['wall_contact_s']=20.
    result=paired(c,b);new,reasons=next_recipe(recipe,result,None,0)
    assert new['wall_weight']>recipe['wall_weight'] and reasons
    interval=seed_scene_interval([b,b,b],b)
    assert interval['hierarchical_ci95']==[0.,0.] and interval['scenes_per_seed']==2
