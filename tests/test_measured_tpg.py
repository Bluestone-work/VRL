"""Geometric and causal invariants required before the EXP0053 pilot."""
import ast
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from marl.tracked_sensors import GeometryCrop
from marl.measured_tpg import measured_route,segment_distance,polyline_distance,RollingTPG,polyline_prefix,tangent_frame
from marl.tpg_learning import TPGAgent,NODE_DIM,PAIR_DIM,LOW_DIM,HISTORY,N_LOW,CANDIDATE_DIM
from scripts.tpg_episode import PROTOCOL


P=json.loads(PROTOCOL.read_text())


def crop(segments):
    segments=np.array(segments,float)
    return GeometryCrop(segments,np.full((len(segments),2),.5))


def test_folded_3d_route_is_not_straight_line_distance():
    c=crop([[[0,0,0],[0,0,3]],[[0,0,3],[1,0,3]],[[1,0,3],[1,0,0]]])
    route=measured_route(c,[0,0,0],[1,0,0],P['route'])
    assert route.remaining_mm==pytest.approx(7.)
    assert route.lower_bound_mm==pytest.approx(1.)


def test_nearby_disconnected_tubes_never_gain_shortcut_edges():
    c=crop([[[0,0,0],[4,0,0]],[[0,.1,0],[4,.1,0]]])
    route=measured_route(c,[1,0,0],[3,.1,0],P['route'])
    assert route.remaining_mm is None
    assert np.allclose(route.points[:,1],0.)


def test_attachment_to_two_indistinguishable_tubes_is_marked_ambiguous():
    c=crop([[[0,0,0],[4,0,0]],[[0,.02,0],[4,.02,0]]])
    r=measured_route(c,[1,.01,0],[3,0,0],P['route'])
    assert r.ambiguous and r.remaining_mm is None


def test_two_inserted_points_on_same_edge_have_direct_distance():
    r=measured_route(crop([[[0,0,0],[1,0,0]]]),[.2,0,0],[.8,0,0],P['route'])
    assert r.remaining_mm==pytest.approx(.6)


def test_unobserved_goal_does_not_return_fake_geodesic():
    r=measured_route(crop([[[0,0,0],[1,0,0]]]),[0,0,0],[10,3,2],P['route'])
    assert r.remaining_mm is None
    assert r.visible_length_mm==pytest.approx(1.)


@pytest.mark.parametrize('a,b,c,d,want',[
    ([0,0,0],[2,0,0],[1,-1,3],[1,1,3],3.),
    ([0,0,0],[2,0,0],[1,-1,0],[1,1,0],0.),
    ([0,0,0],[0,0,0],[1,0,0],[2,0,0],1.),
    ([0,0,0],[1,0,0],[0,2,0],[1,2,0],2.)])
def test_euclidean_segment_distance_in_three_dimensions(a,b,c,d,want):
    assert segment_distance(a,b,c,d)==pytest.approx(want)
    assert segment_distance(c,d,a,b)==pytest.approx(want)


def test_conflict_uses_spatial_separation_even_for_disjoint_routes():
    t=RollingTPG(2,P['tpg'])
    routes=[np.array([[0,0,0],[3,0,0]]),np.array([[0,0,1],[3,0,1]])]
    assert t.inspect(routes,np.ones(2,bool),True)
    t.schedule((1,0))
    assert t.edges[1,0] and not t.ready(np.ones(2,bool))[0]


def test_reservations_release_only_after_fresh_separation_confirmations():
    t=RollingTPG(2,P['tpg']);active=np.ones(2,bool)
    near=[np.array([[0,0,0]]),np.array([[1,0,0]])]
    far=[np.array([[0,0,0]]),np.array([[5,0,0]])]
    t.inspect(near,active,True);t.schedule((0,1))
    for _ in range(10):t.inspect(far,active,False)
    assert t.edges[0,1]
    for _ in range(10):t.inspect(far,np.array([False,True]),True)
    assert t.edges[0,1] and not t.ready(np.array([False,True]))[1]
    for _ in range(3):t.inspect(far,active,True)
    assert not t.edges.any()


def test_new_camera_frames_with_stale_individual_tracks_do_not_release():
    t=RollingTPG(2,P['tpg']);active=np.ones(2,bool)
    near=[np.array([[0,0,0]]),np.array([[1,0,0]])]
    far=[np.array([[0,0,0]]),np.array([[5,0,0]])]
    t.inspect(near,active,[True,True]);t.schedule((0,1))
    for _ in range(10):t.inspect(far,active,[True,False])
    assert t.edges[0,1] and t.clear_counts[0,1]==0
    t.inspect(far,active,[True,True]);t.inspect(far,active,[True,True])
    t.inspect(far,active,[False,True])
    assert t.clear_counts[0,1]==0
    for _ in range(3):t.inspect(far,active,[True,True])
    assert not t.edges.any()


def test_reversed_priority_cannot_create_cycle_or_preempt_active_reservations():
    t=RollingTPG(3,P['tpg']);t.conflicts[:]=True;np.fill_diagonal(t.conflicts,False)
    t.schedule((0,1,2));edges=t.edges.copy();order=t.schedule((2,1,0))
    assert order==(0,1,2)
    np.testing.assert_array_equal(t.edges,edges)


def test_unchanged_geometry_does_not_force_one_second_rescheduling():
    t=RollingTPG(2,P['tpg']);routes=[np.array([[0,0,0]]),np.array([[1,0,0]])]
    t.inspect(routes,np.ones(2,bool),True);t.schedule((0,1))
    for _ in range(100):assert not t.inspect(routes,np.ones(2,bool),True)
    assert t.calls==1


def test_tangent_frame_is_orthonormal_for_vertical_and_oblique_paths():
    for tangent in ([0,0,1],[1,1,1],[1,0,0]):
        frame=tangent_frame(tangent)
        np.testing.assert_allclose(frame@frame.T,np.eye(3),atol=1e-8)


def test_path_prefix_follows_bend_instead_of_chord():
    prefix=polyline_prefix(np.array([[0,0,0],[0,0,1],[1,0,1]]),1.5)
    np.testing.assert_allclose(prefix[-1],[.5,0,1])


def test_models_support_n1_n3_and_measured_dynamics_gradient():
    for n in (1,3):
        net=TPGAgent(16)
        nodes=torch.zeros(2,n,NODE_DIM);nodes[...,-1]=1
        pairs=torch.zeros(2,n,n,PAIR_DIM)
        dist,value=net.high(nodes,pairs)
        assert dist.logits.shape[-1]==(1 if n==1 else 6)
        history=torch.randn(2,n,HISTORY,LOW_DIM)
        candidates=torch.zeros(2,n,N_LOW,CANDIDATE_DIM);candidates[:,:,0,8]=1
        valid=torch.ones(2,n,N_LOW,dtype=torch.bool);active=torch.ones(2,n,dtype=torch.bool)
        d,v=net.low(history,candidates,valid,active)
        assert torch.all(d.logits.argmax(-1)==0)
        loss=net.low.predict_measured_velocity(history,torch.zeros(2,n,3)).square().mean()
        loss.backward()
        assert net.low.temporal.weight_ih_l0.grad.abs().sum()>0


def test_high_policy_masks_orders_that_preempt_existing_dependencies():
    from itertools import permutations
    net=TPGAgent(16)
    nodes=torch.zeros(1,3,NODE_DIM);nodes[...,-1]=1
    pairs=torch.zeros(1,3,3,PAIR_DIM);pairs[0,2,0,4]=1;pairs[0,0,2,4]=-1
    d,_=net.high(nodes,pairs)
    for k,order in enumerate(permutations(range(3))):
        if order.index(0)<order.index(2):assert d.probs[0,k]==0


def test_policy_modules_have_no_environment_or_true_route_access():
    for file in ('marl/measured_tpg.py','marl/tpg_learning.py'):
        tree=ast.parse(Path(file).read_text())
        forbidden={'env','transport','solution','masses','station_ids','edges_true','flow_model'}
        assert not [node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute) and node.attr in forbidden]


def test_short_physical_untrained_equivalence_and_confirmation_guard():
    from scripts.run_tpg_learning import make_episode,high_forward,high_state,low_forward,low_state
    with pytest.raises(ValueError):make_episode(P['confirmation_scene_base'],duration=.1)
    a=make_episode(P['preflight_scene_base']+100,duration=2.)
    b=make_episode(P['preflight_scene_base']+100,duration=2.)
    net=TPGAgent(P['hidden_dim'])
    try:
        while not a.done:
            if a.needs_decision:a.choose_priority(a.conventional_priority())
            a.step_control(a.conventional_low())
        while not b.done:
            if b.needs_decision:
                with torch.no_grad():d,_=high_forward(net,high_state(b))
                action=int(d.logits.argmax(-1)[0])
                assert action==b.conventional_priority()
                b.choose_priority(action)
            with torch.no_grad():d,_=low_forward(net,low_state(b))
            action=d.logits.argmax(-1)[0].numpy()
            np.testing.assert_array_equal(action,b.conventional_low())
            b.step_control(action)
        x,y=a.result('tpg'),b.result('untrained')
        assert x['actual_initial_snapshot_hash']==y['actual_initial_snapshot_hash']
        assert x['final_state_hash']==y['final_state_hash']
        assert x['graph_global_truth_input'] is False
    finally:a.close();b.close()
