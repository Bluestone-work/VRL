"""Scene graph has no routed/teacher inputs; teacher replay is exact; student handles variable graphs."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.graph_transformer_student import GraphTransformerStudent, collate_scenes, student_local_action
from marl.scene_graph import extract_scene
from marl.teacher import TEACHER_CONFIG, execute_local, teacher_config, teacher_label
from scripts.collect_teacher_dataset import student_env_config
from scripts.train_mca_compiled import reset_with_valid_particles

SEED = 940000000


def test_scene_graph_never_reads_routes_or_allocation(monkeypatch):
    env = CompiledMCAPhysicalEnv(student_env_config('mca_m1_lvo', 5))
    reset_with_valid_particles(env, SEED)
    def forbidden(*a, **k):
        raise AssertionError('scene graph touched teacher information')
    for name in ('_route_directions', '_assigned_targets', '_target_distances', '_solve_assignment', '_observation',
                 '_write_target_block', '_apply_action_prior'):
        monkeypatch.setattr(env, name, forbidden)
    monkeypatch.setattr(env.tree, 'route_to', forbidden, raising=False)
    monkeypatch.setattr(env.tree, 'lookahead', forbidden, raising=False)
    env.routes = env._route_next_hop = None
    scene = extract_scene(env)
    assert scene['robot'].shape[0] == 5 and scene['vessel'].shape[0] == len(scene['kept'])


def test_executing_teacher_labels_reproduces_the_teacher_exactly():
    teacher_env = CompiledMCAPhysicalEnv(replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy='coronary_rca'))
    student_env = CompiledMCAPhysicalEnv(student_env_config('coronary_rca', 5))
    tcfg = teacher_config(student_env.config)
    reset_with_valid_particles(teacher_env, SEED); reset_with_valid_particles(student_env, SEED)
    for _ in range(80):
        local, stop, _ = teacher_label(student_env, tcfg)
        act = np.where(stop[:, None], 0., local)*student_env.active[:5, None]
        student_env.step(execute_local(student_env, act)); teacher_env.step(np.zeros((5, 3)))
        assert np.allclose(student_env.positions_mm, teacher_env.positions_mm, atol=1e-5)


@pytest.mark.parametrize('anatomy,robots', [('mca_m1_lvo', 5), ('femoropopliteal_pad', 3), ('cerebral_venous_sinus', 7)])
def test_student_runs_on_any_anatomy_and_robot_count(anatomy, robots):
    scenes = []
    for a, n in ((anatomy, robots), ('renal_artery', 4)):
        env = CompiledMCAPhysicalEnv(student_env_config(a, n)); reset_with_valid_particles(env, SEED)
        scenes.append(extract_scene(env))
    model = GraphTransformerStudent(dim=32, heads=2, layers=2)
    out = model(collate_scenes(scenes))
    assert out['direction'].shape[:2] == (2, max(robots, 4)) and torch.isfinite(out['direction']).all()
    actions = student_local_action(model, scenes)
    assert actions[0].shape == (robots, 3) and actions[1].shape == (4, 3)
    # Padding must not change a scene's output.
    alone = model(collate_scenes(scenes[1:]))['direction'][0, :4]
    assert torch.allclose(alone, out['direction'][1, :4], atol=1e-5)
