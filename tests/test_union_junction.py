"""Union-of-tubes junction model: reference and compiled backends agree; feasibility is a function of position."""
from dataclasses import replace

import numpy as np
import pytest

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from marl.teacher import TEACHER_CONFIG
from scripts.train_mca_compiled import reset_with_valid_particles


def cfg(model, anatomy='mca_m1_lvo'):
    return replace(DynamicsConfig.from_json(TEACHER_CONFIG), action_prior='none', junction_model=model, anatomy=anatomy)


@pytest.mark.parametrize('anatomy', ['mca_m1_lvo', 'cerebral_venous_sinus'])
def test_backends_agree_under_union(anatomy):
    runs = []
    for cls in (CompiledMCAPhysicalEnv, MCAPhysicalEnv):
        env = cls(cfg('union', anatomy)); reset_with_valid_particles(env, 940000000)
        rng = np.random.default_rng(1); trace = []
        for _ in range(40):
            env.step(rng.normal(size=(env.num_robots, 3)))
            trace.append(env.positions_mm.copy())
        runs.append(np.array(trace))
    assert np.allclose(runs[0], runs[1], atol=1e-6)


def test_a_body_follows_the_axis_into_an_acute_side_branch_only_under_union():
    reached = {}
    for model in ('graph', 'union'):
        env = CompiledMCAPhysicalEnv(replace(cfg(model), num_robots=1, command_speed='bounded'))
        reset_with_valid_particles(env, 940000000)
        t = env.transport
        env.positions_mm[0] = t.points[12]; env.edges[0] = int(np.flatnonzero((t.ends == [11, 12]).all(1))[0]); env._sync_public_state()
        for wp in (11, 83, 84, 85):                       # M1 trunk -> perforator take-off -> along the perforator axis
            for _ in range(60):
                d = t.points[wp]-env.positions_mm[0]; nd = np.linalg.norm(d)
                if nd < .02:
                    break
                env.step((d/max(nd, 1e-9)*min(1., nd/.1))[None])
        reached[model] = float(np.linalg.norm(env.positions_mm[0]-t.points[85]))
    assert reached['union'] < .05 and reached['graph'] > .5


def test_graph_model_is_the_default():
    assert DynamicsConfig.from_json(TEACHER_CONFIG).junction_model == 'graph'
