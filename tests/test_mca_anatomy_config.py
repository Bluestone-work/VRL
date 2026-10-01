"""Configurable anatomy and per-anatomy fixed splits."""
from dataclasses import replace

import numpy as np
import pytest

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from environments.vessel_anatomy import TERRITORIES
from scripts.mca_eval_splits import load_registry, split
from scripts.train_mca_physical import reset_with_valid_particles

CONFIG = 'configs/experiments/EXP_0042_ROUTE_AVOID_PRIOR_DYNAMICS.json'


def test_default_anatomy_is_mca_and_unknown_is_rejected():
    assert DynamicsConfig.from_json(CONFIG).anatomy == 'mca_m1_lvo'
    with pytest.raises(ValueError, match='anatomy'):
        replace(DynamicsConfig.from_json(CONFIG), anatomy='spleen')


def test_every_territory_has_disjoint_fixed_splits():
    registry = load_registry()
    assert set(registry['anatomies']) == set(TERRITORIES)
    assert registry['anatomy_order'][0] == 'mca_m1_lvo'
    assert split('test') == (990000000, 500)  # the MCA sealed test never moves


def test_env_builds_the_configured_anatomy():
    cfg = replace(DynamicsConfig.from_json(CONFIG), anatomy='coronary_rca')
    env = CompiledMCAPhysicalEnv(cfg)
    reset_with_valid_particles(env, split('diagnostic', 'coronary_rca')[0])
    assert env.tree.territory.name == 'coronary_rca'
    env.step(np.zeros((cfg.num_robots, 3)))
