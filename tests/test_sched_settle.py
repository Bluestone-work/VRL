"""ScheduledSettle with a = 0 must reproduce AdaptiveSettleGuard exactly (same scene, latency, variation)."""
import numpy as np
import pytest

from marl.deployable_sensing import DeployableConfig
from marl.sched_settle import ACT_DIM, ScheduledSettle
from scripts.benchmark_lysis import AdaptiveSettleGuard, LysisEpisode, SettleGuard


def _run(ctor, n, anatomy, seed, latency, variation, steps, action=None):
    ep = LysisEpisode(n, anatomy, seed, sense_cfg=DeployableConfig(latency_steps=latency),
                      variation=variation, flow_inlet_mm_s=.05)
    ctl = ctor(ep); trace = []
    for _ in range(steps):
        est = ep.observe(); tgt = ep.plan_targets_now(est); rule = ep.ctl.act(tgt, est); hold = ep.hold(est)
        out = ctl(ep, est, tgt, rule, hold) if action is None else ctl(ep, est, tgt, rule, hold, action(ep.n))
        done, _ = ep.step(est, out, hold); trace.append(ep.env.positions_mm[:n].copy())
        if done:
            break
    ep.close()
    return np.array(trace)


@pytest.mark.parametrize('n,anatomy,seed,latency,variation', [
    (1, 'mca_m1_lvo', 2600000001, 1, None), (3, 'pulmonary_saddle', 2600100002, 3, 1.0),
    (2, 'coronary_rca', 2600300001, 2, 1.25)])
def test_zero_action_equals_adaptive_settle(n, anatomy, seed, latency, variation):
    a = _run(AdaptiveSettleGuard, n, anatomy, seed, latency, variation, 600)
    b = _run(ScheduledSettle, n, anatomy, seed, latency, variation, 600)
    c = _run(ScheduledSettle, n, anatomy, seed, latency, variation, 600, action=lambda k: np.zeros((k, ACT_DIM)))
    assert a.shape == b.shape == c.shape
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_gate_off_equals_fixed_settle_radius_behaviour():
    """a1 = -1 (release disabled) with unit response behaves like Fixed Settle inside the radius."""
    a = _run(SettleGuard, 1, 'mca_m1_lvo', 2600000001, 1, None, 300)
    act = np.zeros((1, ACT_DIM)); act[:, 1] = -1.
    b = _run(ScheduledSettle, 1, 'mca_m1_lvo', 2600000001, 1, None, 300, action=lambda k: act)
    # radius differs only through the response EMA; trajectories must start identical and stay close
    m = min(len(a), len(b))
    assert np.mean(np.linalg.norm(a[:m]-b[:m], axis=-1)) < .5


@pytest.mark.parametrize('latency,ref', [(1, 'adaptive'), (3, 'fixed')])
def test_switch_residual_zero_equals_rule(latency, ref):
    from marl.sched_settle import ACT2_DIM, SwitchResidual
    ctor = AdaptiveSettleGuard if ref == 'adaptive' else SettleGuard
    for n, an, seed, var in [(2, 'mca_m1_lvo', 2600000003, None), (3, 'pulmonary_saddle', 2600100002, 1.0)]:
        a = _run(ctor, n, an, seed, latency, var, 500)
        b = _run(SwitchResidual, n, an, seed, latency, var, 500, action=lambda k: np.zeros((k, ACT2_DIM)))
        assert a.shape == b.shape and np.array_equal(a, b)


def test_flow_residual_zero_equals_switch_rule():
    """The flow-identification arm must retain the SwitchSettle safety fallback at zero action."""
    from marl.sched_settle import ACT_FLOW_DIM, FlowResidual
    for latency, ctor in [(1, AdaptiveSettleGuard), (3, SettleGuard)]:
        a = _run(ctor, 1, 'mca_m1_lvo', 2600000004 + latency, latency, None, 120)
        b = _run(FlowResidual, 1, 'mca_m1_lvo', 2600000004 + latency, latency, None, 120,
                 action=lambda k: np.zeros((k, ACT_FLOW_DIM)))
        assert a.shape == b.shape and np.array_equal(a, b)


def test_flow_policy_token_dimension():
    from scripts.train_sched_settle import spec
    C, action_dim, token_dim = spec({'prior': 'flow'})
    assert C.__name__ == 'FlowResidual' and action_dim == 2 and token_dim == 53
