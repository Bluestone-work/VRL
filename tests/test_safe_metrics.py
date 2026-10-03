"""SafeSuccess = TaskSuccess AND total wall contact over all robots < 1.0 robot-second."""
import numpy as np

from scripts.safe_metrics import WallTracker, aggregate, episode_metrics


def run(steps):
    tr = WallTracker(3)
    for wall in steps:
        tr.update(dict(wall_contact_s=np.array(wall)), np.ones(3, bool), .1)
    return tr


def info(success=True):
    return dict(success=success, collision_free_success=success, remaining_mass=0., elapsed_s=1., termination_reason='all_clots_cleared')


def test_threshold_is_strict_and_summed_over_robots():
    assert episode_metrics(info(), run([[.3, .4, .2]]), 1.)['safe_success']        # 0.9 robot-s
    assert not episode_metrics(info(), run([[.4, .4, .4]]), 1.)['safe_success']    # 1.2 robot-s
    assert not episode_metrics(info(), run([[.5, .5, 0.]]), 1.)['safe_success']    # exactly 1.0 is unsafe
    assert not episode_metrics(info(False), run([[0, 0, 0]]), 1.)['safe_success']  # no task success, no safe success


def test_continuous_run_resets_after_a_free_step():
    m = episode_metrics(info(), run([[.1, 0, 0], [.1, 0, 0], [0, 0, 0], [.1, 0, 0]]), 1.)
    assert np.isclose(m['max_continuous_wall_contact_s'], .2) and np.isclose(m['wall_contact_ratio'], .3/1.2)


def test_aggregate_reports_the_unsafe_success_gap():
    rows = [episode_metrics(info(), run([[2., 0, 0]]), 1.), episode_metrics(info(), run([[0, 0, 0]]), 1.)]
    a = aggregate(rows)
    assert a['raw_success'] == 1. and a['safe_success'] == .5 and a['unsafe_success_gap'] == .5
