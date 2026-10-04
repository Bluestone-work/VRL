from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from marl.multicluster import MultiClusterConfig
from marl.tracked_sensors import (TrackingSpec, TrackedSensorAdapter, MeasurementProcessor,
                                 ImageFrame, GeometryCrop, empty_crop)
from scripts.tracked_learning_episode import TrackingManeuverLibrary, TrackedLearningEpisode
from scripts.local_learning_episode import LearningEpisode


class ForbiddenEnv:
    @property
    def velocity_mm_s(self): raise AssertionError('True velocity forbidden')
    @property
    def tree(self): raise AssertionError('Global graph forbidden')
    @property
    def robot_stations(self): raise AssertionError('True station forbidden')
    @property
    def edges(self): raise AssertionError('Body edge ID forbidden')
    @property
    def routes(self): raise AssertionError('Global routes forbidden')


def environment():
    e = ForbiddenEnv()
    e.num_robots = 2
    e.positions_mm = np.array([[0., 0., 0.], [3., 0., 0.], [.8, .1, 0.]])
    e.active = np.ones(3, bool)
    e.transport = SimpleNamespace(points=np.array([[-5., 0, 0], [0., 0, 0], [5., 0, 0]]),
                                  ends=np.array([[0, 1], [1, 2]]))
    e.solution = {'radius_mm': np.ones(3)}
    e.clot_positions_mm = np.array([[1., 0, 0], [20., 0, 0]])
    e.masses = np.array([.9, .9])
    e.elapsed_s = 0.
    e.config = SimpleNamespace(robot_speed_mm_s=1., robot_radius_mm=.08, episode_duration_s=180.)
    return e


def clean_spec(**changes):
    return replace(TrackingSpec(), position_error_mm=0., centerline_error_mm=0.,
                   radius_relative_error=0., preop_error_mm=0., dropout=0.,
                   clot_classification_error=0., latency_s=changes.pop('latency_s', 0.),
                   velocity_alpha=1., **changes)


def sensor(e, spec=None):
    result = TrackedSensorAdapter(e, MultiClusterConfig(clusters=2), spec or clean_spec())
    result.reset(123)
    return result


def test_no_true_velocity_route_station_or_body_edge_access():
    e = environment(); s = sensor(e)
    first = s.observe()
    assert first.active.all()
    assert np.all(first.navigation[:, 3:6] == 0)
    e.positions_mm[0, 0] += .1; e.elapsed_s = .1
    second = s.observe()
    assert second.navigation[0, 3] == pytest.approx(1.)
    assert not hasattr(s.processor, 'env')


def test_exact_mass_and_unseen_global_clearance_do_not_enter_policy():
    a, b = environment(), environment()
    b.masses = np.array([.01, 0.])
    x, y = sensor(a).observe(), sensor(b).observe()
    assert np.array_equal(x.navigation, y.navigation)
    assert np.array_equal(x.clot_ids, y.clot_ids)
    # Visible but uncleared mass fractions 0.9 and 0.01 both produce presence=1.
    assert x.navigation[0, 16] == y.navigation[0, 16] == 1.


def test_latency_and_expired_tracks_are_not_replaced_with_true_state():
    e = environment(); s = sensor(e, clean_spec(latency_s=.1))
    assert not s.observe().active.any()
    e.elapsed_s = .1
    assert s.observe().active.all()
    assert not s.processor.observe(.5).active.any()
    assert np.all(s.processor.observe(.5).navigation == 0)


def test_clearance_requires_repeated_local_binary_evidence():
    e = environment(); e.masses[:] = 0.
    s = sensor(e)
    for k in range(2):
        e.elapsed_s = .1*k; s.observe()
        assert s.processor.alive[0]
    e.elapsed_s = .2; s.observe()
    assert not s.processor.alive[0]
    assert s.processor.alive[1], 'Unseen target must not receive an oracle-clearance update'


def test_local_geometry_packet_has_no_global_ids_and_actions_stay_finite():
    e = environment(); s = sensor(e)
    packet = s.observe()
    cfg = MultiClusterConfig(clusters=2)
    library = TrackingManeuverLibrary(cfg)
    features, actions, valid, details = library.prepare(packet)
    assert np.isfinite(features).all() and np.isfinite(actions).all()
    assert np.linalg.norm(actions, axis=-1).max() <= 1+1e-6
    assert valid[:, 4:8].any()
    assert not hasattr(library, 'env')
    for crop in s.imager.render(0.).geometry:
        assert set(crop.__dataclass_fields__) == {'segments', 'radii'}


def test_measured_frame_processor_is_repeatable():
    a, b = environment(), environment()
    sa, sb = sensor(a, TrackingSpec()), sensor(b, TrackingSpec())
    for k in range(5):
        a.elapsed_s = b.elapsed_s = k*.1
        a.positions_mm[:, 0] += .01; b.positions_mm[:, 0] += .01
        pa, pb = sa.observe(), sb.observe()
        assert np.array_equal(pa.navigation, pb.navigation)
        assert np.array_equal(pa.peer_relative_mm, pb.peer_relative_mm)


@pytest.mark.parametrize('confirmed,expected_done', [(False, False), (True, True)])
def test_simulator_clearance_is_not_an_oracle_stop(monkeypatch, confirmed, expected_done):
    episode = TrackedLearningEpisode.__new__(TrackedLearningEpisode)
    episode.cfg = SimpleNamespace(clusters=2)
    episode.env = SimpleNamespace(active=np.array([True, True]), elapsed_s=1.,
                                  config=SimpleNamespace(episode_duration_s=180.), _done=True)
    episode.packet = SimpleNamespace(active=np.array([True, False]))
    episode.sensor = SimpleNamespace(processor=SimpleNamespace(alive=np.array([not confirmed])))
    episode.walls = SimpleNamespace(active_robot_s=.1)
    episode.info = {'step_duration_s': .1}
    calls = []
    episode.prepare = lambda: calls.append('measured_frame')
    def parent_step(self, choices, *, legacy=False):
        self.done = True
        return np.zeros(2), True, self.packet.active, np.zeros(2, bool)
    monkeypatch.setattr(LearningEpisode, 'step', parent_step)
    _, done, _, _ = episode.step(np.zeros(2, int))
    assert done is expected_done and episode.env._done is expected_done
    assert calls == ['measured_frame']
    assert episode.walls.active_robot_s == pytest.approx(.2)
    assert episode.info['termination_reason'] == ('observed_all_clots_cleared' if confirmed else None)
