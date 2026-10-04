"""Causal command integration and track expiry, using analytic trajectories."""
from dataclasses import replace
import numpy as np
import pytest

from marl.command_aligned_tracking import CommandHistory, CommandAlignedProcessor
from marl.tracked_sensors import TrackingSpec, ImageFrame, GeometryCrop


def frame(t,x):
    crop=GeometryCrop(np.array([[[-10.,0.,0.],[10.,0.,0.]]]),np.ones((1,2)))
    return ImageFrame(t,np.array([[x,0.,0.]]),np.zeros(0,int),np.zeros((0,3)),(crop,),
        np.array([False]),np.array([False]),np.full((1,3),np.nan))


def test_piecewise_command_integral_has_correct_boundary_convention():
    h=CommandHistory(1)
    h.record(0.,[[1.,0.,0.]])
    h.record(.1,[[-1.,0.,0.]])
    np.testing.assert_allclose(h.integral(0.,.1),[[.1,0.,0.]])
    np.testing.assert_allclose(h.integral(.05,.15),np.zeros((1,3)),atol=1e-15)
    h.record(.1,[[0.,0.,0.]])
    np.testing.assert_allclose(h.integral(0.,.2),[[.1,0.,0.]])


def test_future_input_does_not_change_past_integrals():
    h=CommandHistory(1);h.record(0.,[[1.,0.,0.]])
    before=h.integral(0.,.1)
    h.record(1.,[[100.,0.,0.]])
    np.testing.assert_array_equal(h.integral(0.,.1),before)
    with pytest.raises(ValueError):h.record(.5,[[1.,0.,0.]])


def test_delayed_measurement_prediction_integrates_known_command_changes():
    s=replace(TrackingSpec(),velocity_alpha=.25,track_lifetime_s=.3)
    p=CommandAlignedProcessor(1,np.array([[5.,0.,0.]]),s)
    p.record_command(0.,np.array([[1.,0.,0.]]))
    p.ingest(frame(0.,0.))
    # Constant exogenous drift .2 plus +1 command during [0,.1).
    p.record_command(.1,np.array([[-1.,0.,0.]]))
    p.ingest(frame(.1,.12))
    observed=p.observe(.2)
    np.testing.assert_allclose(p.drift[0],[.2,0.,0.],atol=1e-12)
    np.testing.assert_allclose(observed.navigation[0,3:6],[-.8,0.,0.],atol=1e-7)
    # From x=.12 at t=.1, .1*(.2-1) puts the robot at x=.04.
    assert observed.navigation[0,12]*10==pytest.approx(4.96,abs=1e-6)
    assert p.tracks[('robot',0)][0]==.1
    assert not hasattr(p,'env')


def test_missing_track_is_not_kept_alive_by_command_prediction():
    p=CommandAlignedProcessor(1,np.array([[5.,0.,0.]]),TrackingSpec())
    p.record_command(0.,np.ones((1,3)))
    p.ingest(frame(0.,0.))
    assert not p.observe(.31).active.any()


def test_duplicate_frame_cannot_update_filter_twice():
    p=CommandAlignedProcessor(1,np.array([[5.,0.,0.]]),TrackingSpec())
    p.record_command(0.,np.zeros((1,3)))
    p.ingest(frame(0.,0.));p.ingest(frame(.1,.02))
    before=p.drift[0].copy()
    p.ingest(frame(.1,999.))
    np.testing.assert_array_equal(p.drift[0],before)


def test_noise_free_constant_drift_remains_consistent_over_input_switches():
    p=CommandAlignedProcessor(1,np.array([[5.,0.,0.]]),TrackingSpec())
    p.record_command(0.,np.zeros((1,3)));p.ingest(frame(0.,0.))
    x=0.
    for k,command in enumerate([1.,-1.,.5,0.,-.5,1.]):
        t=k*.1;p.record_command(t,np.array([[command,0.,0.]]))
        x+=(command+.3)*.1;p.ingest(frame(t+.1,x))
        np.testing.assert_allclose(p.drift[0],[.3,0.,0.],atol=1e-12)
