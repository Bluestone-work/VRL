import numpy as np
import pytest
import torch
from marl.aligned_option_learning import AllocationPriorActorCritic
from scripts.aligned_option_episode import AlignedOptionEpisode


@pytest.mark.parametrize('decision_steps',[1,50])
def test_untrained_allocation_prior_matches_classical_at_same_interval(decision_steps):
    a=AlignedOptionEpisode(1840000100,duration=2.,option_steps=decision_steps)
    b=AlignedOptionEpisode(1840000100,duration=2.,option_steps=decision_steps)
    model=AllocationPriorActorCritic()
    try:
        while not a.done:a.step_option(a.conventional_options('balanced'))
        while not b.done:
            with torch.no_grad():
                distribution,_=model(torch.from_numpy(b.high_features)[None],torch.from_numpy(b.high_valid)[None])
                choice=distribution.logits.argmax(-1)[0].numpy()
            np.testing.assert_array_equal(choice,b.conventional_options('balanced'))
            b.step_option(choice)
        assert a.result('balanced')['final_state_hash']==b.result('untrained')['final_state_hash']
        assert a.initial_snapshot_hash==b.initial_snapshot_hash
    finally:a.close();b.close()


def test_aligned_sensor_records_only_requested_commands_and_preserves_track_expiry():
    ep=AlignedOptionEpisode(1840000101,duration=.5)
    try:
        while not ep.done:ep.step_option(ep.conventional_options('memory'))
        processor=ep.sensor.processor
        assert not hasattr(processor,'env')
        assert len(processor.commands.times)==ep.steps
        assert not processor.observe(ep.env.elapsed_s+1.).active.any()
        assert ep.result('memory')['exact_mass_input'] is False
    finally:ep.close()


def test_cross_coupling_is_physics_only_and_does_not_replace_command_history():
    ep=AlignedOptionEpisode(1840000102,duration=.5,coupling=.1)
    try:
        commands=np.array([[.8,0.,0.],[-.4,0.,0.],[0.,.3,0.]])
        response=ep.sensor.execute(commands)
        np.testing.assert_array_equal(ep.sensor.processor.last_commands,commands)
        np.testing.assert_array_equal(ep.sensor.processor.commands.at(ep.env.elapsed_s),commands)
        assert not np.array_equal(response,commands)
        assert np.linalg.norm(response,axis=1).max()<=1+1e-12
    finally:ep.close()
