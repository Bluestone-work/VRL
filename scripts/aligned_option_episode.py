"""EXP0052 reuses frozen options/physics with a shared command-aligned tracker."""
import hashlib
import json
import numpy as np
from marl.command_aligned_tracking import CommandAlignedSensorAdapter
from marl.measured_options import measured_assignment,N_OPTIONS
from scripts.option_learning_episode import OptionEpisode,option_hashes,ROOT

PROTOCOL=ROOT/'configs/experiments/EXP_0052_ALIGNED_HIERARCHY.json'


def aligned_hashes():
    result=option_hashes()
    for name in ('marl/command_aligned_tracking.py','marl/aligned_option_learning.py',
        'scripts/aligned_option_episode.py','scripts/run_aligned_options.py',
        'scripts/run_aligned_option_study.py','configs/experiments/EXP_0052_ALIGNED_HIERARCHY.json'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


class AlignedOptionEpisode(OptionEpisode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.protocol=json.loads(PROTOCOL.read_text())
        self.sensor=CommandAlignedSensorAdapter(self.env,self.cfg,self.sensor.spec)
        self.sensor.reset(int(np.random.SeedSequence([self.scene_seed,self.control_seed]).generate_state(1)[0]))
        self._sensor_execute=self.sensor.execute
        if self.coupling:self.sensor.execute=self._synthetic_actuator_execute
        self.prepare()

    def prepare(self):
        super().prepare()
        self.recommendation=measured_assignment(self.packet,self.target_ids)
        prior=np.zeros((self.cfg.clusters,N_OPTIONS),np.float32)
        prior[np.arange(self.cfg.clusters),self.recommendation]=1.
        self.high_features=np.concatenate((self.high_features,prior),axis=1)

    def result(self,policy):
        result=super().result(policy)
        result.update(experiment='EXP_0052_ALIGNED_HIERARCHY',source_hashes=aligned_hashes(),
            observation_contract='command_aligned_tracked_v1',
            measured_command_history_input=True,
            actor_prior='measured balanced assignment with hysteresis; shared with baseline',
            observer_hardware_calibrated=False)
        return result
