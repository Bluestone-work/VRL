"""EXP0050 retains the same sensors, rewards, projection and candidate set."""
import hashlib
import numpy as np

from marl.local_maneuver_learning import FEATURE_DIM, N_ACTIONS
from marl.temporal_candidate_learning import HISTORY, CANDIDATE_DIM
from marl.memory_prior_learning import MemoryRecommendationLibrary
from scripts.temporal_learning_episode import TemporalLearningEpisode, temporal_hashes, ROOT

PROTOCOL = ROOT/'configs/experiments/EXP_0050_MEMORY_PRIOR.json'


def memory_prior_hashes():
    hashes = temporal_hashes()
    for file in ('marl/memory_prior_learning.py', 'scripts/memory_prior_episode.py',
                 'scripts/run_memory_prior_learning.py', 'configs/experiments/EXP_0050_MEMORY_PRIOR.json'):
        hashes[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    return hashes


class MemoryPriorEpisode(TemporalLearningEpisode):
    def prepare(self):
        if not isinstance(self.library, MemoryRecommendationLibrary):
            self.library = MemoryRecommendationLibrary(self.cfg, speed=self.library.speed, dt=self.library.dt)
        super().prepare()
        descriptors = self.features[:, HISTORY*FEATURE_DIM:].reshape(-1, N_ACTIONS, CANDIDATE_DIM)
        descriptors[:, :, 5] = 0.
        descriptors[np.arange(self.cfg.clusters), self.library.memory_recommendation, 5] = 1.

    def result(self, policy):
        result = super().result(policy)
        result.update(experiment='EXP_0050_MEMORY_PRIOR', source_hashes=memory_prior_hashes(),
            learning_architecture='temporal_candidate_scoring_with_memory_prior',
            heuristic_recommendation_from_shared_measurements=True)
        return result
