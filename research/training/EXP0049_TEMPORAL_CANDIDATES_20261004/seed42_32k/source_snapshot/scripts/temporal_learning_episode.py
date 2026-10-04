"""EXP0049: same measured sensing/actuation, causal candidate-score learner."""
from collections import deque
import hashlib

from marl.temporal_candidate_learning import HISTORY, candidate_features, pack_history
from scripts.tracked_learning_episode import TrackedLearningEpisode, tracking_hashes, ROOT

PROTOCOL = ROOT/'configs/experiments/EXP_0049_TEMPORAL_CANDIDATES.json'


def temporal_hashes():
    hashes = tracking_hashes()
    for file in ('marl/temporal_candidate_learning.py', 'scripts/temporal_learning_episode.py',
                 'scripts/run_temporal_learning.py', 'configs/experiments/EXP_0049_TEMPORAL_CANDIDATES.json'):
        hashes[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    return hashes


class TemporalLearningEpisode(TrackedLearningEpisode):
    def __init__(self, *args, **kwargs):
        self.measured_history = deque(maxlen=HISTORY)
        super().__init__(*args, **kwargs)

    def prepare(self):
        super().prepare()
        self.measured_history.append(self.features.copy())
        c = candidate_features(self.packet, self.candidates, self.details,
                               speed=self.library.speed, spacing=self.cfg.min_spacing_mm)
        self.features = pack_history(self.measured_history, c)

    def result(self, policy):
        result = super().result(policy)
        result.update(experiment='EXP_0049_TEMPORAL_CANDIDATES', source_hashes=temporal_hashes(),
            learning_architecture='causal_6_frame_GRU_shared_candidate_score',
            candidate_descriptors_from_measurements_only=True)
        return result
