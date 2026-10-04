"""Train PPO with measured history and executable-action candidate scoring."""
from scripts import run_local_learning as runner
from marl.temporal_candidate_learning import TemporalCandidateActorCritic
from scripts.temporal_learning_episode import TemporalLearningEpisode, temporal_hashes, PROTOCOL


if __name__ == '__main__':
    runner.LearningEpisode = TemporalLearningEpisode
    runner.LocalManeuverActorCritic = TemporalCandidateActorCritic
    runner.learning_hashes = temporal_hashes
    runner.PROTOCOL = PROTOCOL
    runner.main()
