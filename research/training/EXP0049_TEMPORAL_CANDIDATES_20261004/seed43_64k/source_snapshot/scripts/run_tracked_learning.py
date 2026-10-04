"""Run the unchanged PPO procedure with the EXP0048 measurement contract."""
from scripts import run_local_learning as runner
from scripts.tracked_learning_episode import TrackedLearningEpisode, tracking_hashes, PROTOCOL


if __name__ == '__main__':
    runner.LearningEpisode = TrackedLearningEpisode
    runner.learning_hashes = tracking_hashes
    runner.PROTOCOL = PROTOCOL
    runner.main()
