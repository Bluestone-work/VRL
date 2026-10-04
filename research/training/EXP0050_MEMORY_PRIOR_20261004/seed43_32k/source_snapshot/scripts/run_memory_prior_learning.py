"""PPO interventions initialized at a strong matched local memory controller."""
from scripts import run_local_learning as runner
from marl.memory_prior_learning import MemoryPriorActorCritic
from scripts.memory_prior_episode import MemoryPriorEpisode, memory_prior_hashes, PROTOCOL


if __name__ == '__main__':
    runner.LearningEpisode = MemoryPriorEpisode
    runner.LocalManeuverActorCritic = MemoryPriorActorCritic
    runner.learning_hashes = memory_prior_hashes
    runner.PROTOCOL = PROTOCOL
    runner.main()
