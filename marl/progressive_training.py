"""
Progressive Training Framework for Multi-Agent RL

Implements advanced training strategies:
1. Pre-training on simplified tasks
2. Adaptive curriculum learning
3. Multi-stage training pipeline
4. Population-based training (PBT)
5. Self-play for robustness
"""
import numpy as np
import torch
import copy
from typing import Dict, List, Tuple, Optional
from collections import deque


class AdaptiveCurriculum:
    """Adaptive curriculum that adjusts difficulty based on performance.

    Unlike simple threshold-based curriculum, this uses success rate,
    convergence speed, and exploration metrics to decide progression.
    """

    def __init__(
        self,
        initial_difficulty: float = 0.1,
        target_difficulty: float = 1.0,
        success_window: int = 50,
        success_threshold: float = 0.7,
        patience: int = 10,
    ):
        self.difficulty = initial_difficulty
        self.target_difficulty = target_difficulty
        self.success_window = success_window
        self.success_threshold = success_threshold
        self.patience = patience

        self.success_history = deque(maxlen=success_window)
        self.episode_count = 0
        self.stagnant_count = 0
        self.last_increase = 0

    def update(self, success: bool, metrics: Dict[str, float]) -> Dict[str, any]:
        """
        Update curriculum based on episode outcome and metrics.

        Args:
            success: episode success flag
            metrics: additional metrics (removal_rate, exploration, etc.)
        Returns:
            curriculum_info: difficulty changes and statistics
        """
        self.success_history.append(float(success))
        self.episode_count += 1

        # Check if we should increase difficulty
        should_increase = False
        reason = None

        if len(self.success_history) >= self.success_window:
            success_rate = np.mean(self.success_history)

            # Criterion 1: High success rate
            if success_rate >= self.success_threshold:
                should_increase = True
                reason = f"success_rate={success_rate:.2%}"

            # Criterion 2: Stagnation (no improvement for patience episodes)
            elif self.episode_count - self.last_increase > self.patience:
                # Check if performance is plateauing
                recent = list(self.success_history)[-20:]
                if len(recent) >= 20:
                    trend = np.polyfit(range(len(recent)), recent, 1)[0]
                    if abs(trend) < 0.01:  # Flat trend
                        self.stagnant_count += 1
                        if self.stagnant_count >= 3:
                            # Force increase to escape local optimum
                            should_increase = True
                            reason = "stagnation_detected"
                            self.stagnant_count = 0

        # Increase difficulty
        if should_increase and self.difficulty < self.target_difficulty:
            old_difficulty = self.difficulty
            self.difficulty = min(
                self.target_difficulty,
                self.difficulty + 0.1  # Increase by 10%
            )
            self.last_increase = self.episode_count
            self.success_history.clear()

            return {
                'difficulty_changed': True,
                'old_difficulty': old_difficulty,
                'new_difficulty': self.difficulty,
                'reason': reason,
            }

        return {'difficulty_changed': False, 'difficulty': self.difficulty}

    def get_env_params(self) -> Dict[str, any]:
        """
        Translate difficulty to environment parameters.

        Returns:
            env_params: parameters for environment reset
        """
        # Map difficulty [0, 1] to concrete parameters
        num_clots = int(1 + self.difficulty * 4)  # 1-5 clots
        clot_distance = 0.3 + self.difficulty * 0.7  # Near to far
        flow_speed = 0.002 + self.difficulty * 0.006  # Slow to fast

        return {
            'num_clots': num_clots,
            'clot_distance_range': (clot_distance, clot_distance + 0.2),
            'flow_speed': flow_speed,
        }


class PretrainingTasks:
    """Pre-training tasks for learning basic skills."""

    @staticmethod
    def navigation_only(env, num_episodes: int = 100):
        """Learn basic navigation without clots (avoid walls, follow flow)."""
        return {
            'task': 'navigation',
            'num_clots': 0,
            'success_metric': 'wall_avoidance',
            'num_episodes': num_episodes,
        }

    @staticmethod
    def single_clot(env, num_episodes: int = 200):
        """Learn to approach and lyse one clot."""
        return {
            'task': 'single_clot',
            'num_clots': 1,
            'clot_distance': 0.3,  # Close clot
            'success_metric': 'clot_removal',
            'num_episodes': num_episodes,
        }

    @staticmethod
    def coordination(env, num_episodes: int = 200):
        """Learn to split across multiple clots."""
        return {
            'task': 'coordination',
            'num_clots': 3,
            'clot_distance': 0.4,
            'lysis_saturation': 2.0,  # Encourage splitting
            'success_metric': 'all_clots_contacted',
            'num_episodes': num_episodes,
        }

    @staticmethod
    def get_stages():
        """Return ordered list of pre-training stages."""
        return [
            ('navigation', PretrainingTasks.navigation_only),
            ('single_clot', PretrainingTasks.single_clot),
            ('coordination', PretrainingTasks.coordination),
        ]


class PopulationBasedTraining:
    """Population-based training for hyperparameter optimization.

    Maintains a population of agents with different hyperparameters,
    periodically exploits (copies) and explores (perturbs) configurations.
    """

    def __init__(
        self,
        population_size: int = 8,
        exploit_interval: int = 50000,  # steps
        explore_prob: float = 0.2,
    ):
        self.population_size = population_size
        self.exploit_interval = exploit_interval
        self.explore_prob = explore_prob

        # Population: list of (agent, hyperparams, performance)
        self.population = []

    def initialize_population(self, agent_constructor, base_config):
        """Initialize population with varied hyperparameters."""
        hyperparams_to_vary = [
            'lr_actor', 'lr_critic', 'gamma', 'gae_lambda',
            'clip_epsilon', 'entropy_coef', 'hidden_dim',
        ]

        for i in range(self.population_size):
            config = copy.deepcopy(base_config)

            # Randomize hyperparameters
            config['lr_actor'] *= np.random.uniform(0.5, 2.0)
            config['lr_critic'] *= np.random.uniform(0.5, 2.0)
            config['gamma'] = np.random.uniform(0.95, 0.99)
            config['gae_lambda'] = np.random.uniform(0.9, 0.99)
            config['clip_epsilon'] = np.random.uniform(0.1, 0.3)
            config['entropy_coef'] = np.random.uniform(0.001, 0.05)
            config['hidden_dim'] = np.random.choice([64, 128, 256])

            agent = agent_constructor(**config)
            self.population.append({
                'agent': agent,
                'config': config,
                'performance': 0.0,
                'steps': 0,
            })

    def exploit_and_explore(self):
        """
        Exploit: copy weights from better performers.
        Explore: perturb hyperparameters.
        """
        # Sort by performance
        self.population.sort(key=lambda x: x['performance'], reverse=True)

        # Bottom 25% exploits top 25%
        bottom_quartile = self.population_size // 4
        top_quartile = bottom_quartile

        for i in range(self.population_size - bottom_quartile, self.population_size):
            # Exploit: copy weights from top performer
            source_idx = np.random.randint(0, top_quartile)
            source = self.population[source_idx]
            target = self.population[i]

            # Copy network weights
            target['agent'].actor.load_state_dict(source['agent'].actor.state_dict())
            target['agent'].critic.load_state_dict(source['agent'].critic.state_dict())

            # Explore: perturb hyperparameters
            if np.random.rand() < self.explore_prob:
                target['config']['lr_actor'] *= np.random.choice([0.8, 1.2])
                target['config']['lr_critic'] *= np.random.choice([0.8, 1.2])
                target['config']['entropy_coef'] *= np.random.choice([0.8, 1.2])

                # Update optimizer with new learning rates
                for param_group in target['agent'].actor_optimizer.param_groups:
                    param_group['lr'] = target['config']['lr_actor']
                for param_group in target['agent'].critic_optimizer.param_groups:
                    param_group['lr'] = target['config']['lr_critic']

    def update_performance(self, agent_id: int, performance: float):
        """Update performance metric for an agent."""
        self.population[agent_id]['performance'] = performance
        self.population[agent_id]['steps'] += 1


class MultiStageTrainer:
    """Multi-stage training pipeline."""

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.stage_history = []

    def run_pretraining(self, verbose: bool = True):
        """Execute pre-training stages sequentially."""
        stages = PretrainingTasks.get_stages()

        for stage_name, stage_func in stages:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Pre-training Stage: {stage_name}")
                print(f"{'='*80}")

            task_config = stage_func(self.env)
            metrics = self._train_stage(task_config)

            self.stage_history.append({
                'stage': stage_name,
                'metrics': metrics,
            })

            if verbose:
                print(f"Stage {stage_name} completed:")
                print(f"  Success rate: {metrics['success_rate']:.2%}")
                print(f"  Average return: {metrics['avg_return']:.1f}")

    def _train_stage(self, task_config: Dict) -> Dict:
        """Train on a specific task configuration."""
        # Modify environment
        num_episodes = task_config.get('num_episodes', 100)

        # TODO: Implement actual training loop
        # This is a placeholder
        return {
            'success_rate': 0.5,
            'avg_return': 50.0,
        }


class SelfPlayTraining:
    """Self-play training for robustness.

    Trains against historical versions of itself to avoid overfitting
    to specific scenarios.
    """

    def __init__(self, checkpoint_interval: int = 50000):
        self.checkpoint_interval = checkpoint_interval
        self.opponent_pool = []
        self.current_steps = 0

    def add_checkpoint(self, agent):
        """Add current agent to opponent pool."""
        opponent = copy.deepcopy(agent)
        opponent.eval()  # Set to evaluation mode
        self.opponent_pool.append(opponent)

        # Keep pool size manageable (last 10 checkpoints)
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)

    def sample_opponent(self):
        """Sample an opponent from the pool."""
        if not self.opponent_pool:
            return None
        return np.random.choice(self.opponent_pool)

    def should_checkpoint(self) -> bool:
        """Check if it's time to create a checkpoint."""
        return self.current_steps % self.checkpoint_interval == 0


class ExplorationBonus:
    """Intrinsic motivation for exploration.

    Provides bonus rewards for visiting novel states.
    """

    def __init__(self, state_dim: int, bonus_coef: float = 0.01):
        self.state_dim = state_dim
        self.bonus_coef = bonus_coef

        # Simple count-based exploration
        self.visit_counts = {}

    def compute_bonus(self, state: np.ndarray) -> float:
        """
        Compute exploration bonus for a state.

        Args:
            state: state vector
        Returns:
            bonus: intrinsic reward
        """
        # Discretize state for counting
        state_key = tuple(np.round(state, decimals=2))

        # Update count
        count = self.visit_counts.get(state_key, 0)
        self.visit_counts[state_key] = count + 1

        # Bonus inversely proportional to visit count
        bonus = self.bonus_coef / np.sqrt(count + 1)

        return bonus


class AdaptiveBatchSizing:
    """Dynamically adjust batch size based on training stability."""

    def __init__(
        self,
        initial_batch_size: int = 256,
        min_batch_size: int = 64,
        max_batch_size: int = 1024,
    ):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        self.loss_history = deque(maxlen=20)

    def update(self, actor_loss: float, critic_loss: float):
        """Adjust batch size based on loss stability."""
        self.loss_history.append(actor_loss + critic_loss)

        if len(self.loss_history) >= 20:
            # Compute loss variance
            variance = np.var(self.loss_history)

            # High variance: decrease batch size (more frequent updates)
            if variance > 1.0:
                self.batch_size = max(
                    self.min_batch_size,
                    int(self.batch_size * 0.9)
                )

            # Low variance: increase batch size (more stable)
            elif variance < 0.1:
                self.batch_size = min(
                    self.max_batch_size,
                    int(self.batch_size * 1.1)
                )

    def get_batch_size(self) -> int:
        return self.batch_size
