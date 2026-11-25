"""Side-pool replay and Double-DQN-with-min-clipping, ported from the maze project.

Two mechanisms are taken from that project's publishable configuration; one is
deliberately NOT taken:

* Side pool (taken). A bounded secondary pool holds transitions whose reward
  exceeds a threshold, and a fixed fraction of every batch is drawn from it by
  UNIFORM sampling. Here the high-reward events are the +20 joint-success
  terminal (one step in an episode of up to 3000) and the -8 cluster loss; both
  are rare and their gradient is otherwise diluted by ordinary progress steps.

* Double Q with min clipping (taken). The online net picks the next action, the
  target net values it, and the result is clipped against the target net's own
  maximum. This is the direct remedy for what was measured here: plain DQN at
  300k steps scored *worse* than at 100k (28.3% vs 37-58.7% training success),
  the signature of value overestimation.

* Prioritised replay (NOT taken). That project measured uniform > standard PER >
  its own priority rule, and concluded priority replay is not a contribution.
  The main pool therefore stays uniform.
"""
from __future__ import annotations
from collections import deque
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer

HIGH_RETURN = 5.0          # +8 newly-reached, +20 success terminal, -8 loss all clear this
SIDE_FRACTION = .25
SIDE_CAP = 5000


class SidePoolReplayBuffer(ReplayBuffer):
    """Uniform main pool plus a uniformly sampled pool of high-|reward| steps."""

    def __init__(self, *args, high_return=HIGH_RETURN, side_fraction=SIDE_FRACTION,
                 side_capacity=SIDE_CAP, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_return = float(high_return)
        self.side_fraction = float(side_fraction)
        self.side = deque(maxlen=int(side_capacity))
        self._writes = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self._writes += 1
        r = np.asarray(reward).reshape(-1)
        for env_i in range(min(self.n_envs, r.shape[0])):
            # Magnitude, so a costly loss is kept as well as a success.
            if abs(float(r[env_i])) >= self.high_return:
                self.side.append((pos, env_i, self._writes))

    def _live_side(self):
        """Side entries whose slot has not yet been overwritten by the ring buffer."""
        return [(p, e) for p, e, stamp in self.side
                if self._writes - stamp < self.buffer_size]

    def sample(self, batch_size, env=None):
        live = self._live_side()
        n_side = min(len(live), int(batch_size * self.side_fraction))
        n_main = batch_size - n_side
        upper = self.buffer_size if self.full else self.pos
        if upper <= 0:
            return super().sample(batch_size, env=env)
        main_pos = np.random.randint(0, upper, size=n_main)
        main_env = np.random.randint(0, self.n_envs, size=n_main)
        if n_side:
            chosen = [live[i] for i in np.random.choice(len(live), n_side, replace=False)]
            side_pos = np.array([p for p, _ in chosen], dtype=np.int64)
            side_env = np.array([e for _, e in chosen], dtype=np.int64)
            pos_idx = np.concatenate([main_pos, side_pos])
            env_idx = np.concatenate([main_env, side_env])
        else:
            pos_idx, env_idx = main_pos, main_env
        return self._get_samples_at(pos_idx, env_idx, env=env)

    def _get_samples_at(self, batch_inds, env_indices, env=None):
        """ReplayBuffer._get_samples with the env index supplied rather than drawn."""
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices]
             * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def side_size(self):
        return len(self._live_side())


class DoubleDQN(DQN):
    """DQN whose bootstrap uses Double Q with min clipping."""

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                pick = self.q_net(data.next_observations).argmax(dim=1, keepdim=True)
                valued = self.q_net_target(data.next_observations).gather(1, pick)
                target_max = self.q_net_target(data.next_observations).max(dim=1, keepdim=True)[0]
                bootstrap = th.min(valued, target_max).reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * self.gamma * bootstrap
            current_q = th.gather(self.q_net(data.observations), dim=1,
                                  index=data.actions.long())
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        if hasattr(self.replay_buffer, "side_size"):
            self.logger.record("train/side_pool", self.replay_buffer.side_size())
