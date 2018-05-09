import numpy as np
import random

from torchcraft_pure.model.util.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, goal_t, action, reward, obs_tp1, goal_tp1, done):
        data = (obs_t, goal_t, action, reward, obs_tp1, goal_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, goals_t, actions, rewards, obses_tp1, goals_tp1, dones = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, goal_t, action, reward, obs_tp1, goal_tp1, done = data
            # obses_t.append(np.array(obs_t, copy=False))
            obses_t.append(obs_t)
            goals_t.append(goal_t)
            actions.append(action)
            rewards.append([reward])
            obses_tp1.append(obs_tp1)
            goals_tp1.append(goal_tp1)
            dones.append([done])
        return np.array(obses_t), np.array(goals_t), np.array(actions), np.array(rewards), np.array(
            obses_tp1), np.array(goals_tp1), np.array(dones)

    def make_index(self, batch_size):
        # sampled_idx = []
        # while len(sampled_idx) < batch_size:
        #     idx = random.randint(0, len(self._storage) - 1)
        #     if self._storage[idx][-1]: # still alive
        #         sampled_idx.append(idx)
        # # print(len(sampled_idx))
        # return sampled_idx
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
