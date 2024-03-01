import random
import numpy as np
from segmentation_tree import SumSegmentTree, MinSegmentTree

from buffer_store import BufferStore




class ReplayBuffer(BufferStore):
    def __init__(self, buffer_size, buffer_path, buffer_name, load_buffer=False, seed=None):
        super(ReplayBuffer, self).__init__(buffer_path, buffer_name)
        assert isinstance(buffer_size, int)
        self.__all_seed(seed)
        self.buffer = super().load() if load_buffer else []
        self.capacity = buffer_size
        self.pos = 0

    def __all_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, sample):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        self._add(sample)

    def save(self):
        super().save(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, alpha, buffer_path, buffer_name, load_buffer, seed=None):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, buffer_path, buffer_name, load_buffer, seed)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)