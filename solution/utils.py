import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from collections import namedtuple


class ReplayMemoryBase:
    Parameters = []
    Sample = namedtuple("ReplayMemoryBaseSample", Parameters)
    REPLACEMENT_FIFO = 1
    REPLACEMENT_UNIFORM = 2
    def __init__(self, maxsize: int, env: gym.core.Env,
                 replacement=REPLACEMENT_FIFO):
        self.maxsize = maxsize
        self.length = 0
        self.next_index = 0
        self.replacement = replacement
        self.total_adds = 0

        def dtype_to_torch_dtype(ty):
            FLOATS = [np.dtype("float16"), np.dtype("float32"), np.dtype("float64")]
            INTS = [np.dtype("int16"), np.dtype("int32"), np.dtype("int64")]
            PIXELS = [np.dtype("uint8")]
            if ty in FLOATS:
                return torch.float32
            elif ty in INTS:
                return torch.long
            elif ty in PIXELS:
                return torch.uint8
            else:
                raise ValueError(f"unsupported numpy dtype {ty}")

        self.s_shape = env.observation_space.shape
        self.s_type = dtype_to_torch_dtype(env.observation_space.dtype)
        self.a_shape = env.action_space.shape
        self.a_type = dtype_to_torch_dtype(env.action_space.dtype)

    def _getall(self, idxs):
        return ()

    def _add(self, *args):
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, idxs):
        return self.Sample(*self._getall(idxs))

    def reset(self):
        self.length = 0
        self.next_index = 0

    def add(self, *args):
        self.total_adds += 1
        if self.length < self.maxsize:
            # Always add if there is room over
            self._add(*args)
            self.next_index += 1
            self.length += 1
        else:
            if self.replacement == self.REPLACEMENT_FIFO:
                # Replace the oldest thing in the buffer
                self._add(*args)
                self.next_index += 1
            elif self.replacement == self.REPLACEMENT_UNIFORM:
                # With probability |B| / total_adds, replace something at random
                # in the buffer. Otherwise throw the inserted data away.
                p = self.length / self.total_adds
                if p > np.random.random():
                    self.next_index = np.random.randint(self.length)
                    self._add(*args)
        if self.next_index >= self.maxsize:
            self.next_index = 0

    def sample(self, n, device=None):
        """
        Returns a tuple representing a sample from the buffer

        Example usage:
        bs, ba, br, bsn, bd = replay.sample(n)
        for i in range(n):
            # Extract a single batch item
            s = bs[i]
            a = as[i]
            r = br[i]
            sn = bsn[i]
            bd = bd[i]
        """
        idxs = np.random.randint(0, self.length, size=(n,))
        if device is None:
            return self.__getitem__(idxs)
        else:
            return self.Sample(*tuple(t.to(device=device, non_blocking=True) for t in self._getall(idxs)))


class ReplayMemory(ReplayMemoryBase):
    """A standard replay buffer."""
    Parameters = ReplayMemoryBase.Parameters + ["state", "action", "reward", "next_state", "is_done"]
    Sample = namedtuple("ReplayMemorySample", Parameters)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb_state = torch.zeros((self.maxsize,) + self.s_shape, requires_grad=False, dtype=self.s_type)
        self.rb_action = torch.zeros((self.maxsize,) + self.a_shape, requires_grad=False, dtype=self.a_type)
        self.rb_reward = torch.zeros((self.maxsize,), requires_grad=False, dtype=torch.float32)
        self.rb_next_state = torch.zeros((self.maxsize,) + self.s_shape, requires_grad=False, dtype=self.s_type)
        self.rb_is_done = torch.zeros((self.maxsize,), requires_grad=False, dtype=torch.float32)

    def _getall(self, idxs):
        return (
            self.rb_state[idxs],
            self.rb_action[idxs],
            self.rb_reward[idxs],
            self.rb_next_state[idxs],
            self.rb_is_done[idxs],
        )

    def _add(self, state, action, reward, next_state, is_done):
        if not isinstance(state, torch.Tensor):      state = torch.tensor(state, dtype=self.s_type)
        if not isinstance(action, torch.Tensor):     action = torch.tensor(action, dtype=self.a_type)
        if not isinstance(reward, torch.Tensor):     reward = torch.tensor(reward, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor): next_state = torch.tensor(next_state, dtype=self.s_type)
        if not isinstance(is_done, torch.Tensor):    is_done = torch.tensor(is_done, dtype=torch.float32)
        self.rb_state[self.next_index] = state
        self.rb_action[self.next_index] = action
        self.rb_reward[self.next_index] = reward
        self.rb_next_state[self.next_index] = next_state
        self.rb_is_done[self.next_index] = is_done
