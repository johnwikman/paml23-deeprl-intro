"""
This declares some additional PyTorch and Gymnasium modules that are nice to
have.
"""


import gymnasium as gym
import numpy as np
import torch


class BytePixel2FloatPixel(torch.nn.Module):
    """Convert uint8[0,255] pixel to float32[0,1] pixel."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float32) / 255.0


class DiscretizeActions(gym.ActionWrapper):
    """
    Discretizes the action space such that each discrete action corresponds
    to a continuous output.
    """
    def __init__(self, env: gym.core.Env, action_map):
        super().__init__(env)
        self.__action_map = action_map

        self.action_space = gym.spaces.Discrete(len(self.__action_map))

    def action(self, a):
        return self.__action_map[int(a)]
