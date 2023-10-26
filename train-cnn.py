#!/usr/bin/env python3

import argparse
import random
import pickle

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torchvision

from solution.dqn import dqn
from solution.modules import BytePixel2FloatPixel

ENVS = {
    "car": ("CarRacing-v2", {"continuous": False}, 300_000),
}
parser = argparse.ArgumentParser("Train a DQN policy on a specified environment")
parser.add_argument("env", choices=ENVS.keys(), help="The environment to train on.")
parser.add_argument("-d", "--device", type=str, default=None, help="The device to train on.")
args = parser.parse_args()

MODEL_PREFIX = args.env
ENV_NAME, ENV_KWARGS, N_STEPS = ENVS[args.env]

print(f"Env: {ENV_NAME} (kwargs: {ENV_KWARGS})")
env = gym.make(ENV_NAME, **ENV_KWARGS)
env = gym.wrappers.ResizeObservation(env, 84)
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, num_stack=4)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

L, H, W = env.observation_space.shape
n_actions = env.action_space.n

assert (L, H, W) == (4, 84, 84)

# [N, obs_dim] -> [N, n_actions]
critic = nn.Sequential(
    BytePixel2FloatPixel(),
    # (N, 3, 84, 84) -> (N, 16, 20, 20)
    nn.Conv2d(in_channels=L,
              out_channels=16,
              kernel_size=(8, 8),
              stride=4),
    nn.ReLU(),
    # (N, 16, 20, 20) -> (N, 32, 9, 9)
    nn.Conv2d(in_channels=16,
              out_channels=32,
              kernel_size=(4, 4),
              stride=2),
    nn.ReLU(),
    nn.Flatten(), # (N, 32, 9, 9) -> (N, 2592)
    nn.Linear(32 * 9 * 9, n_actions),
)

def save_hook(model):
    return MODEL_PREFIX, {
        "critic": model,
        "env_name": ENV_NAME,
        "env_kwargs": ENV_KWARGS,
        "wrappers": [
            (gym.wrappers.ResizeObservation, {"shape": 84}),
            (gym.wrappers.GrayScaleObservation, {}),
            (gym.wrappers.FrameStack, {"num_stack": 4}),
        ]
    }


output_critic = dqn(critic, env,
    save_hook=save_hook,
    n_steps=N_STEPS,
    device=args.device,
    batch_size=128,
    replay_size=20_000,
    lr=1e-4,
    exploration_rate=1.0,
    exploration_decay=1e-5,
)

print("Done")
