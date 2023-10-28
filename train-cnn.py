#!/usr/bin/env python3

import argparse
import random
import pickle

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torchvision

import solution
from solution.dqn import dqn
from solution.modules import BytePixel2FloatPixel

ENVS = {
    "atari-breakout": ("ALE/Breakout-v5", {}, [], 2_000_000),
    "car": ("CarRacing-v2", {"continuous": False}, [], 1_000_000),
    "flappybird": ("FlappyBird-v0", {"render_mode": "rgb_array"}, [(gym.wrappers.TimeLimit, {"max_episode_steps": 1000})], 2_000_000),
}
parser = argparse.ArgumentParser("Train a DQN policy on a specified environment")
parser.add_argument("env", choices=ENVS.keys(), help="The environment to train on.")
parser.add_argument("-d", "--device", dest="device", type=str, default=None, help="The device to train on.")
args = parser.parse_args()

MODEL_PREFIX = args.env
ENV_NAME, ENV_KWARGS, ENV_WRAPPERS, N_STEPS = ENVS[args.env]

# The environment setup here follows the architecture from the DQN paper. I.e.
# We downsample the observations to a 84x84 grayscale image, then stack the
# last 4 frames into a (4,84,84) tensor, which is then input to the
# convolutional neural network (CNN). The CNN consists of two convolutional
# layers separated by ReLU activations (see below).

ENV_WRAPPERS += [
    (gym.wrappers.ResizeObservation, {"shape": 84}),
    (gym.wrappers.GrayScaleObservation, {}),
    (gym.wrappers.FrameStack, {"num_stack": 4}),
]

print(f"Env: {ENV_NAME} (kwargs: {ENV_KWARGS})")
env = gym.make(ENV_NAME, **ENV_KWARGS)
for wrap, wrap_kwargs in ENV_WRAPPERS:
    env = wrap(env, **wrap_kwargs)
# Same for eval_env
eval_env = gym.make(ENV_NAME, **ENV_KWARGS)
for wrap, wrap_kwargs in ENV_WRAPPERS:
    eval_env = wrap(eval_env, **wrap_kwargs)

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
        "critic": model.to("cpu"),
        "env_name": ENV_NAME,
        "env_kwargs": ENV_KWARGS,
        "wrappers": ENV_WRAPPERS,
    }


output_critic = dqn(critic, env,
    save_hook=save_hook,
    eval_env=eval_env,
    n_steps=N_STEPS,
    device=args.device,
    batch_size=256,
    replay_size=500_000,
    eval_interval=50_000,
    lr=1e-4,
    exploration_rate=1.0,
    exploration_decay=1e-5,
    exploration_min=0.05,
)

print("Done")
