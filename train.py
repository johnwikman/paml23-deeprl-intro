#!/usr/bin/env python3

import argparse
import random
import pickle

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

import solution
from solution.dqn import dqn
from solution.modules import DiscretizeActions

ENVS = {
    "cartpole": ("CartPole-v1", {}, [], 300_000),
    "acrobot": ("Acrobot-v1", {}, [], 200_000),
    "lunarlander": ("LunarLander-v2", {}, [], 300_000),
    "mtncar": ("MountainCar-v0", {}, [], 300_000),
    # Continuous environments with discretized actions
    "pendulum": ("Pendulum-v1", {}, [(DiscretizeActions, {"action_map": np.linspace([-1.0], [1.0], 11)**3})], 200_000),
}
parser = argparse.ArgumentParser("Train a DQN policy on a specified environment")
parser.add_argument("env", choices=ENVS.keys(), help="The environment to train on.")
parser.add_argument("-d", "--device", dest="device", type=str, default=None, help="The device to train on.")
args = parser.parse_args()

MODEL_PREFIX = args.env
ENV_NAME, ENV_KWARGS, ENV_WRAPPERS, N_STEPS = ENVS[args.env]

print(f"Env: {ENV_NAME} (kwargs: {ENV_KWARGS})")
env = gym.make(ENV_NAME, **ENV_KWARGS)
for wrap, wrap_kwargs in ENV_WRAPPERS:
    env = wrap(env, **wrap_kwargs)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# [N, obs_dim] -> [N, n_actions]
critic = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_actions),
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
    n_steps=N_STEPS,
    device=args.device,
    batch_size=64,
    replay_size=150_000,
    lr=1e-3,
    exploration_rate=1.0,
    exploration_decay=1e-4,
    exploration_min=0.05,
)

print("Done")
