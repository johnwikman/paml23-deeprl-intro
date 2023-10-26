#!/usr/bin/env python3

import argparse
import random
import pickle

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from solution.dqn import dqn

ENVS = {
    "cartpole": ("CartPole-v1", {}, 300_000),
    "acrobot": ("Acrobot-v1", {}, 200_000),
    "lunarlander": ("LunarLander-v2", {}, 300_000),
}
parser = argparse.ArgumentParser("Train a DQN policy on a specified environment")
parser.add_argument("env", choices=ENVS.keys(), help="The environment to train on.")
args = parser.parse_args()

MODEL_PREFIX = args.env
ENV_NAME, ENV_KWARGS, N_STEPS = ENVS[args.env]

print(f"Env: {ENV_NAME} (kwargs: {ENV_KWARGS})")
env = gym.make(ENV_NAME, **ENV_KWARGS)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "critic": model,
        "env_name": ENV_NAME,
        "env_kwargs": ENV_KWARGS,
    }


output_critic = dqn(critic, env,
    save_hook=save_hook,
    n_steps=N_STEPS,
    device=device,
    batch_size=64,
    replay_size=150_000,
    lr=1e-3,
    exploration_rate=1.0,
    exploration_decay=1e-5,
)

print("Done")
