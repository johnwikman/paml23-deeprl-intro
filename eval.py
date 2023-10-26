#!/usr/bin/env python3

import argparse
import pickle
import random

import numpy as np
import gymnasium as gym
import torch

from solution.dqn import dqn_policy

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the saved model to evaluate")
parser.add_argument("-r", "--render", dest="render", action="store_true")

args = parser.parse_args()

with open(args.path, "rb") as f:
    d = pickle.load(f)
    critic = d["critic"]
    ENV_NAME = d["env_name"]
    ENV_KWARGS = d.get("env_kwargs", {})

env = gym.make(ENV_NAME, render_mode="human", **ENV_KWARGS)
device = "cuda" if torch.cuda.is_available() else "cpu"

critic.to(device)

episode_return = 0.0
episode_steps = 0

s, _ = env.reset()

while True:
    if args.render:
        env.render()

    a = dqn_policy(critic, s, device=device)
    s_next, r, terminated, truncated, _ = env.step(a)

    episode_return += r
    episode_steps += 1
    if terminated or truncated:
        break

    s = s_next

env.close()
print("Episode totals:")
print(f" * Return: {episode_return}")
print(f" * Steps: {episode_steps}")
