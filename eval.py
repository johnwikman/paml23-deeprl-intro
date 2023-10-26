#!/usr/bin/env python3

import argparse
import pickle
import random

import numpy as np
import gymnasium as gym
import torch

import solution
from solution.dqn import dqn_policy

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the saved model to evaluate")
parser.add_argument("-r", "--render", dest="render", action="store_true")
parser.add_argument("-d", "--device", dest="device", type=str, default="cpu", help="The device to evaluate on.")

args = parser.parse_args()

with open(args.path, "rb") as f:
    d = pickle.load(f)
    critic = d["critic"]
    ENV_NAME = d["env_name"]
    ENV_KWARGS = d.get("env_kwargs", {})
    if "render_mode" in ENV_KWARGS:
        del ENV_KWARGS["render_mode"]

env = gym.make(ENV_NAME, render_mode="human", **ENV_KWARGS)

# Apply any provided wrappers to the environment
for wrap, wrap_kwargs in d.get("wrappers", []):
    env = wrap(env, **wrap_kwargs)


critic.to(args.device)


episode_return = 0.0
episode_steps = 0

s, _ = env.reset()

while True:
    if args.render:
        env.render()

    a = dqn_policy(critic, s, device=args.device)
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
