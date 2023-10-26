#!/usr/bin/env python3

import argparse
import random
import pickle

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

import solution

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Model to tranfer to cpu")
parser.add_argument("-o", "--output-file", dest="output_file", type=str, default=None, help="Name of converted file.")
args = parser.parse_args()

if args.output_file is None:
    if args.file.endswith(".pkl"):
        args.output_file = f"{args.file[:-4]}.cpu.pkl"
    else:
        args.output_file = args.file + ".cpu.pkl"

with open(args.file, "rb") as f:
    d = pickle.load(f)

d["critic"] = d["critic"].to("cpu")

print(f"{args.file} -> {args.output_file}")
with open(args.output_file, "wb+") as f:
    d = pickle.dump(d, f)
print("done")
