"""
This contains a crude implementation of DQN which is primarily inteded for
educational purposes. Spinning Up and Stable Baselines provide more detailed
and probably more efficient implementations.
"""

import copy
import os
import pickle
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from collections import deque
from datetime import datetime
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from .utils import ReplayMemory


# The policy in DQN is called an "epsilon-greedy" policy, which is that it
# greedily chooses the best action w.r.t. current estimated Q-function, but
# with a small epsilon chooses an action uniformly at random. This
# randomness is to allow for exploration to compensate for that the
# estimated Q-function is not very accurate early on.
@torch.no_grad()
def dqn_policy(q, state, epsilon=0.0, device="cpu"):
    # The PyTorch network usually operates on a batch, so we need to
    # unsqueeze out the batch dimension from the state before applying
    # it to the network. Then we remove the batch dimension with
    # squeeze.
    q.eval()
    qvals = q(torch.tensor(state, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()

    if random.random() < epsilon:
        # Choose a random action according to exploration rate
        action = random.randrange(len(qvals))
    else:
        # Choose greedily with argmax
        action = np.argmax(qvals)

    return action


@torch.no_grad()
def dqn_eval(q: nn.Module, env: gym.core.Env, runs=10, device="cpu"):
    """
    Evaluate the DQN policy, averaged over several runs.
    """
    total_return = 0.0
    for run in range(runs):
        episode_return = 0.0

        s, _ = env.reset()
        done = False

        while not done:
            a = dqn_policy(q, s, device=device)

            s_next, r, terminated, truncated, _ = env.step(a)

            episode_return += r
            done = terminated or truncated

            s = s_next

        total_return += episode_return

    return total_return / runs


def dqn(q: nn.Module, env: gym.core.Env,
        save_hook,
        # Training config
        n_steps=1_000_000,
        eval_interval=10_000,
        device="cpu",
        # Hyperparameters
        discount=0.99,
        exploration_rate=0.05,
        exploration_decay=0.0,
        lr=1e-3,
        batch_size=128,
        replay_size=100_000,
        target_update_freq=10):
    """
    The Q function shall be represented as a PyTorch module that acts on 
    Q: [B, obsdim*] -> [B, n_actions]
    """

    # We use tensorboard to log and plot progress
    spec_id = "".join(c for c in env.spec.id.replace("/", "-").lower() if c in "-" or c.isalnum())
    writer_path = os.path.join("_tensorboard", datetime.now().strftime(f"dqn_{spec_id}_%Y%m%d_%H%M%S"))
    print(f"Writing tensorboard logs to {writer_path}")
    writer = SummaryWriter(log_dir=writer_path)

    # Setup replay memory, this is a more optimized version of a deque useful
    # for reinforcement learning. We could also implement it like this:
    #
    #   from collections import deque
    #   replay = deque(maxlen=replay_size)
    #   ...
    #   when exploring:
    #       replay.add((s, a, r, s_next, terminated))
    #   ...
    #   when optimizing:
    #       minibatch = random.choices(replay, k=batch_size)
    #       bs = torch.stack([torch.tensor(s) for s, a, r, s_next, terminated in minibatch])
    #       ... etc
    #
    # But this ReplayMemory class avoids having to do concatenation and
    # conversions to tensors when we want to apply the batched data to a
    # network.
    replay = ReplayMemory(maxsize=replay_size, env=env)

    # Copy the Q-network into a target network
    q_targ = copy.deepcopy(q)

    # Move the networks to where we are going to train it.
    print(f"Using device: {device}")
    q = q.to(device)
    q_targ = q_targ.to(device)

    # Setup the optimizer for the Q-network
    q_opt = torch.optim.Adam(q.parameters(), lr=lr)

    s, _ = env.reset()

    all_returns = deque(maxlen=50)
    all_lengths = deque(maxlen=50)
    all_losses = deque([0], maxlen=1000)
    episode = 1
    episode_return = 0
    episode_length = 0

    # Record the best performing critic that we will return
    best_q = copy.deepcopy(q)
    best_return = None

    progress = trange(n_steps)
    progress_postfix = "Awaiting first episode..."
    progress.set_postfix_str(progress_postfix)
    for step in progress:

        a = dqn_policy(q, s, epsilon=exploration_rate, device=device)
        s_next, r, terminated, truncated, _ = env.step(a)

        replay.add(s, a, r, s_next, terminated)

        s = s_next

        episode_return += r
        episode_length += 1
        if terminated or truncated:
            all_returns.append(episode_return)
            all_lengths.append(episode_length)

            # Format statistics for printing
            avg_return = sum(all_returns) / len(all_returns)
            avg_eplen = sum(all_lengths) / len(all_lengths)
            avg_loss = sum(all_losses) / len(all_losses) if len(all_losses) > 0 else 0.0
            progress_postfix = " | ".join([
                f"Ep {episode}",
                f"Avg Return: {avg_return:.03f}",
                f"Avg EpLen: {avg_eplen:.02f}",
                f"Avg Loss: {avg_loss:.06f}",
                f"Best Eval: {best_return}",
            ])
            progress.set_postfix_str(progress_postfix)

            # Write to tensorboard, so we can see plots over the training
            writer.add_scalar("episode/return", episode_return, step)
            writer.add_scalar("episode/avg-q-loss", avg_loss, step)

            s, _ = env.reset()
            episode += 1
            episode_return = 0
            episode_length = 0
            all_losses.clear()

        # Evaluate the policy
        if step % eval_interval == 0:
            progress.set_postfix_str("Evaluating...")
            eval_return = dqn_eval(q, copy.deepcopy(env), device=device)
            if best_return is None or eval_return >= best_return:
                best_q = copy.deepcopy(q)
                best_return = eval_return
            progress.set_postfix_str(progress_postfix)
            # Save a checkpoint of the evaluated critic as well
            pfx, blob = save_hook(q)
            chkpath = os.path.join("_checkpoints", f"{pfx}-{step:06d}.pkl")
            os.makedirs(os.path.dirname(chkpath), exist_ok=True)
            with open(chkpath, "wb+") as f:
                pickle.dump(blob, f)

        # We also decay the exploration rate here
        exploration_rate *= 1.0 - exploration_decay

        if len(replay) >= batch_size:
            bs, ba, br, bs_next, bterm = replay.sample(batch_size, device=device)

            q.train() # Set the critic network into training mode

            q_opt.zero_grad() # reset the autograd state

            # Recall the definition of the Q-function that we want to learn:
            #
            #            { r(s,a)                      if s is terminal state
            #   Q(s,a) = {
            #            { r(s,a) + γ * Q(s', π(s'))   otherwise
            #
            # Where the policy π in this case maximizes the action over the Q
            # function. We compute the loss as the Mean-Squared Error between
            # the left-hand side and the right-hand side of the equation. But
            # for practical reasons we will only differentiate with respect to
            # the left-hand side, making this a "semi-gradient" optimization
            # algorithm.

            # Compute the loss of a single sample as
            # l = ((r + γ * max_a'[Q_targ(s', a')]) - Q(s,a))²

            v = torch.gather(q(bs), dim=1, index=ba.unsqueeze(-1)).squeeze(-1)

            # We only differentiate w.r.t. v (Q(s,a)) due to the semi-gradient
            # nature of the algorithm. So we explicitly turn off gradients for
            # the target network to avoid unnecessary computations.
            with torch.no_grad():
                v_targ = br + (1.0 - bterm) * discount * q_targ(bs_next).max(dim=1).values

            loss = (v_targ - v).square().mean()
            loss.backward()

            q_opt.step()

            all_losses.append(loss.item())

            # After a while has passed, we copy over the weights as per the DQN
            # algorithm. Here it is required that we disable autograd.
            if step % target_update_freq == 0:
                with torch.no_grad():
                    for p_targ, p in zip(q_targ.parameters(), q.parameters()):
                        p_targ.data.copy_(p.data)

    progress.close()
    print(f"Exiting DQN, best eval return: {best_return}")

    pfx, blob = save_hook(best_q)
    print(f"Saving model as {pfx}.pkl")
    with open(f"{pfx}.pkl", "wb+") as f:
        pickle.dump(blob, f)

    return best_q
