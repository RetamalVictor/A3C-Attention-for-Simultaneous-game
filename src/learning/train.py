#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Obada Aljabasini
# Last modification Date: 28/07/2022
#  - Victor Retamal --  Added documentation, fixed imports and saving paths
# version ='1.0'
# ---------------------------------------------------------------------------
""" This script contain the functions to train a model in the Pommerman 
    environment. It contains two main functions:
    - Collect trajectory: This function send actions to env and collect the
    information from it. Every call to this function runs a full episode until
    done or until the max number of steps are reach.
    - Train: This functions is the main training loop.   """
# ---------------------------------------------------------------------------
# partially inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

from pathlib import Path

import time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from learning.model.agent_loss import AgentLoss
from learning.pommerman_env_utils import create_env
from learning.reward_shaping.reward_shaper import strs_to_reward_shaper

torch.autograd.set_detect_anomaly(True)


def reshape_tensors_for_loss_func(
    steps,
    nb_opponents,
    nb_actions,
    opponent_log_probs,
    opponent_actions_ground_truths,
    device,
):

    """
    Auxiliar function to reshape tensor for loss calculation.
    The Shapes are specified in code comments.

    Parameters:
    :steps: int, number of steps in the simulation.
    :nb_opponents  : int with the number of opponents 1 or 3.
    :nb_actions    : int action space.
    :opponent_log_prob: Tensor, probability distribution of the actions predicted
                        by the opponents models.
    :opponent_actions_ground_truths: real actions of the opponent in the game.
    :device        : device used for training.
    
    Returns:
    :opponent_log_probs: shape(nb_opponents, nb_steps, nb_actions)
    :opponent_actions_ground_truths: shape(nb_opponents, nb_steps)
    """
    # (nb_steps, nb_opponents, nb_actions)
    opponent_log_probs = torch.stack(opponent_log_probs)
    assert opponent_log_probs.shape == (
        steps,
        nb_opponents,
        nb_actions,
    ), f"{opponent_log_probs.shape} != {(steps, nb_opponents, nb_actions)}"

    # (nb_opponents, nb_steps, nb_actions)
    opponent_log_probs = opponent_log_probs.permute(1, 0, 2)
    assert opponent_log_probs.shape == (
        nb_opponents,
        steps,
        nb_actions,
    ), f"{opponent_log_probs.shape} != {(nb_opponents, steps, nb_actions)}"

    # (nb_steps, nb_opponents)
    opponent_actions_ground_truths = torch.stack(opponent_actions_ground_truths).to(
        device
    )
    assert opponent_actions_ground_truths.shape == (
        steps,
        nb_opponents,
    ), f"{opponent_actions_ground_truths.shape} != {(steps, nb_opponents)}"

    # (nb_opponents, nb_steps)
    opponent_actions_ground_truths = opponent_actions_ground_truths.permute(1, 0)
    assert opponent_actions_ground_truths.shape == (
        nb_opponents,
        steps,
    ), f"{opponent_actions_ground_truths.shape} != {(nb_opponents, steps)}"
    opponent_actions_ground_truths = opponent_actions_ground_truths.to(device)

    return opponent_log_probs, opponent_actions_ground_truths


def collect_trajectory(
    env,
    state,
    lock,
    counter,
    agents,
    nb_opponents,
    nb_actions,
    nb_steps,
    device,
    reward_shaper=None,
):

    """
    Collect trajectory runs an episode in the simulation and collects
    the information from the episode.
    Parameters:
    :env: Gym environment.
    :states        : The initial state for the simulation.
    :lock          : The lock for multiprocessing purposes.
    :counter       : Episode counter for multiprocessing purposes.
    :agents        : List of agents to play the game in the Gym environment. More
                        information in the Pommerman repository.
    :nb_opponents  : int with the number of opponents 1 or 3.
    :nb_actions    : int action space.
    :nb_steps      : int max number of steps per simulation.
    :device        : device used for training.
    :reward_shaper : Reward_shaper object to collect rewards.

    Returns:
    :steps         : int number of steps in the game.
    :state         : The last state of the game.
    :done          : bool flag for game finished.
    :running_reward: The accumulated reward collected during the game.
    :agent_trajectory: Tuple containing (agent_rewards, agent_values, agent_log_probs, agent_entropies)
    :opponent_trajectory: Tuple containing (opponent_log_probs, opponent_actions_ground_truths)
    :opponent_influences: Pytorch tensor; Attention values for every opponent.
    """
    agent_rewards = []
    agent_values = []
    agent_log_probs = []
    agent_entropies = []
    opponent_log_probs = []
    opponent_actions_ground_truths = []
    opponent_influences = []
    agent = agents[0]
    done = False
    steps = 0
    running_reward = 0.0

    # Game loop
    while not done and steps < nb_steps:
        steps += 1
        obs = env.get_features(state).to(device)
        (
            agent_policy,
            agent_value,
            opponent_log_prob,
            opponent_influence,
        ) = agent.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_log_prob = F.log_softmax(agent_policy, dim=-1)
        agent_entropy = -(agent_log_prob * agent_prob).sum(1, keepdim=True)
        agent_action = agent_prob.multinomial(num_samples=1).detach()
        agent_log_prob = agent_log_prob.gather(1, agent_action)
        opponent_actions = env.act(state)
        agent_action = agent_action.item()
        actions = [agent_action, *opponent_actions]
        state, rewards, done = env.step(actions)
        agent_reward = rewards[0]
        if reward_shaper is not None and not done and agent_reward == 0:
            agent_reward = reward_shaper.shape(state[0], agent_action)
        running_reward += agent_reward
        with lock:
            counter.value += 1

        # agent
        agent_rewards.append(agent_reward)
        agent_entropies.append(agent_entropy)
        agent_log_probs.append(agent_log_prob)
        agent_values.append(agent_value)

        # opponents
        opponent_log_probs.append(opponent_log_prob.squeeze(0))
        opponent_actions = torch.LongTensor(opponent_actions)
        opponent_actions_ground_truths.append(opponent_actions)
        opponent_influences.append(opponent_influence)
    if done:
        state = env.reset()
        reward_shaper.reset()
        r = torch.zeros(1, 1, device=device)
    else:
        obs = env.get_features(state).to(device)
        _, agent_value, _, _ = agent.estimate(obs)
        r = agent_value.detach()
    r = r.to(device)
    agent_values.append(r)

    agent_trajectory = (agent_rewards, agent_values, agent_log_probs, agent_entropies)
    opponent_trajectory = reshape_tensors_for_loss_func(
        steps,
        nb_opponents,
        nb_actions,
        opponent_log_probs,
        opponent_actions_ground_truths,
        device,
    )
    return (
        steps,
        state,
        done,
        running_reward,
        agent_trajectory,
        opponent_trajectory,
        opponent_influences,
    )


def ensure_shared_grads(model, shared_model):
    """
    Auxiliart function to share gradients in multiprocessing.
    Params:
    :model         : Pytorch based model.
    :shared_model  : pytorch based model.
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(
    rank,
    seed,
    use_cython,
    shared_model,
    optimizer,
    counter,
    lock,
    model_spec,
    nb_steps,
    nb_actions,
    nb_opponents,
    opponent_classes,
    reward_shapers,
    max_grad_norm,
    include_opponent_loss,
    device,
    SAVING_PATH,
):
    """
    Training function.
    1.- Creates environment and agent
    2.- Specify Loss
    3.- Specify reward shaper
    Main Loop
    4.- Collect trajectory
    5.- Compute backward pass
    6.- Reset

    Params:
    :rank: int, auxliar value to set seed
    :seed: int, seed value for training
    :use_cython    : bool, if true, the backend for the simulation runs Cython.
    :model_spec    : Dict(), Params required to initialize the model.
    :nb_opponents  : int with the number of opponents 1 or 3.
    :nb_actions    : int action space.
    :opponent_classes: List[str,str,...] Specification for agent classes
    :reward_shapers  : List[str,str,...] Specification for rewards
    :max_grad_norm   : float, max gradient allowed
    :include_opponent_loss: if true, opponent models loss is included.
    :device        : device used for training.
    :SAVING_PATH: path to save results in .txt
    """

    # Env and agent creation
    agents, env = create_env(
        rank,
        seed,
        use_cython,
        model_spec,
        nb_actions,
        nb_opponents,
        opponent_classes,
        device,
        train=True,
    )
    agent_model = agents[0].agent_model

    # RL Loss definition
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    gae_lambda = 0.95
    opponent_coefs = [0.05] * nb_opponents
    criterion = AgentLoss(
        gamma=gamma,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        gae_lambda=gae_lambda,
    ).to(device)
    reward_shaper = strs_to_reward_shaper(reward_shapers)
    reward_shaper.reset()
    episodes = 0
    episode_batches = 0
    running_total_loss = 0.0
    running_agent_policy_loss = 0.0
    running_agent_value_loss = 0.0
    running_opponent_policy_loss = 0.0
    running_reward = 0.0

    # Main Loop
    state = env.reset()
    while True:
        # sync with the shared model
        agent_model.load_state_dict(shared_model.state_dict())
        (
            steps,
            state,
            done,
            reward,
            agent_trajectory,
            opponent_trajectory,
            opponent_influence,
        ) = collect_trajectory(
            env=env,
            state=state,
            lock=lock,
            counter=counter,
            agents=agents,
            nb_opponents=nb_opponents,
            nb_actions=nb_actions,
            nb_steps=nb_steps,
            device=device,
            reward_shaper=reward_shaper,
        )
        agent_rewards, agent_values, agent_log_probs, agent_entropies = agent_trajectory
        opponent_log_probs, opponent_actions_ground_truths = opponent_trajectory

        # backward step
        optimizer.zero_grad()
        agent_policy_loss, agent_value_loss, opponent_policy_loss = criterion(
            agent_rewards,
            agent_log_probs,
            agent_values,
            agent_entropies,
            opponent_log_probs,
            opponent_actions_ground_truths,
            opponent_coefs,
        )
        total_loss = agent_policy_loss + agent_value_loss
        if include_opponent_loss:
            total_loss = total_loss + opponent_policy_loss
        total_loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        ensure_shared_grads(agent_model, shared_model)
        optimizer.step()

        # Storing running values
        running_total_loss += total_loss.item()
        running_agent_policy_loss += agent_policy_loss.item()
        running_agent_value_loss += agent_value_loss.item()
        running_opponent_policy_loss += opponent_policy_loss.item()
        running_reward += reward
        episode_batches += 1
        with open(SAVING_PATH, "a") as f:
            f.write(f"{reward},{opponent_policy_loss.item()},{steps},{total_loss.item()}\n")
            f.close()

        if done:
            episodes += 1
            episode_batches = 0
            running_total_loss = 0.0
            running_agent_policy_loss = 0.0
            running_agent_value_loss = 0.0
            running_opponent_policy_loss = 0.0
            running_reward = 0.0
