# partially inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
from pathlib import Path

import time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from planning_by_abstracting_over_opponent_models.learning.model.agent_loss import AgentLoss
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_env
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaper import strs_to_reward_shaper

torch.autograd.set_detect_anomaly(True)


def reshape_tensors_for_loss_func(steps,
                                  nb_opponents,
                                  nb_actions,
                                  opponent_log_probs,
                                  opponent_actions_ground_truths,
                                  device):
    # (nb_steps, nb_opponents, nb_actions)
    opponent_log_probs = torch.stack(opponent_log_probs)
    assert opponent_log_probs.shape == (
        steps, nb_opponents,
        nb_actions), f"{opponent_log_probs.shape} != {(steps, nb_opponents, nb_actions)}"
    # (nb_opponents, nb_steps, nb_actions)
    opponent_log_probs = opponent_log_probs.permute(1, 0, 2)
    assert opponent_log_probs.shape == (
        nb_opponents, steps,
        nb_actions), f"{opponent_log_probs.shape} != {(nb_opponents, steps, nb_actions)}"
    # (nb_steps, nb_opponents)
    opponent_actions_ground_truths = torch.stack(opponent_actions_ground_truths).to(device)
    assert opponent_actions_ground_truths.shape == (
        steps, nb_opponents), f"{opponent_actions_ground_truths.shape} != {(steps, nb_opponents)}"
    # (nb_opponents, nb_steps)
    opponent_actions_ground_truths = opponent_actions_ground_truths.permute(1, 0)
    assert opponent_actions_ground_truths.shape == (
        nb_opponents, steps), f"{opponent_actions_ground_truths.shape} != {(nb_opponents, steps)}"
    opponent_actions_ground_truths = opponent_actions_ground_truths.to(device)

    return opponent_log_probs, opponent_actions_ground_truths


def collect_trajectory(env,
                       state,
                       lock,
                       counter,
                       agents,
                       nb_opponents,
                       nb_actions,
                       nb_steps,
                       device,
                       reward_shaper=None):
    agent_rewards = []
    agent_values = []
    agent_log_probs = []
    agent_entropies = []
    opponent_log_probs = []
    opponent_actions_ground_truths = []
    agent = agents[0]
    done = False
    steps = 0
    running_reward = 0.0
    while not done and steps < nb_steps:
        steps += 1
        obs = env.get_features(state).to(device)
        agent_policy, agent_value, opponent_log_prob, opponent_influence = agent.estimate(obs)
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
    opponent_trajectory = reshape_tensors_for_loss_func(steps,
                                                        nb_opponents,
                                                        nb_actions,
                                                        opponent_log_probs,
                                                        opponent_actions_ground_truths,
                                                        device)
    return steps, state, done, running_reward, agent_trajectory, opponent_trajectory


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank,
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
          device):
    combined_opponent_classes = ",".join(opponent_classes)
    agents, env = create_env(rank,
                             seed,
                             use_cython,
                             model_spec,
                             nb_actions,
                             nb_opponents,
                             opponent_classes,
                             device,
                             train=True)
    agent_model = agents[0].agent_model
    state = env.reset()
    # RL
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    gae_lambda = 0.95
    opponent_coefs = [0.05] * nb_opponents
    criterion = AgentLoss(gamma=gamma,
                          value_loss_coef=value_loss_coef,
                          entropy_coef=entropy_coef,
                          gae_lambda=gae_lambda).to(device)
    reward_shaper = strs_to_reward_shaper(reward_shapers)
    reward_shaper.reset()
    episodes = 0
    episode_batches = 0
    running_total_loss = 0.0
    running_agent_policy_loss = 0.0
    running_agent_value_loss = 0.0
    running_opponent_policy_loss = 0.0
    running_reward = 0.0
    start_time = time.time()
    while True:
        # sync with the shared model
        agent_model.load_state_dict(shared_model.state_dict())
        steps, state, done, reward, agent_trajectory, opponent_trajectory = collect_trajectory(env=env,
                                                                                               state=state,
                                                                                               lock=lock,
                                                                                               counter=counter,
                                                                                               agents=agents,
                                                                                               nb_opponents=nb_opponents,
                                                                                               nb_actions=nb_actions,
                                                                                               nb_steps=nb_steps,
                                                                                               device=device,
                                                                                               reward_shaper=reward_shaper)
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
            opponent_coefs)
        total_loss = agent_policy_loss + agent_value_loss
        if include_opponent_loss:
            total_loss = total_loss + opponent_policy_loss
        total_loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        ensure_shared_grads(agent_model, shared_model)
        optimizer.step()

        running_total_loss += total_loss.item()
        running_agent_policy_loss += agent_policy_loss.item()
        running_agent_value_loss += agent_value_loss.item()
        running_opponent_policy_loss += opponent_policy_loss.item()
        running_reward += reward
        episode_batches += 1

        
        if done:

            #Tensorboard logs
            writer= SummaryWriter(log_dir="/home/victo/pommerman/obada/SM-MCTS/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/tensorboard")

            writer.add_scalar('Reward', running_reward,episodes)

            print(f"Episode time {time.time() - start_time}")
            episodes += 1
            episode_batches = 0
            running_total_loss = 0.0
            running_agent_policy_loss = 0.0
            running_agent_value_loss = 0.0
            running_opponent_policy_loss = 0.0
            running_reward = 0.0
