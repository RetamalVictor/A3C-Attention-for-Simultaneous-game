import argparse
import os, sys
from pickle import NONE
import time
from random import randint

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count

import sys
PATH = "/home/victo/pommerman/obada/SM-MCTS/planning-by-abstracting-over-opponent-models"
sys.path.append(PATH)

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.model.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import str_to_agent
from planning_by_abstracting_over_opponent_models.planning.policy_estimator.neural_network_policy_estimator import \
    NeuralNetworkPolicyEstimator
from planning_by_abstracting_over_opponent_models.planning.policy_estimator.uniform_policy_estimator import \
    UniformPolicyEstimator
from planning_by_abstracting_over_opponent_models.planning.smmcts.smmcts import SMMCTS
from planning_by_abstracting_over_opponent_models.planning.value_estimator.random_rollout_value_estimator import \
    RandomRolloutValueEstimator
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.dummy_agent import DummyAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv



if __name__ == '__main__':

    NB_PROCESSES = 1
    MULTIPROCESSING = 'store_true'
    NO_MULTIPROCESSING = 'store_false'
    NB_GAMES = 50
    NB_PLAYS = 1
    NB_PLAYERS = 4
    SS = "random, random, random"
    #SS = ["random","random","random"]
    IGNORE_OPP_ACTIONS = 'store_true'
    SEARCH_OPP_ACTIONS = 'store_false'
    MCTS_ITERATIONS = 5009
    MODEL_ITERATIONS = 25
    EXPLORATION_COEF = 0.001
    FPU = None
    PW_C = None
    PW_ALPHA = None
    POLICY_ESTIMATION = 'uniform'
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    NB_ACTIONS = 6
    COMBINED_OPP_CLASSES = ",".join(SS)
    OPP_CLASSES = [str_to_agent(cl) for cl in SS]
    EXP_COEFS = EXPLORATION_COEF * (NB_PLAYERS-1)
    