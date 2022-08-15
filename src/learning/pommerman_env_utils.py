#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Obada Aljabasini
# Last modification Date: 28/07/2022
#  - Victor Retamal --  Added documentation, fixed imports and saving paths
# version ='1.0'
# ---------------------------------------------------------------------------
""" Auxiliar funtions to create agents and environments """
# ---------------------------------------------------------------------------

from learning.model.agent_model import create_agent_model
from pommerman_env.agents.cautious_agent import CautiousAgent
from pommerman_env.agents.simple_agent import SimpleAgent
from pommerman_env.agents.random_agent import RandomAgent
from pommerman_env.agents.rl_agent import RLAgent
from pommerman_env.agents.smart_random_agent import (
    SmartRandomAgent,
    SmartRandomAgentNoBomb,
)
from pommerman_env.agents.static_agent import StaticAgent
from pommerman_env.pommerman_base_env import PommermanBaseEnv
#from pommerman_env.pommerman_cython_env import PommermanCythonEnv
from pommerman_env.pommerman_python_env import PommermanPythonEnv


def str_to_agent(classes):
    """
    Auxiliar fuction to map a list of strings to agent types.
    Params:
    - classes: list[str, str,...] List containing agent types.

    returns:
    - List[Agent, Agent,...]
    """
    d = {
        "static": StaticAgent,
        "random": RandomAgent,
        "smart_no_bomb": SmartRandomAgentNoBomb,
        "smart": SmartRandomAgent,
        "simple": SimpleAgent,
        "cautious": CautiousAgent,
    }
    return d[classes.strip().lower()]


def create_env(
    rank,
    seed,
    use_cython,
    model_spec,
    nb_actions,
    nb_opponents,
    opponent_classes,
    device,
    train=True,
):
    """
    Creates the environment to play the game.
    Since it requires a list of agents, it also create the agents.

    Params:
    - rank: int, auxliar value to set seed
    - seed: int, seed value for training
    - use_cython    : bool, if true, the backend for the simulation runs Cython.
    - model_spec    : Dict(), Params required to initialize the model.
    - nb_opponents  : int with the number of opponents 1 or 3.
    - nb_actions    : int action space.
    - opponent_classes: List[str,str,...] Specification for agent classes
    - device        : device used for training.
    - train         : Set the model to train mode.

    Returns:
    - agents: List[Agent, Agent,...] List with agents playing the game.
    - env   : Pommerman Env.
    """
    agent_model = create_agent_model(
        rank=rank,
        seed=seed,
        nb_actions=nb_actions,
        nb_opponents=nb_opponents,
        device=device,
        train=train,
        **model_spec
    )
    agent = RLAgent(0, agent_model)
    agents = [str_to_agent(opponent_class)() for opponent_class in opponent_classes]
    agents.insert(0, agent)
    r = seed + rank
    env: PommermanBaseEnv = (
        PommermanCythonEnv(agents, r) if use_cython else PommermanPythonEnv(agents, r)
    )
    return agents, env
