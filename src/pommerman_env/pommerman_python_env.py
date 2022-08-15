#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Obada Aljabasini
# Last modification Date: 22/07/2022
#  - Victor Retamal --  Added documentation, Fixed formatting and bugs in
#                        getters with the core Cython env
# version ='1.0'
# ---------------------------------------------------------------------------
""" The module is the core to initialize Cython environment for the 
    Pommerman competition. Inspired from:
    https://github.com/tambetm/pommerman-baselines/tree/master/cython_env """
# ---------------------------------------------------------------------------

from typing import List
import pommerman

from pommerman_env.pommerman_base_env import PommermanBaseEnv
from pommerman_env.agents.pommerman_agent import PommermanAgent


class PommermanPythonEnv(PommermanBaseEnv):

    """
    With this class, the Python version of the Pommerman env is initialized.
    Its core functionality makes use of the pommerman module. Installation
    and troubleshooting in the github repository or the report.
    """

    def __init__(
        self,
        agents: List[PommermanAgent],
        seed,
        training_agent=0,
        mode="PommeFFACompetition-v0",
    ):
        """
        The init method initializes the environment.
        Getters are not described as they return the object or variable
        in the name.
        Render and Reset follow the same logic as every Gym environment.

        Parameters:
        - agent : List containing the initialized agents
        - seed  : int
        - training_agent: the agent that will be trained. By default, the
                first agent of the list.
        - mode  : The game mode. By default Free For All.
        """
        super().__init__(len(agents))
        self.nb_players = len(agents)
        self.env = pommerman.make(mode, agents)
        self.env.seed(seed)
        self.env.set_training_agent(training_agent)
        self.action_space = self.env.action_space

    def get_observations(self):
        """This methods returns the state space of the current
        step, formated as described in the Pommerman docs.

        Return:
        - state: State observation coming from the Pommerman environment.
        """
        obs = self.env.get_observations()
        return obs

    def get_done(self):
        """
        Return:
        - done: bool
        """
        return self.env._get_done()

    def get_rewards(self):
        """
        Return:
        - rewards: lists of rewards for every agent.
        """
        rewards = self.env._get_rewards()
        rewards = self.transform_rewards(rewards)
        return rewards

    def reset(self):
        self.env.reset()
        return self.get_observations()

    def step(self, actions):
        """
        This method advance further in the game simulation.
        Internally, calls the core env and stores
        all the information in the core env.
        Its different from the act method.

        Parameters:
        - actions: List of actions.

        Returns:
        - state     : State observation coming from the Pommerman environment.
        - rewards   : List
        - done      : bool
        """
        state, rewards, done, _ = self.env.step(actions)
        rewards = self.transform_rewards(rewards)
        return state, rewards, done

    def act(self, state):
        """
        Return:
        - actions: List with the action of every agent.
        """
        return self.env.act(state)

    def render(self, mode=None):
        if mode is None:
            mode = "human"
        return self.env.render(mode)

    def get_game_state(self):
        return self.env.get_json_info()

    def set_game_state(self, game_state):
        """
        This method allows to intialize the game from a fixed
        state
        """
        self.env._init_game_state = game_state
        self.env.set_json_info()
