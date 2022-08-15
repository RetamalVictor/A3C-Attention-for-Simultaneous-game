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

import random
from typing import List

import numpy as np
import cpommerman
import pommerman

from pommerman_env.agents.pommerman_agent import PommermanAgent
from pommerman_env.pommerman_base_env import PommermanBaseEnv


class PommermanCythonEnv(PommermanBaseEnv):
    """
    With this class, the Cython version of the Pommerman env is initialized.
    Its core functionality makes use of the cpommerman module. Installation
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
        The init method initializes two environments. The Cpommerman
        and the python Pommerman. The computations are done in the
        Cython environment, while the python one is for rendering.

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
        self._seed = seed
        self.seed(self._seed)
        self.agents = agents
        self.env_render = pommerman.make(mode, agents)
        self.env = cpommerman.make()

        # Setting the training agent from the Cpommerman API
        if 0 <= training_agent <= 3:
            self.env.set_training_agent(training_agent)
        self.action_space = self.env.get_action_space()

    @staticmethod
    def seed(s):
        """
        This method ensures a common shared seed for the env.
        """
        np.random.seed(s)
        random.seed(s)

    def step(self, actions):
        """
        This method advance further in the game simulation.
        Internally, calls the core env and stores
        all the information in the core env.
        Its different from the act method.

        Parameters:
        - actions: List of actions.

        Returns:
        - obs       : State observation coming from the Pommerman environment.
        - rewards   : List
        - done      : bool
        """
        actions = np.asarray(actions).astype(np.uint8)
        self.env.step(actions)
        return self.get_observations(), self.get_rewards(), self.get_done()

    def act(self, state):
        """
        This method calls for the act method in every agent. It
        will return the selected action of the agents.

        Parameters:
        - state: State observation coming from the Pommerman environment.

        Return:
        - agent_actions: List containing the actions.
        """
        agent_actions = []
        for i in range(1, self.nb_players):
            agent_actions.append(self.agents[i].act(state[i], self.action_space))
        return agent_actions

    def reset(self):
        self.seed(self._seed)
        self.env.reset()
        for agent in self.agents:
            agent.reset_agent()
        return self.get_observations()

    def render(self, mode=None):
        if mode is None:
            mode = "human"
        self.env_render._init_game_state = self.env.get_json_info()
        self.env_render.set_json_info()
        return self.env_render.render(mode)

    def get_observations(self):
        """
        This methods returns the state space of the current
        step, formated as described in the Pommerman docs.

        Return:
        - state: State observation coming from the Pommerman environment.
        """
        obs = self.env.get_observations()
        step_count = self.env.get_step_count()
        for ob in obs:
            ob["step_count"] = step_count
        return obs

    def get_done(self):
        """
        Return:
        - done: bool
        """
        return self.env.get_done()

    def get_rewards(self):
        return self.transform_rewards(self.env.get_rewards())

    def get_game_state(self):
        return self.env.get_state()

    def set_game_state(self, game_state):
        self.env.set_state(game_state)
