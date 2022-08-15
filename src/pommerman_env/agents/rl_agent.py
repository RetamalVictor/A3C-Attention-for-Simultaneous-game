#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Obada Aljabasini
# Last modification Date: 30/07/2022
#  - Victor Retamal --  Added documentation, Fixed formatting
# version ='1.0'
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
import torch.nn.functional as F

from pommerman_env.agents.pommerman_agent import PommermanAgent


class RLAgent(PommermanAgent):
    """
    Class containing a Reinforcement Learning guided agent.
    """

    def reset_agent(self):
        pass

    def __init__(self, agent_id, agent_model, stochastic=False):
        """
        Initialization of the Agent requires a model, and the id
        of the agent in the simulation. This Id wil be used to retrieve
        the information of the agent from the game.

        Parameterss:
        - agent_id      :     agent Id matching the one in the simulation for
                        the agent in training.
        - agent_model   :  The model to stimate the actions and values from
                        given input state.
        - stochastic    :   bool: if False, actions will be retrieved in a greedy
                        form. Otherwise, actions will be sampled from the
                        the action probability distribution.
        """
        super().__init__()
        self.agent_id = agent_id
        self.agent_model = agent_model
        self.stochastic = stochastic

    def act(self, obs):
        """
        Method to act in the simulation. Do not confuse with estimate.
        This method select an action given a state.

        Parameters:
        - obs:  state of the simulation. More on the shape information in the Pommerman
                official repository.

        returns:
        - agent_action: The action selected by the agent_model to be performed.
        """
        action_probs, _, _, _ = self.estimate(obs)
        action_probs = F.softmax(action_probs, dim=-1).view(-1)
        agent_action = (
            action_probs.argmax()
            if not self.stochastic
            else action_probs.multinomial(num_samples=1)
        )
        agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        """
        Estimation of probability distribution of actions given an input state

        Params:
        - obs: State in the simulation.
        
        returns:
        - action_probs: probability distribution of actions.
        """
        obs = obs[self.agent_id]
        # (1, 18, 11, 11)
        obs = obs.unsqueeze(0)
        return self.agent_model(obs)
