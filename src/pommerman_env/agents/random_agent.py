from pommerman_env.agents.pommerman_agent import PommermanAgent


class RandomAgent(PommermanAgent):
    def reset_agent(self):
        pass

    def act(self, obs, action_space):
        return action_space.sample()