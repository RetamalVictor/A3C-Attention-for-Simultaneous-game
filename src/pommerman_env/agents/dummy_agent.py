from pommerman_env.agents.pommerman_agent import PommermanAgent


class DummyAgent(PommermanAgent):
    def reset_agent(self):
        pass

    def act(self, obs, action_space):
        pass
