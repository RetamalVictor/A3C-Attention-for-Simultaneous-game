# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

from learning.reward_shaping.reward_shaping_component import RewardShapingComponent


class AmmoUsageComponent(RewardShapingComponent):
    """
    Class implementaiton of the reward component given the usage of Ammo.

    It creates a counter checking then steps where the player don't use the ammo.
    After 20 steps, if the agent has not used the ammo it yields a negative reward by default.
    """

    def __init__(self, not_using_ammo_reward=-0.0001):
        super().__init__()
        self.not_using_ammo_reward = not_using_ammo_reward
        self.not_using_ammo_counter = 0

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            if curr_state["ammo"] == self.prev_state["ammo"]:
                self.not_using_ammo_counter += 1
            else:
                self.not_using_ammo_counter = 0
            if self.not_using_ammo_counter >= 20:
                return self.not_using_ammo_reward
        return 0

    def reset(self):
        super().reset()
        self.not_using_ammo_counter = 0
