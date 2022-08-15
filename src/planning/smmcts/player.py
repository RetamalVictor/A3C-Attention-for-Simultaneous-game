from functools import partial
import abc
import math
import random
import torch


class Player(abc.ABC):
    """
    A generic player class.
    """
    def __init__(self, idd, nb_actions):
        self.idd = idd
        self.nb_actions = nb_actions

    @abc.abstractmethod
    def most_visited_action(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def best_action(self, nb_visits):
        raise NotImplementedError()

    @abc.abstractmethod
    def max_action(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def update_action_estimate(self, action, estimate):
        raise NotImplementedError()


class RandomPlayer(Player):
    """
    A random player class (used to 'ignore' other players move).
    """
    def __init__(self, idd, nb_actions):
        super().__init__(idd, nb_actions)
        self.rand = partial(random.randint, a=0, b=nb_actions - 1)

    def most_visited_action(self):
        return self.rand()

    def best_action(self, nb_visits):
        return self.rand()

    def max_action(self):
        return self.rand()

    def update_action_estimate(self, action, estimate):
        pass


class MCTSPlayer(Player):
    """
     A MCTS player that uses UCB score to select the best action
    """
    def __init__(self, idd, nb_actions, action_probs_estimate, exploration_coef, fpu, pw_c=None, pw_alpha=None):
        """

        :param idd:
        :param nb_actions:
        :param action_probs_estimate: the probability of actions (could be equal if the neural network is not used)
        :param exploration_coef:
        :param fpu:
        :param pw_c: Progressive Widening's c
        :param pw_alpha: Progressive Widening's alpha
        """
        super().__init__(idd, nb_actions)
        self.action_estimations = torch.zeros(nb_actions)
        self.nb_action_visits = torch.zeros(nb_actions)
        self.exploration_coef = exploration_coef
        self.fpu = fpu
        self.action_probs_estimate = action_probs_estimate
        self.use_progressive_widening = pw_alpha is not None and pw_c is not None
        if self.use_progressive_widening:
            """
            if progressive widening is used, then we should sort action according to their probs.
            """
            self.pw_alpha = pw_alpha
            self.pw_c = pw_c
            self.action_probs_estimate, indices = torch.sort(self.action_probs_estimate, dim=-1, descending=True)
            indices = indices.tolist()
            """
            Keep track of the original order
            """
            self.sorted_to_original_actions = {k: v for k, v in enumerate(indices)}
            self.original_to_sorted_actions = {v: k for k, v in self.sorted_to_original_actions.items()}

    def _compute_k(self, nb_visits):
        k = self.pw_c * (nb_visits ** self.pw_alpha)
        k = int(math.ceil(k))
        k = max(k, 1)
        return k

    def best_action(self, nb_visits):
        k = self._compute_k(nb_visits) if self.use_progressive_widening else self.nb_actions
        result = self.ucb(nb_visits, k)
        result = result.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def max_action(self):
        result = self.action_estimations.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def most_visited_action(self):
        result = self.nb_action_visits.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def ucb(self, nb_visits, k):
        probs = self.action_probs_estimate[:k]
        x, n = self.action_estimations[:k], self.nb_action_visits[:k]
        exploitation_term = x / n
        exploration_term = self.exploration_coef * torch.sqrt(math.log2(nb_visits) / n)
        # optionally, multiply by the probs
        # exploration_term = exploration_term * probs
        ucb = exploitation_term + exploration_term
        # when an action is not explored, assign fpu
        ucb = torch.nan_to_num(ucb, self.fpu, self.fpu, self.fpu)
        return ucb

    def update_action_estimate(self, action, estimate):
        if self.use_progressive_widening:
            "back to the original order"
            action = self.original_to_sorted_actions[action]
        self.action_estimations[action] += estimate
        self.nb_action_visits[action] += 1
