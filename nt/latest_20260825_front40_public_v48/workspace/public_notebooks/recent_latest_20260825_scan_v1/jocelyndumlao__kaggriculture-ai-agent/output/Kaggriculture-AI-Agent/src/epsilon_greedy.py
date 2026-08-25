"""
epsilon_greedy.py

Epsilon-Greedy Policy for the Kaggriculture AI Agent.

Balances exploration and exploitation.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import random


class EpsilonGreedy:
    """
    Standard epsilon-greedy action selector.
    """

    def __init__(self, epsilon: float = 0.10):
        self.epsilon = epsilon

    # ---------------------------------------------------------

    def choose(self, q_table, state, actions):
        """
        Select an action.

        With probability epsilon:
            choose a random action.

        Otherwise:
            choose the best known action.
        """

        if not actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(actions)

        best = q_table.best_action(state)

        if best is None:
            return random.choice(actions)

        return best

    # ---------------------------------------------------------

    def set_epsilon(self, epsilon: float):
        """
        Update epsilon.
        """

        self.epsilon = epsilon

    # ---------------------------------------------------------

    def decay(self, factor: float = 0.99):
        """
        Decay epsilon after each episode.
        """

        self.epsilon *= factor

        if self.epsilon < 0.01:
            self.epsilon = 0.01