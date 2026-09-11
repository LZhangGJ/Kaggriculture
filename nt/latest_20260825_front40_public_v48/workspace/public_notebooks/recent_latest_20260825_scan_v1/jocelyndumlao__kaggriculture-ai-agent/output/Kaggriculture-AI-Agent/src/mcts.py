"""
mcts.py

Monte Carlo Tree Search (MCTS)

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import math


class MCTSNode:
    """
    One node in the Monte Carlo Tree.
    """

    def __init__(self, action=None, parent=None):

        self.action = action
        self.parent = parent

        self.children = []

        self.visits = 0
        self.reward = 0.0

    # -----------------------------------------------------

    def add_child(self, action):

        child = MCTSNode(
            action=action,
            parent=self,
        )

        self.children.append(child)

        return child

    # -----------------------------------------------------

    def update(self, reward):

        self.visits += 1
        self.reward += reward

    # -----------------------------------------------------

    def average_reward(self):

        if self.visits == 0:
            return 0.0

        return self.reward / self.visits

    # -----------------------------------------------------

    def ucb1(self, exploration=1.414):

        if self.visits == 0:
            return float("inf")

        if self.parent is None:
            return self.average_reward()

        return (
            self.average_reward()
            + exploration
            * math.sqrt(
                math.log(self.parent.visits)
                / self.visits
            )
        )