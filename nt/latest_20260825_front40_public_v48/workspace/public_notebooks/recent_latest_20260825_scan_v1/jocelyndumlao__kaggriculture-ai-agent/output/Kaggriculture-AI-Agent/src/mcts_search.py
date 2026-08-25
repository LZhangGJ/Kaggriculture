"""
mcts_search.py

Simplified Monte Carlo Tree Search Engine.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import random

from src.mcts import MCTSNode


class MCTSSearch:
    """
    Simplified Monte Carlo Tree Search.
    """

    def __init__(self):
        self.root = MCTSNode()

    # -----------------------------------------------------

    def expand(self, actions):
        """
        Expand the root node.
        """

        for action in actions:
            self.root.add_child(action)

    # -----------------------------------------------------

    def simulate(self, node):
        """
        Simulate a random reward.
        """

        reward = random.uniform(0.0, 100.0)

        node.update(reward)

        return reward

    # -----------------------------------------------------

    def backpropagate(self, node, reward):
        """
        Propagate reward to the root.
        """

        current = node

        while current is not None:

            current.update(reward)

            current = current.parent

    # -----------------------------------------------------

    def run(self, actions):
        """
        Execute one MCTS iteration.
        """

        self.expand(actions)

        if not self.root.children:
            return None

        node = random.choice(self.root.children)

        reward = self.simulate(node)

        self.backpropagate(node.parent, reward)

        return node

    # -----------------------------------------------------

    def best_action(self):
        """
        Return the most visited child.
        """

        if not self.root.children:
            return None

        return max(
            self.root.children,
            key=lambda node: node.visits,
        )