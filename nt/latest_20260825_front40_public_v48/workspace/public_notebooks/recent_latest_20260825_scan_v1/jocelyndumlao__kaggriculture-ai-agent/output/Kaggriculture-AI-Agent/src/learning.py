"""
learning.py

Simple reinforcement-style learning module for the
Kaggriculture AI Agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class LearningModule:
    """
    Stores how successful each strategy has been.
    """

    def __init__(self):
        self.results = {}

    # ---------------------------------------------------------

    def record(self, strategy: str, reward: float):
        """
        Record the outcome of a strategy.
        """

        self.results.setdefault(strategy, []).append(reward)

    # ---------------------------------------------------------

    def average_reward(self, strategy: str) -> float:
        """
        Average reward of a strategy.
        """

        history = self.results.get(strategy, [])

        if not history:
            return 0.0

        return sum(history) / len(history)

    # ---------------------------------------------------------

    def best_strategy(self):
        """
        Return the historically best strategy.
        """

        if not self.results:
            return None

        return max(
            self.results,
            key=self.average_reward,
        )

    # ---------------------------------------------------------

    def total_games(self):
        """
        Return the total number of recorded games.
        """

        return sum(len(v) for v in self.results.values())

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear all learning history.
        """

        self.results.clear()