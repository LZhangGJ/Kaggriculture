"""
adaptive_strategy.py

Adaptive strategy selector for the Kaggriculture AI Agent.

Chooses the best strategy using historical learning.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AdaptiveStrategy:
    """
    Select the best strategy using historical rewards.
    """

    def __init__(self, learning):
        self.learning = learning

    # ---------------------------------------------------------

    def choose(self):
        """
        Return the historically best strategy.
        """

        strategy = self.learning.best_strategy()

        if strategy is None:
            return "BALANCED"

        return strategy

    # ---------------------------------------------------------

    def recommend(self):
        """
        Alias for choose().
        """

        return self.choose()

    # ---------------------------------------------------------

    def confidence(self):
        """
        Estimate confidence from recorded games.
        """

        games = self.learning.total_games()

        if games == 0:
            return 0.0

        return min(games / 20.0, 1.0)