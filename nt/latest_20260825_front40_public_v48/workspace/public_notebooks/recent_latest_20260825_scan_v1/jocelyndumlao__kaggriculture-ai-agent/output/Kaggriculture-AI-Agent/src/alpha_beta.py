"""
alpha_beta.py

Alpha-Beta Pruning Engine for the Kaggriculture AI Agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AlphaBeta:
    """
    Simplified Alpha-Beta evaluator.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------

    def maximize(
        self,
        scores: list[float],
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:
        """
        Return the best maximizing score.
        """

        value = float("-inf")

        for score in scores:

            value = max(value, score)

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        if value == float("-inf"):
            return 0.0

        return value

    # ---------------------------------------------------------

    def minimize(
        self,
        scores: list[float],
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:
        """
        Return the best minimizing score.
        """

        value = float("inf")

        for score in scores:

            value = min(value, score)

            beta = min(beta, value)

            if beta <= alpha:
                break

        if value == float("inf"):
            return 0.0

        return value

    # ---------------------------------------------------------

    def choose(
        self,
        scores: list[float],
        maximizing: bool = True,
    ) -> float:
        """
        Choose the best score.
        """

        if maximizing:
            return self.maximize(scores)

        return self.minimize(scores)