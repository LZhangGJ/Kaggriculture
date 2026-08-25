"""
evaluation_engine.py

Evaluation Engine for the Kaggriculture AI Agent.

Evaluates trained models across benchmark episodes.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class EvaluationEngine:
    """
    Collects and summarizes evaluation scores.
    """

    def __init__(self):
        self._scores = []

    # ---------------------------------------------------------

    def add_score(self, score: float):
        """
        Add an evaluation score.
        """
        self._scores.append(score)

    # ---------------------------------------------------------

    def scores(self):
        """
        Return all evaluation scores.
        """
        return list(self._scores)

    # ---------------------------------------------------------

    def average_score(self) -> float:
        """
        Compute the average score.
        """
        if not self._scores:
            return 0.0

        return sum(self._scores) / len(self._scores)

    # ---------------------------------------------------------

    def best_score(self) -> float:
        """
        Return the highest score.
        """
        if not self._scores:
            return 0.0

        return max(self._scores)

    # ---------------------------------------------------------

    def summary(self):
        """
        Return evaluation statistics.
        """
        return {
            "count": len(self._scores),
            "average": self.average_score(),
            "best": self.best_score(),
        }

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear all evaluation results.
        """
        self._scores.clear()