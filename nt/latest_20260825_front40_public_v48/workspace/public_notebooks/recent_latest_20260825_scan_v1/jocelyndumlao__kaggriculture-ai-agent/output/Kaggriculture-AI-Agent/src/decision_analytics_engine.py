"""
decision_analytics_engine.py

Decision Analytics Engine for the Kaggriculture AI Agent.

Collects statistics about AI decisions.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionAnalyticsEngine:
    """
    Tracks decision statistics.
    """

    def __init__(self):

        self._history = []

    # ---------------------------------------------------------

    def record(
        self,
        action: str,
        score: float,
    ) -> None:
        """
        Record one decision.
        """

        self._history.append(
            {
                "action": action,
                "score": float(score),
            }
        )

    # ---------------------------------------------------------

    def count(self) -> int:
        """
        Number of decisions.
        """

        return len(self._history)

    # ---------------------------------------------------------

    def average_score(self) -> float:
        """
        Return average decision score.
        """

        if not self._history:
            return 0.0

        return sum(
            item["score"]
            for item in self._history
        ) / len(self._history)

    # ---------------------------------------------------------

    def best(self):
        """
        Highest-scoring decision.
        """

        if not self._history:
            return None

        return max(
            self._history,
            key=lambda item: item["score"],
        )

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all history.
        """

        self._history.clear()