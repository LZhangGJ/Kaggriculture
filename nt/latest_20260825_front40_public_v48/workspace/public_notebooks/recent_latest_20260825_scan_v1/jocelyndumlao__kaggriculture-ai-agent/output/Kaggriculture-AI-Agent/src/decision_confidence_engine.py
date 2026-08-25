"""
decision_confidence_engine.py

Decision Confidence Engine for the Kaggriculture AI Agent.

Computes a confidence score from ranked action
priorities.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionConfidenceEngine:
    """
    Estimate confidence in the chosen action.
    """

    # ---------------------------------------------------------

    def confidence(
        self,
        priorities: list[float],
    ) -> float:
        """
        Return a confidence score between
        0.0 and 1.0.
        """

        if not priorities:
            return 0.0

        if len(priorities) == 1:
            return 1.0

        ordered = sorted(
            priorities,
            reverse=True,
        )

        best = ordered[0]
        second = ordered[1]

        if best <= 0:
            return 0.0

        gap = best - second

        return min(
            1.0,
            max(
                0.0,
                gap / best,
            ),
        )

    # ---------------------------------------------------------

    def confident(
        self,
        priorities: list[float],
        threshold: float = 0.25,
    ) -> bool:
        """
        Return True if confidence exceeds
        the threshold.
        """

        return (
            self.confidence(
                priorities,
            )
            >= threshold
        )