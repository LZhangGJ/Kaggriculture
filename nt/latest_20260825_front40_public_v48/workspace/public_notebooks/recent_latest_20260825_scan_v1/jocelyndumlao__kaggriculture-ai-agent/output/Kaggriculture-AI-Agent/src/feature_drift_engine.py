"""
feature_drift_engine.py

Feature Drift Engine for the Kaggriculture AI Agent.

Monitors changes in extracted features across
game states to detect significant shifts.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureDriftEngine:
    """
    Detect feature drift.
    """

    def difference(
        self,
        previous: dict[str, float],
        current: dict[str, float],
    ) -> dict[str, float]:
        """
        Return absolute feature differences.
        """

        keys = set(previous) | set(current)

        return {
            key: abs(
                current.get(key, 0.0)
                - previous.get(key, 0.0)
            )
            for key in keys
        }

    # ---------------------------------------------------------

    def drift_score(
        self,
        previous: dict[str, float],
        current: dict[str, float],
    ) -> float:
        """
        Return total drift score.
        """

        return sum(
            self.difference(
                previous,
                current,
            ).values()
        )

    # ---------------------------------------------------------

    def has_drift(
        self,
        previous: dict[str, float],
        current: dict[str, float],
        *,
        threshold: float = 1.0,
    ) -> bool:
        """
        Return True if drift exceeds threshold.
        """

        return (
            self.drift_score(
                previous,
                current,
            )
            >= threshold
        )