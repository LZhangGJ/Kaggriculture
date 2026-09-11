"""
feature_importance_engine.py

Feature Importance Engine for the Kaggriculture AI Agent.

Maintains importance scores for extracted
features to support adaptive learning and
decision analysis.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureImportanceEngine:
    """
    Track feature importance.
    """

    def __init__(self):
        self._importance: dict[str, float] = {}

    # ---------------------------------------------------------

    def update(
        self,
        feature: str,
        score: float,
    ) -> None:
        """
        Update a feature's importance score.
        """

        self._importance[feature] = (
            self._importance.get(feature, 0.0)
            + score
        )

    # ---------------------------------------------------------

    def importance(
        self,
        feature: str,
    ) -> float:
        """
        Return a feature's importance.
        """

        return self._importance.get(feature, 0.0)

    # ---------------------------------------------------------

    def ranking(
        self,
    ) -> list[tuple[str, float]]:
        """
        Return features ranked by importance.
        """

        return sorted(
            self._importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    # ---------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Clear all importance scores.
        """

        self._importance.clear()