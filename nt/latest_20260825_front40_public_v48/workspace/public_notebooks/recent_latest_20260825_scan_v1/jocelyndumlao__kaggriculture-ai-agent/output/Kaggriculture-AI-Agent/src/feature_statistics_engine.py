"""
feature_statistics_engine.py

Feature Statistics Engine for the Kaggriculture AI Agent.

Maintains running statistics for extracted
features to support monitoring, analytics,
and future machine learning improvements.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureStatisticsEngine:
    """
    Maintain feature statistics.
    """

    def __init__(self):
        self._values: dict[str, list[float]] = {}

    # ---------------------------------------------------------

    def update(
        self,
        features: dict[str, float],
    ) -> None:
        """
        Add a feature observation.
        """

        for name, value in features.items():
            self._values.setdefault(
                name,
                [],
            ).append(float(value))

    # ---------------------------------------------------------

    def mean(
        self,
        feature: str,
    ) -> float:
        """
        Return the mean value.
        """

        values = self._values.get(feature, [])

        if not values:
            return 0.0

        return sum(values) / len(values)

    # ---------------------------------------------------------

    def count(
        self,
        feature: str,
    ) -> int:
        """
        Return observation count.
        """

        return len(
            self._values.get(
                feature,
                [],
            )
        )

    # ---------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Clear all statistics.
        """

        self._values.clear()