"""
metrics_aggregator.py

Metrics Aggregator for the Kaggriculture AI Agent.

Aggregates observations into useful statistics.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MetricsAggregator:
    """
    Aggregates metric observations.
    """

    def __init__(self):
        self._metrics = {}

    # ---------------------------------------------------------

    def add(self, name: str, value):
        """
        Add an observation.
        """
        self._metrics.setdefault(name, []).append(value)

    # ---------------------------------------------------------

    def mean(self, name: str):
        """
        Return the mean value.
        """
        values = self._metrics.get(name)

        if not values:
            return None

        return sum(values) / len(values)

    # ---------------------------------------------------------

    def minimum(self, name: str):
        """
        Return the minimum value.
        """
        values = self._metrics.get(name)

        if not values:
            return None

        return min(values)

    # ---------------------------------------------------------

    def maximum(self, name: str):
        """
        Return the maximum value.
        """
        values = self._metrics.get(name)

        if not values:
            return None

        return max(values)

    # ---------------------------------------------------------

    def total(self, name: str):
        """
        Return the sum of observations.
        """
        values = self._metrics.get(name)

        if not values:
            return 0

        return sum(values)

    # ---------------------------------------------------------

    def reset(self):
        """
        Remove every stored metric.
        """
        self._metrics.clear()