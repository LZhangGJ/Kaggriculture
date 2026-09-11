"""
metrics_collector.py

Metrics Collector for the Kaggriculture AI Agent.

Collects metric observations and computes summary statistics.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MetricsCollector:
    """
    Collects metric observations over time.
    """

    def __init__(self):
        self._metrics = {}

    # ---------------------------------------------------------

    def record(self, name: str, value):
        """
        Record a metric observation.
        """
        self._metrics.setdefault(name, []).append(value)

    # ---------------------------------------------------------

    def history(self, name: str):
        """
        Return the history for a metric.
        """
        return list(self._metrics.get(name, []))

    # ---------------------------------------------------------

    def count(self, name: str) -> int:
        """
        Return the number of observations.
        """
        return len(self._metrics.get(name, []))

    # ---------------------------------------------------------

    def average(self, name: str):
        """
        Return the average value.

        Returns None if no observations exist.
        """
        values = self._metrics.get(name)

        if not values:
            return None

        return sum(values) / len(values)

    # ---------------------------------------------------------

    def reset(self):
        """
        Remove all collected metrics.
        """
        self._metrics.clear()

        # ---------------------------------------------------------

    def latest(self, name: str):
        """
        Return the latest observation.

        Returns None if no observations exist.
        """
        values = self._metrics.get(name)

        if not values:
            return None

        return values[-1]

    # ---------------------------------------------------------

    def minimum(self, name: str):
        """
        Return the minimum recorded value.
        """

        values = self._metrics.get(name)

        if not values:
            return None

        return min(values)

    # ---------------------------------------------------------

    def maximum(self, name: str):
        """
        Return the maximum recorded value.
        """

        values = self._metrics.get(name)

        if not values:
            return None

        return max(values)

    # ---------------------------------------------------------

    def metric_names(self):
        """
        Return all metric names.
        """

        return sorted(self._metrics.keys())