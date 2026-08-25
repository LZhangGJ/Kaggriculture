"""
experiment_analyzer.py

Experiment Analyzer for the Kaggriculture AI Agent.

Analyzes collections of experiment results.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExperimentAnalyzer:
    """
    Analyze experiment metrics.
    """

    # ---------------------------------------------------------

    def best(self, experiments, metric: str):
        """
        Return highest metric value.
        """
        if not experiments:
            return None

        return max(
            experiment.get(metric, float("-inf"))
            for experiment in experiments
        )

    # ---------------------------------------------------------

    def average(self, experiments, metric: str):
        """
        Return average metric.
        """
        if not experiments:
            return None

        values = [
            experiment.get(metric, 0)
            for experiment in experiments
        ]

        return sum(values) / len(values)

    # ---------------------------------------------------------

    def worst(self, experiments, metric: str):
        """
        Return lowest metric.
        """
        if not experiments:
            return None

        return min(
            experiment.get(metric, float("inf"))
            for experiment in experiments
        )

    # ---------------------------------------------------------

    def count(self, experiments):
        """
        Return experiment count.
        """
        return len(experiments)

    # ---------------------------------------------------------

    def summary(self, experiments, metric: str):
        """
        Return summary statistics.
        """
        return {
            "count": self.count(experiments),
            "best": self.best(experiments, metric),
            "average": self.average(experiments, metric),
            "worst": self.worst(experiments, metric),
        }