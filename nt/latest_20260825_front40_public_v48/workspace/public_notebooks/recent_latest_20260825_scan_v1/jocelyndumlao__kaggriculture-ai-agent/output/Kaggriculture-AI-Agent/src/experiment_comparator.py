"""
experiment_comparator.py

Experiment Comparator for the Kaggriculture AI Agent.

Compares experiment metrics and rankings.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExperimentComparator:
    """
    Compares experiment results.
    """

    def best(self, experiments, metric: str):
        """
        Return the name of the experiment with the
        highest metric value.
        """
        if not experiments:
            return None

        return max(
            experiments,
            key=lambda name: experiments[name].get(metric, float("-inf")),
        )

    # ---------------------------------------------------------

    def compare(self, first, second, metric: str):
        """
        Return metric difference:
        first - second.
        """
        return (
            first.get(metric, 0)
            - second.get(metric, 0)
        )

    # ---------------------------------------------------------

    def rank(self, experiments, metric: str):
        """
        Rank experiments from best to worst.
        """
        return sorted(
            experiments.keys(),
            key=lambda name: experiments[name].get(metric, float("-inf")),
            reverse=True,
        )

    # ---------------------------------------------------------

    def better(self, first, second, metric: str):
        """
        Return True if first beats second.
        """
        return (
            first.get(metric, float("-inf"))
            >
            second.get(metric, float("-inf"))
        )

    # ---------------------------------------------------------

    def equal(self, first, second, metric: str):
        """
        Return True if metric values are equal.
        """
        return (
            first.get(metric)
            ==
            second.get(metric)
        )