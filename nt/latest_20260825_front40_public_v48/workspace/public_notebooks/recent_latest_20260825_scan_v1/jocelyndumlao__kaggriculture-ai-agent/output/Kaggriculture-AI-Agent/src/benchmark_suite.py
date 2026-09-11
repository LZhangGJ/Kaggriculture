"""
benchmark_suite.py

Benchmark Suite for the Kaggriculture AI Agent.

Runs and summarizes benchmark scenarios.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class BenchmarkSuite:
    """
    Stores benchmark results.
    """

    def __init__(self):
        self._results = {}

    # ---------------------------------------------------------

    def add_result(self, scenario: str, score: float):
        """
        Record the score for a benchmark scenario.
        """
        self._results[scenario] = score

    # ---------------------------------------------------------

    def score(self, scenario: str):
        """
        Return the score of a benchmark.
        """
        return self._results.get(scenario)

    # ---------------------------------------------------------

    def average_score(self):
        """
        Return the average benchmark score.
        """
        if not self._results:
            return 0.0

        return sum(self._results.values()) / len(self._results)

    # ---------------------------------------------------------

    def best_scenario(self):
        """
        Return the highest-scoring scenario.
        """
        if not self._results:
            return None

        return max(
            self._results.items(),
            key=lambda item: item[1],
        )[0]

    # ---------------------------------------------------------

    def summary(self):
        """
        Return benchmark statistics.
        """
        return {
            "count": len(self._results),
            "average_score": self.average_score(),
            "best_scenario": self.best_scenario(),
        }

    # ---------------------------------------------------------

    def reset(self):
        """
        Remove all benchmark results.
        """
        self._results.clear()