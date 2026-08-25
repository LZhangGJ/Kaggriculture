"""
latency_tracker.py

Latency Tracker for the Kaggriculture AI Agent.

Tracks inference latency statistics.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class LatencyTracker:
    """
    Tracks request latencies.
    """

    def __init__(self):
        self._latencies = []

    # ---------------------------------------------------------

    def record(
        self,
        latency: float,
    ):
        """
        Record a latency value.
        """

        self._latencies.append(latency)

    # ---------------------------------------------------------

    def latest(self):
        """
        Return the latest latency.
        """

        if not self._latencies:
            return None

        return self._latencies[-1]

    # ---------------------------------------------------------

    def minimum(self):
        """
        Return the minimum latency.
        """

        if not self._latencies:
            return None

        return min(self._latencies)

    # ---------------------------------------------------------

    def maximum(self):
        """
        Return the maximum latency.
        """

        if not self._latencies:
            return None

        return max(self._latencies)

    # ---------------------------------------------------------

    def average(self):
        """
        Return the average latency.
        """

        if not self._latencies:
            return None

        return sum(self._latencies) / len(self._latencies)

    # ---------------------------------------------------------

    def count(self):
        """
        Return the number of latency samples.
        """

        return len(self._latencies)