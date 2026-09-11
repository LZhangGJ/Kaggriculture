"""
experiment_tracker.py

Experiment Tracker for the Kaggriculture AI Agent.

Tracks AI experiments, parameters, and metrics.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExperimentTracker:
    """
    Tracks machine learning experiments.
    """

    def __init__(self):
        self._experiments = {}

    # ---------------------------------------------------------

    def start(self, name: str):
        """
        Start a new experiment.
        """
        self._experiments[name] = {
            "parameters": {},
            "metrics": {},
            "status": "running",
        }

    # ---------------------------------------------------------

    def log_parameter(self, experiment: str, key: str, value):
        """
        Log an experiment parameter.
        """
        self._experiments[experiment]["parameters"][key] = value

    # ---------------------------------------------------------

    def log_metric(self, experiment: str, key: str, value):
        """
        Log an experiment metric.
        """
        self._experiments[experiment]["metrics"][key] = value

    # ---------------------------------------------------------

    def finish(self, experiment: str):
        """
        Mark an experiment as completed.
        """
        self._experiments[experiment]["status"] = "completed"

    # ---------------------------------------------------------

    def get(self, experiment: str):
        """
        Return experiment data.
        """
        return self._experiments.get(experiment)

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all experiments.
        """
        self._experiments.clear()