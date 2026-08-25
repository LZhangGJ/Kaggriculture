"""
experiment_manager.py

Experiment Manager for the Kaggriculture AI Agent.

Tracks machine learning experiments and their results.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from datetime import datetime, UTC


class ExperimentManager:
    """
    Manages training experiments.
    """

    def __init__(self, name: str):
        self.name = name
        self.created = datetime.now(UTC).isoformat()
        self.hyperparameters = {}
        self.metrics = {}

    # ---------------------------------------------------------

    def set_hyperparameter(self, key: str, value):
        """
        Store one hyperparameter.
        """
        self.hyperparameters[key] = value

    # ---------------------------------------------------------

    def get_hyperparameter(self, key: str):
        """
        Retrieve a hyperparameter.
        """
        return self.hyperparameters.get(key)

    # ---------------------------------------------------------

    def log_metric(self, key: str, value):
        """
        Store one evaluation metric.
        """
        self.metrics[key] = value

    # ---------------------------------------------------------

    def get_metric(self, key: str):
        """
        Retrieve a metric.
        """
        return self.metrics.get(key)

    # ---------------------------------------------------------

    def summary(self):
        """
        Return experiment information.
        """
        return {
            "name": self.name,
            "created": self.created,
            "hyperparameters": dict(self.hyperparameters),
            "metrics": dict(self.metrics),
        }

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear experiment data.
        """
        self.hyperparameters.clear()
        self.metrics.clear()