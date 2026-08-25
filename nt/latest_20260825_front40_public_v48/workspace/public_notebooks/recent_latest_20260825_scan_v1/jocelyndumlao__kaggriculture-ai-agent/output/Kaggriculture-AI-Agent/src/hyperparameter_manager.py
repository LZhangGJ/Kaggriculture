"""
hyperparameter_manager.py

Hyperparameter Manager for the Kaggriculture AI Agent.

Centralizes reinforcement learning hyperparameters.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class HyperparameterManager:
    """
    Stores RL hyperparameters.
    """

    DEFAULTS = {
        "learning_rate": 0.10,
        "discount_factor": 0.95,
        "epsilon": 0.10,
        "batch_size": 32,
        "replay_buffer_size": 1000,
    }

    def __init__(self):
        self._params = dict(self.DEFAULTS)

    # ---------------------------------------------------------

    def get(self, name):
        """
        Get a hyperparameter.
        """
        return self._params[name]

    # ---------------------------------------------------------

    def set(self, name, value):
        """
        Update a hyperparameter.
        """
        self._params[name] = value

    # ---------------------------------------------------------

    def exists(self, name):
        """
        Return True if parameter exists.
        """
        return name in self._params

    # ---------------------------------------------------------

    def all(self):
        """
        Return a copy of all parameters.
        """
        return dict(self._params)

    # ---------------------------------------------------------

    def reset(self):
        """
        Restore default parameters.
        """
        self._params = dict(self.DEFAULTS)