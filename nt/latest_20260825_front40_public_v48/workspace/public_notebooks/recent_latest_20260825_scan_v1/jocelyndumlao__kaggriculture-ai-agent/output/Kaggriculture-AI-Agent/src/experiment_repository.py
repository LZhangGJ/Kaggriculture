"""
experiment_repository.py

Experiment Repository for the Kaggriculture AI Agent.

Stores experiment records.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExperimentRepository:
    """
    Stores experiment records.
    """

    def __init__(self):
        self._storage = {}

    # ---------------------------------------------------------

    def save(self, name: str, experiment):
        """
        Save an experiment.
        """
        self._storage[name] = experiment

    # ---------------------------------------------------------

    def get(self, name: str):
        """
        Retrieve an experiment.
        """
        return self._storage.get(name)

    # ---------------------------------------------------------

    def update(self, name: str, experiment):
        """
        Update an existing experiment.
        """
        self._storage[name] = experiment

    # ---------------------------------------------------------

    def delete(self, name: str):
        """
        Delete an experiment.
        """
        self._storage.pop(name, None)

    # ---------------------------------------------------------

    def list_experiments(self):
        """
        Return experiment names.
        """
        return sorted(self._storage.keys())

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all experiments.
        """
        self._storage.clear()