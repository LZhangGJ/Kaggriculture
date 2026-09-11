"""
feature_version_manager.py

Feature Version Manager for the Kaggriculture AI Agent.

Tracks feature schema versions to ensure
compatibility across learning, prediction,
and replay components.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureVersionManager:
    """
    Manage feature schema versions.
    """

    def __init__(self):
        self._version = 1
        self._history = [1]

    # ---------------------------------------------------------

    def current(self) -> int:
        """
        Return the current version.
        """

        return self._version

    # ---------------------------------------------------------

    def increment(self) -> int:
        """
        Create a new version.
        """

        self._version += 1
        self._history.append(
            self._version
        )
        return self._version

    # ---------------------------------------------------------

    def history(self) -> list[int]:
        """
        Return version history.
        """

        return list(self._history)

    # ---------------------------------------------------------

    def reset(self) -> None:
        """
        Reset version tracking.
        """

        self._version = 1
        self._history = [1]