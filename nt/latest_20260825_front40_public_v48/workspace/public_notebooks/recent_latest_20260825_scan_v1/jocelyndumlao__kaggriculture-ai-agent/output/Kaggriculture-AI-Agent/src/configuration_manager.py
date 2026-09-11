"""
configuration_manager.py

Configuration Manager for the Kaggriculture AI Agent.

Stores and manages runtime configuration.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ConfigurationManager:
    """
    Stores runtime configuration.
    """

    def __init__(self):
        self._config = {}

    # ---------------------------------------------------------

    def set(
        self,
        key: str,
        value,
    ):
        """
        Set a configuration value.
        """
        self._config[key] = value

    # ---------------------------------------------------------

    def get(
        self,
        key: str,
        default=None,
    ):
        """
        Retrieve a configuration value.
        """
        return self._config.get(
            key,
            default,
        )

    # ---------------------------------------------------------

    def update(
        self,
        values: dict,
    ):
        """
        Update multiple configuration values.
        """
        self._config.update(values)

    # ---------------------------------------------------------

    def remove(
        self,
        key: str,
    ):
        """
        Remove a configuration value.
        """
        self._config.pop(key, None)

    # ---------------------------------------------------------

    def exists(
        self,
        key: str,
    ) -> bool:
        """
        Return True if a configuration key exists.
        """
        return key in self._config

    # ---------------------------------------------------------

    def all(self):
        """
        Return a copy of all configuration values.
        """
        return dict(self._config)

    # ---------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of configuration entries.
        """
        return len(self._config)

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear all configuration.
        """
        self._config.clear()