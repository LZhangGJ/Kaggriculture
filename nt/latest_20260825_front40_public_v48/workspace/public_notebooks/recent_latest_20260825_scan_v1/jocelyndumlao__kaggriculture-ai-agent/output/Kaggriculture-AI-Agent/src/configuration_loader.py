"""
configuration_loader.py

Configuration Loader for the Kaggriculture AI Agent.

Loads and saves application configuration.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import json
from pathlib import Path


class ConfigurationLoader:
    """
    Loads and manages configuration settings.
    """

    def __init__(self):
        self._config = {}

    # ---------------------------------------------------------

    def load(self, filename):
        """
        Load configuration from a JSON file.
        """
        with open(filename, "r", encoding="utf-8") as file:
            self._config = json.load(file)

    # ---------------------------------------------------------

    def save(self, filename):
        """
        Save configuration to a JSON file.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._config, file, indent=4)

    # ---------------------------------------------------------

    def get(self, key, default=None):
        """
        Retrieve a configuration value.
        """
        return self._config.get(key, default)

    # ---------------------------------------------------------

    def set(self, key, value):
        """
        Update a configuration value.
        """
        self._config[key] = value

    # ---------------------------------------------------------

    def all(self):
        """
        Return a copy of the configuration.
        """
        return dict(self._config)

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear all configuration values.
        """
        self._config.clear()