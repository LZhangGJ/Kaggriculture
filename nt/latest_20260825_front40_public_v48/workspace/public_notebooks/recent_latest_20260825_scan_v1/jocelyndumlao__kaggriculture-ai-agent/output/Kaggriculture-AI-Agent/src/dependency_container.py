"""
dependency_container.py

Dependency Injection Container for the Kaggriculture AI Agent.

Provides lightweight dependency registration and resolution.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DependencyContainer:
    """
    Lightweight dependency injection container.
    """

    def __init__(self):
        self._services = {}
        self._instances = {}

    # ---------------------------------------------------------

    def register(self, name: str, factory):
        """
        Register a factory function or class.
        """
        self._services[name] = factory

    # ---------------------------------------------------------

    def register_instance(self, name: str, instance):
        """
        Register a singleton instance.
        """
        self._instances[name] = instance

    # ---------------------------------------------------------

    def resolve(self, name: str):
        """
        Resolve a dependency.
        """
        if name in self._instances:
            return self._instances[name]

        if name in self._services:
            return self._services[name]()

        raise KeyError(f"Dependency '{name}' is not registered.")

    # ---------------------------------------------------------

    def exists(self, name: str) -> bool:
        """
        Check whether a dependency exists.
        """
        return (
            name in self._services
            or name in self._instances
        )

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all registrations.
        """
        self._services.clear()
        self._instances.clear()