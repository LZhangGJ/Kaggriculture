"""
authentication_manager.py

Authentication Manager for the Kaggriculture AI Agent.

Manages API keys for inference services.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AuthenticationManager:
    """
    Manages API authentication.
    """

    def __init__(self):
        self._clients = {}

    # ---------------------------------------------------------

    def register(
        self,
        client: str,
        api_key: str,
    ):
        """
        Register an API key.
        """

        self._clients[client] = api_key

    # ---------------------------------------------------------

    def authenticate(
        self,
        client: str,
        api_key: str,
    ) -> bool:
        """
        Validate an API key.
        """

        return self._clients.get(client) == api_key

    # ---------------------------------------------------------

    def revoke(
        self,
        client: str,
    ):
        """
        Remove a client.
        """

        self._clients.pop(client, None)

    # ---------------------------------------------------------

    def authorized(
        self,
        client: str,
    ) -> bool:
        """
        Return True if the client is registered.
        """

        return client in self._clients

    # ---------------------------------------------------------

    def list_clients(self):
        """
        Return registered clients.
        """

        return sorted(self._clients.keys())

    # ---------------------------------------------------------

    def count(self) -> int:
        """
        Return number of registered clients.
        """

        return len(self._clients)