"""
inference_request_router.py

Inference Request Router for the Kaggriculture AI Agent.

Routes inference requests to registered handlers.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class InferenceRequestRouter:
    """
    Routes inference requests.
    """

    def __init__(self):
        self._routes = {}

    # ---------------------------------------------------------

    def register(
        self,
        route: str,
        handler,
    ):
        """
        Register a route.
        """

        self._routes[route] = handler

    # ---------------------------------------------------------

    def dispatch(
        self,
        route: str,
        payload,
    ):
        """
        Dispatch a request.
        """

        handler = self._routes.get(route)

        if handler is None:
            return None

        return handler(payload)

    # ---------------------------------------------------------

    def exists(
        self,
        route: str,
    ) -> bool:
        """
        Return True if the route exists.
        """

        return route in self._routes

    # ---------------------------------------------------------

    def remove(
        self,
        route: str,
    ):
        """
        Remove a route.
        """

        self._routes.pop(route, None)

    # ---------------------------------------------------------

    def list_routes(self):
        """
        Return registered routes.
        """

        return sorted(self._routes.keys())

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all routes.
        """

        self._routes.clear()