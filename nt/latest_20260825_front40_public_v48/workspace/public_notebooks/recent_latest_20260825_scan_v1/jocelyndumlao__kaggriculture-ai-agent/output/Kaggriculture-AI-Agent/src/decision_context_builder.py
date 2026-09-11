"""
decision_context_builder.py

Decision Context Builder for the Kaggriculture AI Agent.

Builds a unified decision context from
the normalized game state.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionContextBuilder:
    """
    Build decision context.
    """

    def build(
        self,
        *,
        game_state: dict,
        search_algorithm: str,
    ) -> dict:
        """
        Build the decision context.
        """

        return {
            "day": game_state.get("day", 0),
            "hour": game_state.get("hour", 0),
            "money": game_state.get("money", 0),
            "tiles": game_state.get("tiles", []),
            "market": game_state.get("market", {}),
            "workers": game_state.get("workers", []),
            "inventory": game_state.get("inventory", {}),
            "algorithm": search_algorithm,
        }

    # ---------------------------------------------------------

    def tile_count(
        self,
        context: dict,
    ) -> int:
        """
        Return number of tiles.
        """

        return len(
            context.get("tiles", [])
        )

    # ---------------------------------------------------------

    def worker_count(
        self,
        context: dict,
    ) -> int:
        """
        Return number of workers.
        """

        return len(
            context.get("workers", [])
        )

    # ---------------------------------------------------------

    def has_inventory(
        self,
        context: dict,
    ) -> bool:
        """
        Return True if inventory exists.
        """

        return bool(
            context.get("inventory")
        )

    # ---------------------------------------------------------

    def algorithm(
        self,
        context: dict,
    ) -> str:
        """
        Return selected search algorithm.
        """

        return context["algorithm"]