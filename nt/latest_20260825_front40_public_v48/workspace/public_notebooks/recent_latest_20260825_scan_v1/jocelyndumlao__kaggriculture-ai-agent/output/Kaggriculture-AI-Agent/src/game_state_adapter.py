"""
game_state_adapter.py

Game State Adapter for the Kaggriculture AI Agent.

Converts parsed observations into a normalized game state.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.observation_parser import ObservationParser


class GameStateAdapter:
    """
    Converts observations into a normalized game state.
    """

    def __init__(self):
        self.parser = ObservationParser()

    # ---------------------------------------------------------

    def adapt(
        self,
        observation: dict,
    ) -> dict:
        """
        Return a normalized game state.
        """

        state = self.parser.parse(observation)

        return {
            # Legacy fields (keeps existing tests passing)
            "money": state.get("money", 0),
            "tiles": state.get("tiles", []),
            "inventory": state.get("inventory", {}),
            "workers": state.get("workers", []),
            "market": state.get("market", {}),
            "opponent": state.get("opponent", {}),

            # New Kaggriculture fields
            "player": state.get("player", 0),
            "day": state.get("day", 0),
            "hour": state.get("hour", 0),
            "farmer": state.get("farmer", [0, 0]),
            "hands": state.get("hands", []),
            "unlocked_quadrants": state.get(
                "unlocked_quadrants",
                [],
            ),
            "hires_today": state.get(
                "hires_today",
                0,
            ),
            "town": state.get("town", {}),
            "shed": state.get("shed", {}),
            "seeds": state.get("seeds", {}),
            "inventories": state.get(
                "inventories",
                [],
            ),
        }

    # ---------------------------------------------------------

    def tile_count(
        self,
        observation: dict,
    ) -> int:
        """
        Return the number of tiles.
        """

        state = self.adapt(observation)

        tiles = state["tiles"]

        if not tiles:
            return 0

        return sum(len(row) for row in tiles)

    # ---------------------------------------------------------

    def inventory_size(
        self,
        observation: dict,
    ) -> int:
        """
        Return number of inventory entries.
        """

        state = self.adapt(observation)

        return len(state["inventory"])

    # ---------------------------------------------------------

    def worker_count(
        self,
        observation: dict,
    ) -> int:
        """
        Return number of workers.
        """

        state = self.adapt(observation)

        return len(state["workers"])