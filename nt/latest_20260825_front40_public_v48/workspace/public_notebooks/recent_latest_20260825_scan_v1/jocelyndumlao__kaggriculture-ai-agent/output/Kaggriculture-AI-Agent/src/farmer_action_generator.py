"""
farmer_action_generator.py

Farmer Action Generator for the Kaggriculture AI Agent.

Generates valid farmer actions based on the
current game state.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FarmerActionGenerator:
    """
    Generates farmer actions.
    """

    def generate(
        self,
        state: dict,
    ) -> list:
        """
        Generate legal farmer actions.
        """

        actions = []

        farmer = state.get("farmer", [0, 0])
        x, y = farmer

        tiles = state.get("tiles", [])

        if (
            y < len(tiles)
            and x < len(tiles[y])
        ):
            tile = tiles[y][x]

            if isinstance(tile, dict):

                if (
                    tile.get("kind") == "PLANT"
                    and tile.get("yield_units", 0) > 0
                ):
                    actions.append(["HARVEST"])

                elif (
                    tile.get("kind") == "PLANT"
                    and not tile.get(
                        "watered_today",
                        False,
                    )
                ):
                    actions.append(["WATER"])

            elif tile is None:

                seeds = state.get(
                    "seeds",
                    {},
                )

                if seeds.get("MELON", 0) > 0:
                    actions.append(
                        [
                            "PLANT",
                            "MELON",
                        ]
                    )

        # Always allow PASS
        actions.append(["PASS"])

        return actions