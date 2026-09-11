"""
legal_action_generator.py

Legal Action Generator for the Kaggriculture AI Agent.

Generates legal candidate actions from the current
observation.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class LegalActionGenerator:
    """
    Generates legal actions.

    Version 1 only handles mature crops.
    """

    def generate(self, observation: dict):
        """
        Generate legal actions.
        """

        actions = []

        farm = observation.get("farm", {})
        tiles = farm.get("tiles", [])

        for row_index, row in enumerate(tiles):

            for column_index, tile in enumerate(row):

                if not isinstance(tile, dict):
                    continue

                if (
                    tile.get("kind") == "PLANT"
                    and tile.get("mature", False)
                ):
                    actions.append(
                        {
                            "action": "HARVEST",
                            "position": (
                                row_index,
                                column_index,
                            ),
                        }
                    )

        return actions

    # ---------------------------------------------------------

    def count(
        self,
        observation: dict,
    ) -> int:
        """
        Return number of legal actions.
        """

        return len(
            self.generate(observation)
        )

    # ---------------------------------------------------------

    def has_actions(
        self,
        observation: dict,
    ) -> bool:
        """
        Return True if legal actions exist.
        """

        return self.count(observation) > 0