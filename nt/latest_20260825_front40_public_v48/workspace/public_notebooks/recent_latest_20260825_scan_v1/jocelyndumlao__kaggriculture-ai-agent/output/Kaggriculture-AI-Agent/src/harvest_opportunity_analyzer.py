"""
harvest_opportunity_analyzer.py

Harvest Opportunity Analyzer for the Kaggriculture AI Agent.

Identifies harvestable crops and ranks
harvest opportunities.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class HarvestOpportunityAnalyzer:
    """
    Analyze harvest opportunities.
    """

    # ---------------------------------------------------------

    def harvestable(
        self,
        tiles: list[dict],
    ) -> list[dict]:
        """
        Return harvestable crops.
        """

        return [
            tile
            for tile in tiles
            if tile.get("kind") == "PLANT"
            and tile.get("yield_units", 0) > 0
        ]

    # ---------------------------------------------------------

    def count(
        self,
        tiles: list[dict],
    ) -> int:
        """
        Return number of harvestable crops.
        """

        return len(self.harvestable(tiles))

    # ---------------------------------------------------------

    def best(
        self,
        tiles: list[dict],
    ) -> dict | None:
        """
        Return crop with highest yield.
        """

        crops = self.harvestable(tiles)

        if not crops:
            return None

        return max(
            crops,
            key=lambda tile: tile.get(
                "yield_units",
                0,
            ),
        )