"""
market_action_generator.py

Market Action Generator for the Kaggriculture AI Agent.

Generates valid market actions based on the
current game state.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketActionGenerator:
    """
    Generates market actions.
    """

    def generate(
        self,
        state: dict,
    ) -> list:
        """
        Generate market actions.
        """

        actions = []

        shed = state.get("shed", {})
        seeds = state.get("seeds", {})
        prices = (
            state.get("market", {})
            .get("prices", {})
        )

        # Example strategy:
        # Sell melons when price is high.
        if (
            shed.get("MELON", 0) > 0
            and prices.get("MELON", 0) >= 200
        ):
            actions.append(
                [
                    "SELL",
                    "MELON",
                    shed["MELON"],
                ]
            )

        # Keep at least one melon seed.
        if (
            seeds.get("MELON", 0) == 0
            and state.get("money", 0) >= 120
        ):
            actions.append(
                [
                    "BUY_SEED",
                    "MELON",
                    1,
                ]
            )

        return actions