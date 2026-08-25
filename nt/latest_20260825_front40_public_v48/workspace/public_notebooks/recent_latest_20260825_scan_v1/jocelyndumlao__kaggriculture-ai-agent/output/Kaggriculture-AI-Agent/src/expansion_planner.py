"""
expansion_planner.py

Expansion Planner for the Kaggriculture AI Agent.

Determines when purchasing additional land is a
good long-term investment.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExpansionPlanner:
    """
    Plans farm expansion decisions.
    """

    def __init__(
        self,
        minimum_cash: float = 5000.0,
    ):
        self.minimum_cash = minimum_cash

    # ---------------------------------------------------------

    def can_expand(
        self,
        *,
        money: float,
        available_land: int,
    ) -> bool:
        """
        Return True if expansion is possible.
        """

        return (
            money >= self.minimum_cash
            and available_land > 0
        )

    # ---------------------------------------------------------

    def expansion_priority(
        self,
        *,
        money: float,
        available_land: int,
    ) -> float:
        """
        Compute expansion priority.
        """

        if not self.can_expand(
            money=money,
            available_land=available_land,
        ):
            return 0.0

        score = 50.0

        score += min(
            money / 1000.0,
            50.0,
        )

        return score

    # ---------------------------------------------------------

    def should_expand(
        self,
        *,
        money: float,
        available_land: int,
    ) -> bool:

        return (
            self.expansion_priority(
                money=money,
                available_land=available_land,
            )
            >= 60.0
        )

    # ---------------------------------------------------------

    def recommended_land(
        self,
        *,
        money: float,
        available_land: int,
    ) -> int:
        """
        Recommend how many plots to purchase.
        """

        if not self.should_expand(
            money=money,
            available_land=available_land,
        ):
            return 0

        return min(
            available_land,
            int(money // 5000),
        )