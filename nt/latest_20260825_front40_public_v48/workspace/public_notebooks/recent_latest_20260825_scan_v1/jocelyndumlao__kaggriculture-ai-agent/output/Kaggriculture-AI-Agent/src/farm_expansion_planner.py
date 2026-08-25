"""
farm_expansion_planner.py

Farm Expansion Planner for the Kaggriculture AI Agent.

Provides helper methods for deciding when
farm expansion is possible.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FarmExpansionPlanner:
    """
    Plan farm expansion.
    """

    # ---------------------------------------------------------

    def can_expand(
        self,
        *,
        money: int,
        expansion_cost: int,
    ) -> bool:
        """
        Return True if expansion is affordable.
        """

        return money >= expansion_cost

    # ---------------------------------------------------------

    def remaining_quadrants(
        self,
        unlocked: list[str],
    ) -> int:
        """
        Return number of locked quadrants.
        """

        all_quadrants = {
            "NW",
            "NE",
            "SW",
            "SE",
        }

        return len(
            all_quadrants - set(unlocked)
        )

    # ---------------------------------------------------------

    def fully_expanded(
        self,
        unlocked: list[str],
    ) -> bool:
        """
        Return True if every quadrant
        is unlocked.
        """

        return (
            self.remaining_quadrants(unlocked)
            == 0
        )

    # ---------------------------------------------------------

    def next_priority(
        self,
        unlocked: list[str],
    ) -> str | None:
        """
        Return the next suggested
        quadrant to unlock.
        """

        for quadrant in (
            "NE",
            "SW",
            "SE",
        ):
            if quadrant not in unlocked:
                return quadrant

        return None

    # ---------------------------------------------------------

    def expansion_score(
        self,
        *,
        money: int,
        expansion_cost: int,
    ) -> float:
        """
        Return affordability score.
        """

        if expansion_cost <= 0:
            return 0.0

        return money / expansion_cost