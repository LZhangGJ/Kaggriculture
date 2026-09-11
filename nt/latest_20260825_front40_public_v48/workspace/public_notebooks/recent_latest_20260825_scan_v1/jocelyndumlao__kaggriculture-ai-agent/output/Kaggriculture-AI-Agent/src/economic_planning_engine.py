"""
economic_planning_engine.py

Economic Planning Engine for the Kaggriculture AI Agent.

Generates simple economic recommendations
based on forecasted market direction.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class EconomicPlanningEngine:
    """
    Generate economic decisions.
    """

    ACTIONS = {
        "UP": "HOLD",
        "DOWN": "SELL",
        "STABLE": "WAIT",
        "UNKNOWN": "WAIT",
    }

    # ---------------------------------------------------------

    def recommend(
        self,
        direction: str,
    ) -> str:
        """
        Return recommended economic action.
        """

        return self.ACTIONS.get(
            direction.upper(),
            "WAIT",
        )

    # ---------------------------------------------------------

    def should_sell(
        self,
        direction: str,
    ) -> bool:
        """
        Return True if selling is recommended.
        """

        return (
            self.recommend(direction)
            == "SELL"
        )

    # ---------------------------------------------------------

    def should_hold(
        self,
        direction: str,
    ) -> bool:
        """
        Return True if holding is recommended.
        """

        return (
            self.recommend(direction)
            == "HOLD"
        )

    # ---------------------------------------------------------

    def should_wait(
        self,
        direction: str,
    ) -> bool:
        """
        Return True if waiting is recommended.
        """

        return (
            self.recommend(direction)
            == "WAIT"
        )