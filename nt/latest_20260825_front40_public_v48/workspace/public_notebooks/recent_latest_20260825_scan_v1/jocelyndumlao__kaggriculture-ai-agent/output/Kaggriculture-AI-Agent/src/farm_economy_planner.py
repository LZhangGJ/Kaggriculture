"""
farm_economy_planner.py

Farm Economy Planner for the Kaggriculture AI Agent.

Manages the farm budget by recommending how much
money should be reserved for different activities.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FarmEconomyPlanner:
    """
    Manage the farm budget.
    """

    def reserve_cash(
        self,
        *,
        money: float,
        reserve_ratio: float = 0.20,
    ) -> float:
        """
        Cash to keep in reserve.
        """

        return money * reserve_ratio

    # ---------------------------------------------------------

    def spendable_cash(
        self,
        *,
        money: float,
        reserve_ratio: float = 0.20,
    ) -> float:
        """
        Cash available for spending.
        """

        return max(
            0.0,
            money - self.reserve_cash(
                money=money,
                reserve_ratio=reserve_ratio,
            ),
        )

    # ---------------------------------------------------------

    def can_afford(
        self,
        *,
        money: float,
        cost: float,
        reserve_ratio: float = 0.20,
    ) -> bool:
        """
        Return True if purchase keeps reserve.
        """

        return (
            self.spendable_cash(
                money=money,
                reserve_ratio=reserve_ratio,
            )
            >= cost
        )

    # ---------------------------------------------------------

    def emergency_fund(
        self,
        *,
        money: float,
    ) -> float:
        """
        Emergency cash target.
        """

        return max(
            500.0,
            money * 0.10,
        )

    # ---------------------------------------------------------

    def investment_ratio(
        self,
        *,
        money: float,
        investment: float,
    ) -> float:
        """
        Fraction of money being invested.
        """

        if money <= 0:
            return 0.0

        return investment / money