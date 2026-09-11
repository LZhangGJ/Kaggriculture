"""
budget_optimizer.py

Budget management for the Kaggriculture AI Agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class BudgetOptimizer:
    """
    Controls spending decisions.
    """

    def __init__(self, money: int, reserve: int = 200):
        self.money = money
        self.reserve = reserve

    # ---------------------------------------------------------

    def spendable(self) -> int:
        """
        Money available after keeping a reserve.
        """
        return max(0, self.money - self.reserve)

    # ---------------------------------------------------------

    def can_afford(self, cost: int) -> bool:
        """
        Return True if an item can be purchased safely.
        """
        return cost <= self.spendable()

    # ---------------------------------------------------------

    def remaining_after_purchase(self, cost: int) -> int:
        """
        Remaining money after buying an item.
        """
        return self.money - cost

    # ---------------------------------------------------------

    def reserve_ratio(self) -> float:
        """
        Fraction of money kept as reserve.
        """
        if self.money == 0:
            return 0.0

        return self.reserve / self.money