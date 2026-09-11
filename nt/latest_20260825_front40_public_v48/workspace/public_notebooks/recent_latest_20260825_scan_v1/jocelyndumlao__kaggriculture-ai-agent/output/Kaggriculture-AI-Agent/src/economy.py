"""
economy.py

Economy module for the Kaggriculture AI Agent.

Responsible for investment decisions and money management.

Author: Jocelyn Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class Economy:
    """
    Handles financial decisions.
    """

    def __init__(self):

        self.money = 0

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    def update(self, money: float):
        """
        Update current money.
        """
        self.money = money

    # ---------------------------------------------------------
    # Affordability
    # ---------------------------------------------------------

    def can_afford(self, cost: float) -> bool:
        """
        Check whether enough money is available.
        """
        return self.money >= cost

    # ---------------------------------------------------------
    # Investment Decisions
    # ---------------------------------------------------------

    def should_buy_seed(self, cost: float) -> bool:
        """
        Decide whether purchasing seeds is affordable.
        """
        return self.can_afford(cost)

    def should_buy_animal(self, cost: float) -> bool:
        """
        Decide whether purchasing livestock is affordable.
        """
        return self.can_afford(cost)

    def should_buy_land(self, cost: float) -> bool:
        """
        Buy land only when plenty of money is available.
        """
        return self.money >= cost + 500

    def should_hire_worker(self, hire_cost: float) -> bool:
        """
        Hire workers only when a cash reserve remains.
        """
        return self.money >= hire_cost + 250

    # ---------------------------------------------------------
    # Reserve
    # ---------------------------------------------------------

    def reserve(self) -> float:
        """
        Keep a minimum emergency reserve.
        """
        return 250.0

    def available_cash(self) -> float:
        """
        Cash available after reserve.
        """
        return max(0.0, self.money - self.reserve())

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):

        return {
            "money": self.money,
            "available_cash": self.available_cash(),
            "reserve": self.reserve(),
        }