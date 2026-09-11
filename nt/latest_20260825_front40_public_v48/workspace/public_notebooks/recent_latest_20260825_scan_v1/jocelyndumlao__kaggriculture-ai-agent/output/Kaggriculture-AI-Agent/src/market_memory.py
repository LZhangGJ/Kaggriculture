"""
market_memory.py

Stores historical market prices and provides
simple trend analysis.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketMemory:
    """
    Remembers market prices across turns.
    """

    def __init__(self):
        self.history = {}

    # -----------------------------------------------------

    def update(self, prices: dict):
        """
        Store the latest market prices.
        """

        for product, price in prices.items():

            self.history.setdefault(product, []).append(price)

            # Keep only the latest 20 observations
            if len(self.history[product]) > 20:
                self.history[product].pop(0)

    # -----------------------------------------------------

    def latest(self, product: str):

        values = self.history.get(product, [])

        if not values:
            return None

        return values[-1]

    # -----------------------------------------------------

    def average(self, product: str):

        values = self.history.get(product, [])

        if not values:
            return None

        return sum(values) / len(values)

    # -----------------------------------------------------

    def trend(self, product: str):
        """
        Return:

            "UP"
            "DOWN"
            "FLAT"
            "UNKNOWN"
        """

        values = self.history.get(product, [])

        if len(values) < 2:
            return "UNKNOWN"

        if values[-1] > values[-2]:
            return "UP"

        if values[-1] < values[-2]:
            return "DOWN"

        return "FLAT"