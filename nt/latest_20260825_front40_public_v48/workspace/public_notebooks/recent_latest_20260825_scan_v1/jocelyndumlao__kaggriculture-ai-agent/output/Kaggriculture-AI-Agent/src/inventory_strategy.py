"""
inventory_strategy.py

Inventory management for the Kaggriculture AI Agent.

Decides whether products should be sold, stored,
or reserved for future use.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class InventoryStrategy:
    """
    Intelligent inventory management.
    """

    def __init__(self, parser, market):
        self.parser = parser
        self.market = market

        self.shed = parser.shed
        self.seeds = parser.seeds

    # ---------------------------------------------------------
    # Sell Decision
    # ---------------------------------------------------------

    def should_sell(self, product: str) -> bool:
        """
        Decide whether a product should be sold.
        """

        return self.market.should_sell(product)

    # ---------------------------------------------------------
    # Store Decision
    # ---------------------------------------------------------

    def should_store(self, product: str) -> bool:
        """
        Store products if selling is not recommended.
        """

        return not self.should_sell(product)

    # ---------------------------------------------------------
    # Keep Seeds
    # ---------------------------------------------------------

    def should_keep_seed(self, crop: str) -> bool:
        """
        Keep seeds if inventory is low.
        """

        return self.seeds.get(crop, 0) < 5

    # ---------------------------------------------------------
    # Inventory Value
    # ---------------------------------------------------------

    def inventory_value(self) -> float:
        """
        Total market value of the shed.
        """

        total = 0

        for product, amount in self.shed.items():

            total += amount * self.market.current_price(product)

        return total

    # ---------------------------------------------------------
    # Liquidation
    # ---------------------------------------------------------

    def liquidate_inventory(self):
        """
        Return products that should be sold immediately.
        """

        products = []

        for product, amount in self.shed.items():

            if amount <= 0:
                continue

            products.append(product)

        return products