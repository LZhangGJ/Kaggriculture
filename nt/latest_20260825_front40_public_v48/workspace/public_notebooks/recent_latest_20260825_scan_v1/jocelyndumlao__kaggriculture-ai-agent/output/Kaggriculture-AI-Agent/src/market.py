"""
market.py

Market Intelligence Module for Kaggriculture AI.

Analyzes market prices and provides economic
recommendations to the planner.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from statistics import mean


class Market:
    """
    Market Intelligence.

    Responsible for:

    - Reading current prices
    - Finding the best crop
    - Finding the best product to sell
    - Buy recommendations
    - Sell recommendations
    """

    def __init__(self, parser):
        self.parser = parser

        # Make Market compatible with lightweight test parsers.
        self.prices = getattr(parser, "prices", {})
        self.inventory = getattr(parser, "inventory", {})
        self.shed = getattr(parser, "shed", {})

    # ---------------------------------------------------------
    # Price Lookup
    # ---------------------------------------------------------

    def current_price(self, item: str) -> float:
        """Return the current market price."""
        return self.prices.get(item, 0)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def average_price(self) -> float:
        """Return the average market price."""

        values = list(self.prices.values())

        if not values:
            return 0

        return mean(values)

    def highest_price(self):
        """Return highest priced item."""

        if not self.prices:
            return None, 0

        item = max(self.prices, key=self.prices.get)

        return item, self.prices[item]

    def lowest_price(self):
        """Return lowest priced item."""

        if not self.prices:
            return None, 0

        item = min(self.prices, key=self.prices.get)

        return item, self.prices[item]

    # ---------------------------------------------------------
    # Crop Analysis
    # ---------------------------------------------------------

    def best_crop(self):
        """Return crop with highest current market price."""

        crops = (
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
            "MELON",
        )

        available = {
            crop: self.current_price(crop)
            for crop in crops
        }

        return max(available, key=available.get)

    # ---------------------------------------------------------
    # Selling
    # ---------------------------------------------------------

    def best_product_to_sell(self):
        """Return the most valuable product currently in storage."""

        if not self.shed:
            return None

        best_item = None
        best_price = -1

        for product, amount in self.shed.items():

            if amount <= 0:
                continue

            price = self.current_price(product)

            if price > best_price:
                best_price = price
                best_item = product

        return best_item

    def should_sell(self, product: str) -> bool:
        """Simple selling rule."""
        return self.current_price(product) > 0

    # ---------------------------------------------------------
    # Buying
    # ---------------------------------------------------------

    def should_buy_wheat(self) -> bool:
        return self.current_price("WHEAT") < 100

    def should_buy_fertilizer(self) -> bool:
        return self.current_price("FERTILIZER") < 60

    def should_buy_seed(self, crop: str) -> bool:
        return self.current_price(crop) > self.average_price()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def is_high_price(self, product: str) -> bool:
        return self.current_price(product) >= self.average_price()

    def is_low_price(self, product: str) -> bool:
        return self.current_price(product) < self.average_price()

    def summary(self):

        item, price = self.highest_price()

        return {
            "highest": item,
            "highest_price": price,
            "average": self.average_price(),
            "best_crop": self.best_crop(),
        }

    # ---------------------------------------------------------
    # Market Intelligence
    # ---------------------------------------------------------

    def is_good_sell_price(
        self,
        product: str,
        memory,
    ) -> bool:
        """
        Current price is at least 10% above the recent average.
        """

        current = self.prices.get(product)
        average = memory.average(product)

        if current is None or average is None:
            return False

        return current >= average * 1.10

    def is_good_buy_price(
        self,
        product: str,
        memory,
    ) -> bool:
        """
        Current price is at least 10% below the recent average.
        """

        current = self.prices.get(product)
        average = memory.average(product)

        if current is None or average is None:
            return False

        return current <= average * 0.90