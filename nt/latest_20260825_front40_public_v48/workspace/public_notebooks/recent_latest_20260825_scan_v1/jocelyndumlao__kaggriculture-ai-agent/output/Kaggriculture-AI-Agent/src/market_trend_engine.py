"""
market_trend_engine.py

Market Trend Engine for the Kaggriculture AI Agent.

Tracks historical market prices and identifies
basic price trends.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketTrendEngine:
    """
    Analyze simple market trends.
    """

    def __init__(self):

        self.history = {}

    # ---------------------------------------------------------

    def record(
        self,
        item: str,
        price: float,
    ) -> None:
        """
        Store a market price observation.
        """

        self.history.setdefault(
            item,
            [],
        ).append(
            float(price)
        )

    # ---------------------------------------------------------

    def latest_price(
        self,
        item: str,
    ) -> float:
        """
        Return the latest recorded price.
        """

        if item not in self.history:
            return 0.0

        return self.history[item][-1]

    # ---------------------------------------------------------

    def average_price(
        self,
        item: str,
    ) -> float:
        """
        Return the average observed price.
        """

        if item not in self.history:
            return 0.0

        prices = self.history[item]

        return sum(prices) / len(prices)

    # ---------------------------------------------------------

    def trend(
        self,
        item: str,
    ) -> str:
        """
        Return the current trend.
        """

        prices = self.history.get(
            item,
            [],
        )

        if len(prices) < 2:
            return "UNKNOWN"

        if prices[-1] > prices[-2]:
            return "RISING"

        if prices[-1] < prices[-2]:
            return "FALLING"

        return "STABLE"

    # ---------------------------------------------------------

    def observations(
        self,
        item: str,
    ) -> int:
        """
        Return the number of observations.
        """

        return len(
            self.history.get(
                item,
                [],
            )
        )