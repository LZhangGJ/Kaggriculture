"""
market_trend_analyzer.py

Market Trend Analyzer for the Kaggriculture AI Agent.

Uses PriceHistory to analyze market movement.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.price_history import PriceHistory


class MarketTrendAnalyzer:
    """
    Analyzes historical market prices.
    """

    def __init__(
        self,
        history: PriceHistory,
    ):
        self.history = history

    # ---------------------------------------------------------

    def trend(
        self,
        product: str,
    ) -> str:
        """
        Return price trend.
        """

        return self.history.trend(product)

    # ---------------------------------------------------------

    def average_price(
        self,
        product: str,
    ) -> float:
        """
        Return historical average price.
        """

        return self.history.average(product)

    # ---------------------------------------------------------

    def latest_price(
        self,
        product: str,
    ) -> float:
        """
        Return latest observed price.
        """

        return self.history.latest(product)

    # ---------------------------------------------------------

    def is_good_time_to_sell(
        self,
        product: str,
    ) -> bool:
        """
        Sell when the latest price is at least
        the historical average.
        """

        return (
            self.latest_price(product)
            >= self.average_price(product)
        )

    # ---------------------------------------------------------

    def is_price_rising(
        self,
        product: str,
    ) -> bool:

        return self.trend(product) == "UP"

    # ---------------------------------------------------------

    def is_price_falling(
        self,
        product: str,
    ) -> bool:

        return self.trend(product) == "DOWN"