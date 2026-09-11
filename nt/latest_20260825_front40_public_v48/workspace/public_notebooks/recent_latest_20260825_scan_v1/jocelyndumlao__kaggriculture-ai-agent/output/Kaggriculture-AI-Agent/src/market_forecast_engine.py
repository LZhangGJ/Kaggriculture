"""
market_forecast_engine.py

Market Forecast Engine for the Kaggriculture AI Agent.

Uses recent market observations to estimate
future price direction.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketForecastEngine:
    """
    Forecast future market prices.
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
        Record a market observation.
        """

        self.history.setdefault(
            item,
            [],
        ).append(
            float(price)
        )

    # ---------------------------------------------------------

    def forecast(
        self,
        item: str,
    ) -> float:
        """
        Forecast the next price.

        Uses a simple moving average.
        """

        prices = self.history.get(
            item,
            [],
        )

        if not prices:
            return 0.0

        if len(prices) == 1:
            return prices[0]

        return sum(prices[-3:]) / len(prices[-3:])

    # ---------------------------------------------------------

    def expected_direction(
        self,
        item: str,
    ) -> str:
        """
        Estimate future direction.
        """

        prices = self.history.get(
            item,
            [],
        )

        if len(prices) < 2:
            return "UNKNOWN"

        prediction = self.forecast(
            item,
        )

        latest = prices[-1]

        if prediction > latest:
            return "UP"

        if prediction < latest:
            return "DOWN"

        return "STABLE"

    # ---------------------------------------------------------

    def has_history(
        self,
        item: str,
    ) -> bool:
        """
        Return True if observations exist.
        """

        return item in self.history