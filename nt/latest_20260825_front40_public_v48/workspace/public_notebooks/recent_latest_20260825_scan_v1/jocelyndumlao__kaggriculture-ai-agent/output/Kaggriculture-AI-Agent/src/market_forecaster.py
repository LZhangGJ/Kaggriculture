"""
market_forecaster.py

Simple market forecasting for the Kaggriculture AI Agent.

Uses MarketMemory to estimate future price trends.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketForecaster:
    """
    Forecast future market direction using historical prices.
    """

    def __init__(self, memory):
        self.memory = memory

    # ---------------------------------------------------------
    # Price Trend
    # ---------------------------------------------------------

    def trend(self, product: str) -> float:
        """
        Estimate the market trend using the two most recent prices.

        Positive value  -> Rising market
        Negative value  -> Falling market
        Zero            -> Stable or insufficient history
        """

        history = self.memory.history.get(product, [])

        if len(history) < 2:
            return 0.0

        return history[-1] - history[-2]

    # ---------------------------------------------------------
    # Rising Market
    # ---------------------------------------------------------

    def will_rise(self, product: str) -> bool:
        """
        Return True if the price is trending upward.
        """

        return self.trend(product) > 0

    # ---------------------------------------------------------
    # Falling Market
    # ---------------------------------------------------------

    def will_fall(self, product: str) -> bool:
        """
        Return True if the price is trending downward.
        """

        return self.trend(product) < 0

    # ---------------------------------------------------------
    # Price Prediction
    # ---------------------------------------------------------

    def predicted_price(self, product: str) -> float:
        """
        Predict the next market price using a simple linear trend.
        """

        history = self.memory.history.get(product, [])

        if not history:
            return 0.0

        return history[-1] + self.trend(product)