"""
dynamic_market_analyzer.py

Dynamic Market Analyzer for the Kaggriculture AI Agent.

Provides simple market analysis utilities.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DynamicMarketAnalyzer:
    """
    Analyze market prices.
    """

    def profit_margin(
        self,
        buy_price: float,
        sell_price: float,
    ) -> float:
        """
        Return the absolute profit margin.
        """
        return sell_price - buy_price

    # ---------------------------------------------------------

    def is_profitable(
        self,
        buy_price: float,
        sell_price: float,
    ) -> bool:
        """
        Return True if selling is profitable.
        """
        return sell_price > buy_price

    # ---------------------------------------------------------

    def best_buy(
        self,
        market: dict,
    ) -> str:
        """
        Return the cheapest item.
        """
        return min(
            market,
            key=lambda item: market[item]["buy_price"],
        )

    # ---------------------------------------------------------

    def best_sell(
        self,
        market: dict,
    ) -> str:
        """
        Return the highest-value item.
        """
        return max(
            market,
            key=lambda item: market[item]["sell_price"],
        )

    # ---------------------------------------------------------

    def spread(
        self,
        market: dict,
        item: str,
    ) -> float:
        """
        Return the buy/sell spread for an item.
        """
        prices = market[item]

        return (
            prices["sell_price"]
            - prices["buy_price"]
        )