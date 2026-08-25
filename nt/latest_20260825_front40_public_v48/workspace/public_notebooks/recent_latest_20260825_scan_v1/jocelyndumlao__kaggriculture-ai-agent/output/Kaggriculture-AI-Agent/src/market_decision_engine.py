"""
market_decision_engine.py

Market Decision Engine for the Kaggriculture AI Agent.

Determines whether products should be bought or sold
according to current market prices.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketDecisionEngine:
    """
    Market-aware buy/sell decisions.
    """

    def should_sell(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> bool:
        """
        Sell when price is at or above average.
        """

        return current_price >= average_price

    # ---------------------------------------------------------

    def should_buy(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> bool:
        """
        Buy when price is below average.
        """

        return current_price < average_price

    # ---------------------------------------------------------

    def price_ratio(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> float:
        """
        Return price ratio.
        """

        if average_price <= 0:
            return 0.0

        return float(current_price / average_price)

    # ---------------------------------------------------------

    def market_state(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> str:
        """
        Classify market conditions.
        """

        ratio = self.price_ratio(
            current_price=current_price,
            average_price=average_price,
        )

        if ratio >= 1.20:
            return "HOT"

        if ratio <= 0.80:
            return "CHEAP"

        return "NORMAL"