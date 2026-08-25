"""
market_opportunity_evaluator.py

Market Opportunity Evaluator for the Kaggriculture AI Agent.

Evaluates whether current market prices
represent a good selling opportunity.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class MarketOpportunityEvaluator:
    """
    Evaluates market opportunities.
    """

    # ---------------------------------------------------------

    def is_good_price(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> bool:
        """
        Return True when the current price is
        at least as good as the average.
        """

        return current_price >= average_price

    # ---------------------------------------------------------

    def price_ratio(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> float:
        """
        Return current / average.
        """

        if average_price <= 0:
            return 0.0

        return current_price / average_price

    # ---------------------------------------------------------

    def score(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> float:
        """
        Return a normalized market score.
        """

        return self.price_ratio(
            current_price=current_price,
            average_price=average_price,
        )