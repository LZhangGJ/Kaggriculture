"""
crop_profit_evaluator.py

Crop Profit Evaluator for the Kaggriculture AI Agent.

Estimates crop profitability.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CropProfitEvaluator:
    """
    Estimates crop profit.
    """

    def evaluate(
        self,
        crop: dict,
        market: dict,
    ) -> float:
        """
        Return estimated profit.
        """

        crop_type = crop["type"]

        seed_cost = market[crop_type]["seed_cost"]
        sell_price = market[crop_type]["sell_price"]

        return sell_price - seed_cost