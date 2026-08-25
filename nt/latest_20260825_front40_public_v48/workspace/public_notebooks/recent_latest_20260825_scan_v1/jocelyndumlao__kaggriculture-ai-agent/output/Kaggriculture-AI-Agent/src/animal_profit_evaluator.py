"""
animal_profit_evaluator.py

Animal Profit Evaluator for the Kaggriculture AI Agent.

Estimates animal profitability.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AnimalProfitEvaluator:
    """
    Estimates expected animal profit.
    """

    def evaluate(
        self,
        animal: dict,
        market: dict,
    ) -> float:
        """
        Estimate profit for an animal.
        """

        animal_type = animal["type"]

        purchase_cost = market[animal_type]["purchase_cost"]
        product_value = market[animal_type]["product_value"]
        feed_cost = market[animal_type]["feed_cost"]

        return product_value - purchase_cost - feed_cost