"""
crop_priority_evaluator.py

Crop Priority Evaluator for the Kaggriculture AI Agent.

Ranks crops according to profitability.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CropPriorityEvaluator:
    """
    Evaluates crop profitability.
    """

    def score(
        self,
        *,
        price: float,
        yield_units: int,
        grow_days: int,
    ) -> float:
        """
        Profit score.

        Higher is better.
        """

        if grow_days <= 0:
            grow_days = 1

        return (price * yield_units) / grow_days

    # ---------------------------------------------------------

    def best_crop(
        self,
        crops: dict,
    ):
        """
        Return the crop with the highest score.

        crops format:

        {
            "MELON": {
                "price": ...,
                "yield_units": ...,
                "grow_days": ...
            }
        }
        """

        if not crops:
            return None

        best_name = None
        best_score = float("-inf")

        for name, values in crops.items():

            score = self.score(
                price=values["price"],
                yield_units=values["yield_units"],
                grow_days=values["grow_days"],
            )

            if score > best_score:

                best_score = score
                best_name = name

        return best_name