"""
crop_planner.py

Crop Planner for the Kaggriculture AI Agent.

Plans which crop should be planted based on
the current season and crop profitability.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.scoring import CropScorer


class CropPlanner:
    """
    Plans crop selection.
    """

    def __init__(
        self,
        scorer: CropScorer,
    ):
        self.scorer = scorer

    # ---------------------------------------------------------

    def choose_crop(self):

        return self.scorer.best_crop()

    # ---------------------------------------------------------

    def can_plant(
        self,
        crop_name: str,
    ) -> bool:

        return self.scorer.can_finish(
            crop_name,
        )

    # ---------------------------------------------------------

    def affordable(
        self,
        crop_name: str,
    ) -> bool:

        return self.scorer.affordable(
            crop_name,
        )

    # ---------------------------------------------------------

    def score(
        self,
        crop_name: str,
    ) -> float:

        return self.scorer.score(
            crop_name,
        )