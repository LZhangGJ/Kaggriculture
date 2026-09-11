"""
crop_strategy.py

Intelligent crop selection for the Kaggriculture AI Agent.

Chooses the most profitable crop while considering:

- Market prices
- Seed cost
- Yield
- Growth duration
- Remaining season length

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.crops import CROPS


class CropStrategy:
    """
    Crop recommendation engine.
    """

    def __init__(self, parser):
        self.parser = parser
        self.market = parser.prices

    # -----------------------------------------------------
    # Basic Profit
    # -----------------------------------------------------

    def expected_profit(self, crop: str) -> float:
        """
        Estimate total profit from planting one crop.
        """

        info = CROPS[crop]

        price = self.market.get(crop, info.base_price)

        revenue = price * info.max_yield

        cost = info.seed_cost

        return revenue - cost

    # -----------------------------------------------------
    # Profit per Day
    # -----------------------------------------------------

    def crop_score(self, crop: str) -> float:
        """
        Profit normalized by growth duration.
        """

        info = CROPS[crop]

        profit = self.expected_profit(crop)

        growth_days = max(info.first_yield_day, 1)

        return profit / growth_days

    # -----------------------------------------------------
    # Season Adjustment
    # -----------------------------------------------------

    def adjusted_crop_score(self, crop: str) -> float:
        """
        Penalize slow crops late in the season.
        """

        day = getattr(self.parser, "day", 1)

        remaining = max(30 - day, 0)

        return self.crop_score_with_remaining_days(
            crop,
            remaining,
        )

    # -----------------------------------------------------
    # Remaining Days Score
    # -----------------------------------------------------

    def crop_score_with_remaining_days(
        self,
        crop: str,
        remaining_days: int,
    ) -> float:
        """
        Score considering remaining season length.
        """

        info = CROPS[crop]

        # Crop cannot mature before season ends.
        if remaining_days < info.first_yield_day:
            return -1e9

        score = self.crop_score(crop)

        # Penalize long-growing crops near season end.
        if remaining_days <= 5:

            if info.first_yield_day >= 10:
                score *= 0.20

            elif info.first_yield_day >= 8:
                score *= 0.50

        return score

    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    def rank_crops(self, remaining_days: int | None = None):
        """
        Return all crops ranked by score.
        """

        if remaining_days is None:
            remaining_days = max(
                30 - getattr(self.parser, "day", 1),
                0,
            )

        ranking = []

        for crop in CROPS:

            ranking.append(
                (
                    crop,
                    self.crop_score_with_remaining_days(
                        crop,
                        remaining_days,
                    ),
                )
            )

        ranking.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return ranking

    # -----------------------------------------------------
    # Best Crop
    # -----------------------------------------------------

    def best_crop(
        self,
        remaining_days: int | None = None,
    ) -> str:
        """
        Return the highest-ranked crop.
        """

        return self.rank_crops(
            remaining_days,
        )[0][0]