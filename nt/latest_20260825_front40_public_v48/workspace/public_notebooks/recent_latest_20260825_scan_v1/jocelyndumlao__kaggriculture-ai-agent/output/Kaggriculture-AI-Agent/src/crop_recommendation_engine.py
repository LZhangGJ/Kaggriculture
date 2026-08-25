"""
crop_recommendation_engine.py

Crop Recommendation Engine for the Kaggriculture AI Agent.

Recommends the best crop to plant using
profitability, season awareness, and market value.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CropRecommendationEngine:
    """
    Recommends the best crop.
    """

    # ---------------------------------------------------------

    def recommend(
        self,
        crops: list[dict],
    ) -> dict | None:
        """
        Return the crop with the highest score.

        Expected crop format:

        {
            "name": "CARROT",
            "score": 123.4,
        }
        """

        if not crops:
            return None

        return max(
            crops,
            key=lambda crop: crop.get(
                "score",
                float("-inf"),
            ),
        )

    # ---------------------------------------------------------

    def recommendation_name(
        self,
        crops: list[dict],
    ) -> str | None:
        """
        Return only the crop name.
        """

        recommendation = self.recommend(crops)

        if recommendation is None:
            return None

        return recommendation.get("name")

    # ---------------------------------------------------------

    def recommendation_score(
        self,
        crops: list[dict],
    ) -> float:
        """
        Return the score of the recommendation.
        """

        recommendation = self.recommend(crops)

        if recommendation is None:
            return 0.0

        return float(
            recommendation.get(
                "score",
                0.0,
            )
        )