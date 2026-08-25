"""
feature_extraction_engine.py

Feature Extraction Engine for the Kaggriculture AI Agent.

Extracts reusable numerical features from the
current game state for planning, evaluation,
learning, and prediction.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureExtractionEngine:
    """
    Extract numerical features from a game state.
    """

    def extract(
        self,
        game_state: dict,
    ) -> dict:
        """
        Return extracted features.
        """

        inventory = game_state.get("inventory", {})
        market = game_state.get("market", {})
        crops = game_state.get("crops", [])
        animals = game_state.get("animals", [])

        return {
            "money": float(game_state.get("money", 0)),
            "day": int(game_state.get("day", 0)),
            "hour": int(game_state.get("hour", 0)),
            "crop_count": len(crops),
            "animal_count": len(animals),
            "inventory_size": len(inventory),
            "market_items": len(market.get("prices", {})),
        }

    # ---------------------------------------------------------

    def vector(
        self,
        game_state: dict,
    ) -> list[float]:
        """
        Return a feature vector.
        """

        features = self.extract(game_state)

        return [
            float(features["money"]),
            float(features["day"]),
            float(features["hour"]),
            float(features["crop_count"]),
            float(features["animal_count"]),
            float(features["inventory_size"]),
            float(features["market_items"]),
        ]

    # ---------------------------------------------------------

    def feature_names(
        self,
    ) -> list[str]:
        """
        Return feature names.
        """

        return [
            "money",
            "day",
            "hour",
            "crop_count",
            "animal_count",
            "inventory_size",
            "market_items",
        ]