"""
feature_normalization_engine.py

Feature Normalization Engine for the Kaggriculture AI Agent.

Normalizes extracted numerical features into a
consistent range for learning and prediction.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureNormalizationEngine:
    """
    Normalize feature dictionaries.
    """

    def normalize(
        self,
        features: dict[str, float],
        maximums: dict[str, float],
    ) -> dict[str, float]:
        """
        Normalize features into [0, 1].
        """

        normalized = {}

        for key, value in features.items():

            maximum = maximums.get(
                key,
                1.0,
            )

            if maximum <= 0:
                normalized[key] = 0.0
            else:
                normalized[key] = min(
                    1.0,
                    value / maximum,
                )

        return normalized

    # ---------------------------------------------------------

    def normalize_vector(
        self,
        values: list[float],
        maximums: list[float],
    ) -> list[float]:
        """
        Normalize a feature vector.
        """

        result = []

        for value, maximum in zip(
            values,
            maximums,
        ):

            if maximum <= 0:
                result.append(0.0)
            else:
                result.append(
                    min(
                        1.0,
                        value / maximum,
                    )
                )

        return result

    # ---------------------------------------------------------

    def clip(
        self,
        value: float,
    ) -> float:
        """
        Clip a normalized value.
        """

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )