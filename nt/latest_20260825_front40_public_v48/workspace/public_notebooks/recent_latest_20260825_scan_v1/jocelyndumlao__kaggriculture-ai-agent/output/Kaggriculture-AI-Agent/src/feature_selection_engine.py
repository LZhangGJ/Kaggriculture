"""
feature_selection_engine.py

Feature Selection Engine for the Kaggriculture AI Agent.

Selects subsets of extracted features for
specific learning, prediction, and planning
components.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureSelectionEngine:
    """
    Select subsets of features.
    """

    def select(
        self,
        features: dict[str, float],
        names: list[str],
    ) -> dict[str, float]:
        """
        Return selected features.
        """

        return {
            name: features[name]
            for name in names
            if name in features
        }

    # ---------------------------------------------------------

    def remove(
        self,
        features: dict[str, float],
        names: list[str],
    ) -> dict[str, float]:
        """
        Remove selected features.
        """

        return {
            key: value
            for key, value in features.items()
            if key not in names
        }

    # ---------------------------------------------------------

    def available(
        self,
        features: dict[str, float],
    ) -> list[str]:
        """
        Return available feature names.
        """

        return sorted(features.keys())