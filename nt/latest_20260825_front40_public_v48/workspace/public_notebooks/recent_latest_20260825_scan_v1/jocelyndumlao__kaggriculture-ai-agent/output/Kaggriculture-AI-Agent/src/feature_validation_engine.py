"""
feature_validation_engine.py

Feature Validation Engine for the Kaggriculture AI Agent.

Validates extracted features before they are
used by learning and decision-making modules.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import math


class FeatureValidationEngine:
    """
    Validate feature dictionaries.
    """

    def valid(
        self,
        features: dict[str, float],
    ) -> bool:
        """
        Return True if all features are valid.
        """

        return len(self.errors(features)) == 0

    # ---------------------------------------------------------

    def errors(
        self,
        features: dict[str, float],
    ) -> list[str]:
        """
        Return validation errors.
        """

        issues = []

        for name, value in features.items():

            if not isinstance(value, (int, float)):
                issues.append(
                    f"{name}: invalid type"
                )
                continue

            if math.isnan(value):
                issues.append(
                    f"{name}: NaN"
                )

            if math.isinf(value):
                issues.append(
                    f"{name}: infinite"
                )

        return issues

    # ---------------------------------------------------------

    def sanitize(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """
        Replace invalid values with zero.
        """

        clean = {}

        for name, value in features.items():

            if (
                isinstance(value, (int, float))
                and not math.isnan(value)
                and not math.isinf(value)
            ):
                clean[name] = float(value)
            else:
                clean[name] = 0.0

        return clean