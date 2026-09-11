"""
feature_pipeline_coordinator.py

Feature Pipeline Coordinator for the Kaggriculture AI Agent.

Coordinates feature extraction, validation,
normalization, statistics, caching, and
selection into one reusable pipeline.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.feature_cache_engine import FeatureCacheEngine
from src.feature_extraction_engine import FeatureExtractionEngine
from src.feature_normalization_engine import FeatureNormalizationEngine
from src.feature_selection_engine import FeatureSelectionEngine
from src.feature_statistics_engine import FeatureStatisticsEngine
from src.feature_validation_engine import FeatureValidationEngine


class FeaturePipelineCoordinator:
    """
    Coordinate the complete feature pipeline.
    """

    def __init__(self):

        self.extractor = FeatureExtractionEngine()
        self.normalizer = FeatureNormalizationEngine()
        self.selector = FeatureSelectionEngine()
        self.validator = FeatureValidationEngine()
        self.statistics = FeatureStatisticsEngine()
        self.cache = FeatureCacheEngine()

    # ---------------------------------------------------------

    def process(
        self,
        *,
        state_id: str,
        game_state: dict,
        maximums: dict[str, float] | None = None,
        selected: list[str] | None = None,
    ) -> dict:
        """
        Execute the feature pipeline.
        """

        cached = self.cache.retrieve(
            state_id,
        )

        if cached is not None:
            return cached

        features = self.extractor.extract(
            game_state,
        )

        if maximums is not None:
            features = self.normalizer.normalize(
                features,
                maximums,
            )

        if selected is not None:
            features = self.selector.select(
                features,
                selected,
            )

        if not self.validator.valid(
            features,
        ):
            features = self.validator.sanitize(
                features,
            )

        self.statistics.update(
            features,
        )

        self.cache.store(
            state_id,
            features,
        )

        return features

    # ---------------------------------------------------------

    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached features.
        """

        self.cache.clear()