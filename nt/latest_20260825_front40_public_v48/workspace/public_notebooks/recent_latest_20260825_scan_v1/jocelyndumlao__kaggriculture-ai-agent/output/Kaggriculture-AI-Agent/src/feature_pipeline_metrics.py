"""
feature_pipeline_metrics.py

Feature Pipeline Metrics for the Kaggriculture AI Agent.

Tracks operational statistics for the feature
processing pipeline.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeaturePipelineMetrics:
    """
    Collect feature pipeline metrics.
    """

    def __init__(self):
        self.reset()

    # ---------------------------------------------------------

    def record_processed(self) -> None:
        self.processed += 1

    # ---------------------------------------------------------

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    # ---------------------------------------------------------

    def record_validation_failure(self) -> None:
        self.validation_failures += 1

    # ---------------------------------------------------------

    def cache_hit_rate(self) -> float:

        if self.processed == 0:
            return 0.0

        return self.cache_hits / self.processed

    # ---------------------------------------------------------

    def summary(self) -> dict:

        return {
            "processed": self.processed,
            "cache_hits": self.cache_hits,
            "validation_failures": self.validation_failures,
            "cache_hit_rate": self.cache_hit_rate(),
        }

    # ---------------------------------------------------------

    def reset(self) -> None:

        self.processed = 0
        self.cache_hits = 0
        self.validation_failures = 0