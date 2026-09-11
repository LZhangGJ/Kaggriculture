"""
feature_cache_engine.py

Feature Cache Engine for the Kaggriculture AI Agent.

Caches extracted feature dictionaries using a
state identifier to avoid repeated computation.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FeatureCacheEngine:
    """
    Cache extracted features.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}

    # ---------------------------------------------------------

    def store(
        self,
        state_id: str,
        features: dict,
    ) -> None:
        """
        Store features for a state.
        """

        self._cache[state_id] = dict(features)

    # ---------------------------------------------------------

    def retrieve(
        self,
        state_id: str,
    ) -> dict | None:
        """
        Retrieve cached features.
        """

        return self._cache.get(state_id)

    # ---------------------------------------------------------

    def contains(
        self,
        state_id: str,
    ) -> bool:
        """
        Return True if the cache contains the state.
        """

        return state_id in self._cache

    # ---------------------------------------------------------

    def clear(
        self,
    ) -> None:
        """
        Remove all cached entries.
        """

        self._cache.clear()

    # ---------------------------------------------------------

    def size(
        self,
    ) -> int:
        """
        Return the number of cached states.
        """

        return len(self._cache)