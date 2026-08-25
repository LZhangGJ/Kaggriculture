"""
animal_care_analyzer.py

Animal Care Analyzer for the Kaggriculture AI Agent.

Analyzes feeding and care priorities
for farm animals.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AnimalCareAnalyzer:
    """
    Analyze animal care priorities.
    """

    # ---------------------------------------------------------

    def needs_feed(
        self,
        tile: dict,
    ) -> bool:
        """
        Return True if the animal needs feeding.
        """

        return (
            tile.get("animal") is not None
            and not tile.get("fed_today", False)
        )

    # ---------------------------------------------------------

    def needs_care(
        self,
        tile: dict,
    ) -> bool:
        """
        Return True if the animal needs care.
        """

        return (
            tile.get("animal") is not None
            and not tile.get("cared_today", False)
        )

    # ---------------------------------------------------------

    def urgent(
        self,
        animals: list[dict],
    ) -> dict | None:
        """
        Return the highest-priority animal.
        """

        if not animals:
            return None

        return max(
            animals,
            key=lambda animal: animal.get(
                "consecutive_unfed",
                0,
            ),
        )

    # ---------------------------------------------------------

    def feed_count(
        self,
        animals: list[dict],
    ) -> int:
        """
        Return number of animals needing food.
        """

        return sum(
            self.needs_feed(animal)
            for animal in animals
        )

    # ---------------------------------------------------------

    def care_count(
        self,
        animals: list[dict],
    ) -> int:
        """
        Return number of animals needing care.
        """

        return sum(
            self.needs_care(animal)
            for animal in animals
        )