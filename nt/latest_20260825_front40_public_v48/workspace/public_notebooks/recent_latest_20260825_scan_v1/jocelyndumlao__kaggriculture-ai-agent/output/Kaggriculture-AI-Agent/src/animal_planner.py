"""
animal_planner.py

Animal Planner for the Kaggriculture AI Agent.

Plans animal-related decisions such as feeding,
harvesting, collecting products, and purchasing.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AnimalPlanner:
    """
    Plans animal actions.
    """

    # ---------------------------------------------------------

    def can_feed(
        self,
        animal: dict,
    ) -> bool:
        """
        Return True if the animal needs food.
        """

        return animal.get("hungry", False)

    # ---------------------------------------------------------

    def can_collect(
        self,
        animal: dict,
    ) -> bool:
        """
        Return True if an animal product is ready.
        """

        return animal.get("product_ready", False)

    # ---------------------------------------------------------

    def can_harvest(
        self,
        animal: dict,
    ) -> bool:
        """
        Return True if the animal can be harvested.
        """

        return animal.get("harvest_ready", False)

    # ---------------------------------------------------------

    def priority(
        self,
        animal: dict,
    ) -> float:
        """
        Compute animal priority.
        """

        score = 0.0

        if self.can_feed(animal):
            score += 40.0

        if self.can_collect(animal):
            score += 30.0

        if self.can_harvest(animal):
            score += 50.0

        return score

    # ---------------------------------------------------------

    def best_action(
        self,
        animal: dict,
    ) -> str:

        if self.can_harvest(animal):
            return "HARVEST"

        if self.can_collect(animal):
            return "COLLECT"

        if self.can_feed(animal):
            return "FEED"

        return "WAIT"