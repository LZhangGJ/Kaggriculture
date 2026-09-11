"""
farm_state_evaluator.py

Farm State Evaluator for the Kaggriculture AI Agent.

Evaluates the overall strength of a farm based on the current
observation.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FarmStateEvaluator:
    """
    Evaluates the current farm state.
    """

    MONEY_WEIGHT = 1.0
    LAND_WEIGHT = 500.0
    CROP_WEIGHT = 50.0
    ANIMAL_WEIGHT = 200.0

    def evaluate(self, observation: dict) -> float:
        """
        Compute an overall farm score.
        """

        farm = observation["farm"]

        score = 0.0

        # Money
        score += farm["money"] * self.MONEY_WEIGHT

        # Land
        score += (
            len(farm["unlocked_quadrants"])
            * self.LAND_WEIGHT
        )

        crop_count = 0
        animal_count = 0

        for row in farm["tiles"]:
            for tile in row:

                if not isinstance(tile, dict):
                    continue

                if tile.get("kind") == "PLANT":
                    crop_count += 1

                elif (
                    tile.get("kind")
                    in (
                        "COOP",
                        "PASTURE",
                    )
                    and tile.get("animal")
                ):
                    animal_count += 1

        score += crop_count * self.CROP_WEIGHT
        score += animal_count * self.ANIMAL_WEIGHT

        return score
    