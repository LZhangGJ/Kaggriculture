"""
crop_scheduler_engine.py

Crop Scheduler Engine for the Kaggriculture AI Agent.

Schedules planting, watering, and harvesting
tasks based on crop growth.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CropSchedulerEngine:
    """
    Manage crop schedules.
    """

    def __init__(self):

        self.crops = {}

    # ---------------------------------------------------------

    def plant(
        self,
        position: tuple[int, int],
        crop: str,
        current_day: int,
        growth_days: int,
    ) -> None:
        """
        Register a planted crop.
        """

        self.crops[position] = {
            "crop": crop.upper(),
            "plant_day": current_day,
            "growth_days": growth_days,
        }

    # ---------------------------------------------------------

    def ready_to_harvest(
        self,
        position: tuple[int, int],
        current_day: int,
    ) -> bool:
        """
        Return True if the crop is mature.
        """

        crop = self.crops.get(position)

        if crop is None:
            return False

        return (
            current_day - crop["plant_day"]
            >= crop["growth_days"]
        )

    # ---------------------------------------------------------

    def watering_due(
        self,
        current_day: int,
    ) -> list[tuple[int, int]]:
        """
        Return crops needing watering today.

        (Currently assumes all planted crops
        require watering each day.)
        """

        _ = current_day

        return list(
            self.crops.keys()
        )

    # ---------------------------------------------------------

    def harvest(
        self,
        position: tuple[int, int],
    ) -> bool:
        """
        Remove a harvested crop.
        """

        if position not in self.crops:
            return False

        del self.crops[position]

        return True

    # ---------------------------------------------------------

    def crop_count(
        self,
    ) -> int:
        """
        Return the number of active crops.
        """

        return len(
            self.crops
        )