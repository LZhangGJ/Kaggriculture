"""
crop_lifecycle_analyzer.py

Crop Lifecycle Analyzer for the Kaggriculture AI Agent.

Analyzes crop growth state and determines
whether crops should be planted, watered,
or harvested.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CropLifecycleAnalyzer:
    """
    Analyze crop lifecycle.
    """

    # ---------------------------------------------------------

    def age(
        self,
        *,
        current_day: int,
        planted_day: int,
    ) -> int:
        """
        Return crop age.
        """

        return max(
            0,
            current_day - planted_day,
        )

    # ---------------------------------------------------------

    def ready_to_harvest(
        self,
        *,
        current_day: int,
        planted_day: int,
        mature_day: int,
    ) -> bool:
        """
        Return True if mature.
        """

        return (
            self.age(
                current_day=current_day,
                planted_day=planted_day,
            )
            >= mature_day
        )

    # ---------------------------------------------------------

    def needs_water(
        self,
        tile: dict,
    ) -> bool:
        """
        Return True if watering is needed.
        """

        return not tile.get(
            "watered_today",
            False,
        )

    # ---------------------------------------------------------

    def days_until_harvest(
        self,
        *,
        current_day: int,
        planted_day: int,
        mature_day: int,
    ) -> int:
        """
        Return remaining days.
        """

        remaining = (
            mature_day
            - self.age(
                current_day=current_day,
                planted_day=planted_day,
            )
        )

        return max(
            0,
            remaining,
        )

    # ---------------------------------------------------------

    def status(
        self,
        *,
        current_day: int,
        planted_day: int,
        mature_day: int,
        watered_today: bool,
    ) -> str:
        """
        Return lifecycle status.
        """

        if self.ready_to_harvest(
            current_day=current_day,
            planted_day=planted_day,
            mature_day=mature_day,
        ):
            return "HARVEST"

        if not watered_today:
            return "WATER"

        return "GROWING"