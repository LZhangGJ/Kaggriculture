"""
crop_rotation_planner.py

Crop Rotation Planner for the Kaggriculture AI Agent.

Tracks recently planted crops and recommends
diverse planting strategies.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CropRotationPlanner:
    """
    Maintains a simple crop rotation history.
    """

    def __init__(self):
        self._history = []

    # ---------------------------------------------------------

    def record(
        self,
        crop: str,
    ):
        """
        Record a planted crop.
        """

        self._history.append(crop)

    # ---------------------------------------------------------

    def last_crop(self):
        """
        Return the last planted crop.
        """

        if not self._history:
            return None

        return self._history[-1]

    # ---------------------------------------------------------

    def should_rotate(
        self,
        crop: str,
    ) -> bool:
        """
        Recommend rotation when the same crop
        was planted previously.
        """

        return self.last_crop() == crop

    # ---------------------------------------------------------

    def history(self):
        """
        Return planting history.
        """

        return list(self._history)

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear history.
        """

        self._history.clear()