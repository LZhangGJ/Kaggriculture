"""
action_diversity_engine.py

Action Diversity Engine for the Kaggriculture AI Agent.

Encourages action diversity by tracking recently
selected actions.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionDiversityEngine:
    """
    Track recently selected actions.
    """

    def __init__(
        self,
        window: int = 5,
    ):

        self.window = window
        self.history = []

    # ---------------------------------------------------------

    def record(
        self,
        action: str,
    ) -> None:
        """
        Record an executed action.
        """

        self.history.append(
            action.upper(),
        )

        if len(self.history) > self.window:
            self.history.pop(0)

    # ---------------------------------------------------------

    def frequency(
        self,
        action: str,
    ) -> int:
        """
        Return recent frequency.
        """

        return self.history.count(
            action.upper(),
        )

    # ---------------------------------------------------------

    def diversity_bonus(
        self,
        action: str,
    ) -> float:
        """
        Less frequent actions receive
        a higher bonus.
        """

        count = self.frequency(
            action,
        )

        return max(
            0.0,
            1.0 - 0.2 * count,
        )

    # ---------------------------------------------------------

    def clear(
        self,
    ) -> None:
        """
        Clear history.
        """

        self.history.clear()