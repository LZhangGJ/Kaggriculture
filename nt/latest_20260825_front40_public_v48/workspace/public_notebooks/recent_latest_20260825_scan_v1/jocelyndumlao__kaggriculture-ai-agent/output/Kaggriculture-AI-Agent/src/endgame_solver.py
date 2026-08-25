"""
endgame_solver.py

Endgame Solver for the Kaggriculture AI Agent.

Optimizes decisions during the final days
of the farming season.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class EndgameSolver:
    """
    Endgame decision helper.
    """

    def __init__(self, end_day: int = 30):
        self.end_day = end_day

    # ---------------------------------------------------------

    def remaining_days(self, current_day: int) -> int:
        """
        Return remaining days in the season.
        """
        return max(0, self.end_day - current_day)

    # ---------------------------------------------------------

    def is_endgame(self, current_day: int) -> bool:
        """
        Determine whether the game
        is in the endgame phase.
        """
        return self.remaining_days(current_day) <= 5

    # ---------------------------------------------------------

    def should_sell_everything(self, current_day: int) -> bool:
        """
        Sell all inventory near season end.
        """
        return self.remaining_days(current_day) <= 2

    # ---------------------------------------------------------

    def should_plant(self, current_day: int, grow_days: int) -> bool:
        """
        Determine whether a crop
        has enough time to mature.
        """
        return grow_days <= self.remaining_days(current_day)

    # ---------------------------------------------------------

    def priority(self, current_day: int) -> str:
        """
        Current strategic focus.
        """

        if self.should_sell_everything(current_day):
            return "LIQUIDATE"

        if self.is_endgame(current_day):
            return "HARVEST"

        return "NORMAL"
    