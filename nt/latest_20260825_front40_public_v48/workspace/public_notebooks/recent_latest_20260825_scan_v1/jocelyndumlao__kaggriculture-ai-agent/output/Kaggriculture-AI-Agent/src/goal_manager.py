"""
goal_manager.py

Long-term goal management for the Kaggriculture AI Agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class GoalManager:
    """
    Maintains the current strategic goal of the AI.
    """

    def __init__(self):
        self.goal = "MAKE_PROFIT"

    # ---------------------------------------------------------

    def current_goal(self) -> str:
        """
        Return the active goal.
        """

        return self.goal

    # ---------------------------------------------------------

    def update(
        self,
        money: int,
        day: int,
    ):
        """
        Update the long-term goal.
        """

        if day >= 28:
            self.goal = "FINAL_PROFIT"

        elif money < 500:
            self.goal = "SAVE_MONEY"

        elif money > 8000:
            self.goal = "EXPAND_FARM"

        else:
            self.goal = "MAKE_PROFIT"

    # ---------------------------------------------------------

    def is_goal(self, goal: str) -> bool:
        """
        Check whether the supplied goal is active.
        """

        return self.goal == goal

    # ---------------------------------------------------------

    def reset(self):
        """
        Reset to the default goal.
        """

        self.goal = "MAKE_PROFIT"