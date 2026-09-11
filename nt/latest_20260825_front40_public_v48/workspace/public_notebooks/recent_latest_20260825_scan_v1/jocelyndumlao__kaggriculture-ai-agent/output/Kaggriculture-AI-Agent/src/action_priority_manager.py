"""
action_priority_manager.py

Action Priority Manager for the Kaggriculture AI Agent.

Assigns priorities to actions before
decision making.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionPriorityManager:
    """
    Manage action priorities.
    """

    DEFAULT_PRIORITY = 0

    _PRIORITIES = {
        "HARVEST": 100,
        "WATER": 90,
        "FEED": 85,
        "CARE": 80,
        "PLANT": 70,
        "SELL": 60,
        "BUY_SEED": 50,
        "BUY_LAND": 40,
        "MOVE": 20,
        "PASS": 0,
    }

    # ---------------------------------------------------------

    def priority(
        self,
        action: str,
    ) -> int:
        """
        Return priority for an action.
        """

        return self._PRIORITIES.get(
            action,
            self.DEFAULT_PRIORITY,
        )

    # ---------------------------------------------------------

    def higher_priority(
        self,
        first: str,
        second: str,
    ) -> str:
        """
        Return the higher-priority action.
        """

        if self.priority(first) >= self.priority(second):
            return first

        return second

    # ---------------------------------------------------------

    def sort_actions(
        self,
        actions: list[str],
    ) -> list[str]:
        """
        Sort actions by descending priority.
        """

        return sorted(
            actions,
            key=self.priority,
            reverse=True,
        )

    # ---------------------------------------------------------

    def is_critical(
        self,
        action: str,
    ) -> bool:
        """
        Return True for critical actions.
        """

        return self.priority(action) >= 90

    # ---------------------------------------------------------

    def available_priorities(
        self,
    ) -> dict[str, int]:
        """
        Return priority table.
        """

        return dict(self._PRIORITIES)