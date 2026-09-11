"""
action_priority_engine.py

Action Priority Engine for the Kaggriculture AI Agent.

Ranks candidate actions according to their
priority score.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionPriorityEngine:
    """
    Prioritizes actions.
    """

    # ---------------------------------------------------------

    def rank(
        self,
        actions: list[dict],
    ) -> list[dict]:
        """
        Return actions sorted by priority.
        """

        return sorted(
            actions,
            key=lambda action: action.get(
                "priority",
                0,
            ),
            reverse=True,
        )

    # ---------------------------------------------------------

    def best_action(
        self,
        actions: list[dict],
    ) -> dict | None:
        """
        Return the highest-priority action.
        """

        ranked = self.rank(actions)

        if not ranked:
            return None

        return ranked[0]

    # ---------------------------------------------------------

    def priorities(
        self,
        actions: list[dict],
    ) -> list[int]:
        """
        Return priority values.
        """

        return [
            action.get("priority", 0)
            for action in self.rank(actions)
        ]

    # ---------------------------------------------------------

    def highest_priority(
        self,
        actions: list[dict],
    ) -> int:
        """
        Return the highest priority value.
        """

        if not actions:
            return 0

        return max(
            action.get(
                "priority",
                0,
            )
            for action in actions
        )