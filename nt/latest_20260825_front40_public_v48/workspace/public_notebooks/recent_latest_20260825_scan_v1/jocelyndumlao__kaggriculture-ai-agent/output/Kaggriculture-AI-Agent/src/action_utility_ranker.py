"""
action_utility_ranker.py

Action Utility Ranker for the Kaggriculture AI Agent.

Ranks actions using their computed utility.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionUtilityRanker:
    """
    Rank actions by utility.
    """

    def rank(
        self,
        actions: list[dict],
    ) -> list[dict]:
        """
        Return actions sorted by utility.
        """

        return sorted(
            actions,
            key=lambda action: action.get(
                "utility",
                0.0,
            ),
            reverse=True,
        )

    # ---------------------------------------------------------

    def best(
        self,
        actions: list[dict],
    ) -> dict | None:
        """
        Return the highest-utility action.
        """

        ranked = self.rank(actions)

        if not ranked:
            return None

        return ranked[0]

    # ---------------------------------------------------------

    def utilities(
        self,
        actions: list[dict],
    ) -> list[float]:
        """
        Return utilities in ranked order.
        """

        return [
            action.get("utility", 0.0)
            for action in self.rank(actions)
        ]