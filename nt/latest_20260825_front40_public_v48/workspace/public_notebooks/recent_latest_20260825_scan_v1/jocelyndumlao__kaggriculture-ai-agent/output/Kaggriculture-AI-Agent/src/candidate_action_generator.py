"""
candidate_action_generator.py

Candidate Action Generator for the Kaggriculture AI Agent.

Generates candidate actions from the current observation.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class CandidateActionGenerator:
    """
    Generates candidate actions.
    """

    DEFAULT_ACTIONS = (
        "PLANT",
        "HARVEST",
        "WATER",
        "FEED",
        "COLLECT",
        "BUY",
        "SELL",
    )

    def generate(
        self,
        observation: dict,
    ):
        """
        Generate candidate actions.

        Version 1 returns the default action set.
        Future versions will inspect the observation
        and only return legal actions.
        """
        return list(self.DEFAULT_ACTIONS)

    # ---------------------------------------------------------

    def supports(
        self,
        action: str,
    ) -> bool:
        """
        Return True if the generator supports the action.
        """
        return action.upper() in self.DEFAULT_ACTIONS

    # ---------------------------------------------------------

    def action_count(self) -> int:
        """
        Return number of supported actions.
        """
        return len(self.DEFAULT_ACTIONS)

    # ---------------------------------------------------------

    def actions(self):
        """
        Return all supported actions.
        """
        return list(self.DEFAULT_ACTIONS)