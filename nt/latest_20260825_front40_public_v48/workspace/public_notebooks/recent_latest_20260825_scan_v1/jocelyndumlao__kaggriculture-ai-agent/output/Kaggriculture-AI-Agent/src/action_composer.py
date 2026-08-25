"""
action_composer.py

Action Composer for the Kaggriculture AI Agent.

Combines farmer, hand, and market actions into the
official Kaggriculture action format.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionComposer:
    """
    Creates the final competition action.
    """

    def compose(
        self,
        farmer_actions=None,
        hand_actions=None,
        market_actions=None,
    ) -> dict:
        """
        Compose the final action dictionary.
        """

        if farmer_actions is None:
            farmer_actions = [["PASS"]]

        if hand_actions is None:
            hand_actions = []

        if market_actions is None:
            market_actions = []

        farmer = (
            farmer_actions[0]
            if farmer_actions
            else ["PASS"]
        )

        return {
            "farmer": farmer,
            "hands": hand_actions,
            "market": market_actions,
        }

    # ---------------------------------------------------------

    def empty(self) -> dict:
        """
        Return an empty action.
        """

        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        }