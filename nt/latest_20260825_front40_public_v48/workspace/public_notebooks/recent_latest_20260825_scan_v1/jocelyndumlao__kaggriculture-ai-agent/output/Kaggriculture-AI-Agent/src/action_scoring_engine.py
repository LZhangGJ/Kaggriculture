"""
action_scoring_engine.py

Action Scoring Engine for the Kaggriculture AI Agent.

Scores candidate actions using evaluation components
and optional market-aware adjustments.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionScoringEngine:
    """
    Scores candidate actions.
    """

    ACTION_BONUSES = {
        "HARVEST": 50.0,
        "PLANT": 25.0,
        "WATER": 10.0,
        "FERTILIZE": 15.0,
        "BUY": 5.0,
        "BUY_SEED": 5.0,
        "SELL": 20.0,
        "FEED": 10.0,
        "COLLECT": 30.0,
        "EXPAND": 40.0,
    }

    # ---------------------------------------------------------

    def score(
        self,
        *,
        action: str,
        farm_score: float = 0.0,
        crop_profit: float = 0.0,
        animal_profit: float = 0.0,
        market_score: float = 0.0,
        game_state: dict | None = None,
    ) -> float:
        """
        Compute a score for a candidate action.
        """

        action = action.upper()

        bonus = self.ACTION_BONUSES.get(
            action,
            0.0,
        )

        score = (
            farm_score
            + crop_profit
            + animal_profit
            + market_score
            + bonus
        )

        # ----------------------------------------------
        # Optional market-aware scoring
        # ----------------------------------------------

        if game_state is not None:

            market = game_state.get(
                "market",
                {},
            )

            prices = market.get(
                "prices",
                {},
            )

            highest_price = max(
                prices.values(),
                default=0,
            )

            if action == "SELL":
                score += highest_price / 10.0

            elif action == "HARVEST":
                score += highest_price / 20.0

            elif action == "BUY_SEED":
                score -= highest_price / 40.0

        return float(score)

    # ---------------------------------------------------------

    def action_bonus(
        self,
        action: str,
    ) -> float:
        """
        Return configured action bonus.
        """

        return float(
            self.ACTION_BONUSES.get(
                action.upper(),
                0.0,
            )
        )

    # ---------------------------------------------------------

    def supported_actions(
        self,
    ) -> list[str]:
        """
        Return supported actions.
        """

        return sorted(
            self.ACTION_BONUSES.keys()
        )

    # ---------------------------------------------------------

    def is_supported(
        self,
        action: str,
    ) -> bool:
        """
        Return True if action is supported.
        """

        return (
            action.upper()
            in self.ACTION_BONUSES
        )