"""
adaptive_strategy_engine.py

Adaptive Strategy Engine for the Kaggriculture AI Agent.

Adjusts action priorities according to the
opponent's inferred strategy.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AdaptiveStrategyEngine:
    """
    Adjust action scores based on opponent strategy.
    """

    STRATEGY_RULES = {
        "UNKNOWN": {},
        "BALANCED": {},
        "ECONOMIC": {
            "SELL": 20,
            "HARVEST": 15,
        },
        "AGGRESSIVE": {
            "EXPAND": 20,
            "PLANT": 10,
        },
        "EXPANSION": {
            "BUY_SEED": 15,
            "PLANT": 15,
        },
    }

    # ---------------------------------------------------------

    def adjustment(
        self,
        action: str,
        strategy: str,
    ) -> float:
        """
        Return adjustment for an action under
        a given opponent strategy.
        """

        strategy = strategy.upper()
        action = action.upper()

        return float(
            self.STRATEGY_RULES.get(
                strategy,
                {},
            ).get(
                action,
                0,
            )
        )

    # ---------------------------------------------------------

    def apply(
        self,
        score: float,
        *,
        action: str,
        strategy: str,
    ) -> float:
        """
        Return adjusted score.
        """

        return float(
            score +
            self.adjustment(
                action,
                strategy,
            )
        )

    # ---------------------------------------------------------

    def supported_strategies(
        self,
    ) -> list[str]:
        """
        Return supported strategies.
        """

        return sorted(
            self.STRATEGY_RULES.keys()
        )